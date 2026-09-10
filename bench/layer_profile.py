"""Per-layer-type time breakdown for a qwen3_5 model on MLX.

Modules are wrapped with synchronous timing (call, mx.eval, wall clock) at
three levels. Each level runs on its own so the sync overhead of one level
does not inflate the numbers of another:
  layers  embed, each decoder layer (gdn or attn), final norm, lm_head
  blocks  input_layernorm, linear_attn or self_attn, post_attention_layernorm, mlp
  ops     submodules of the mixers and the mlp, plus gated_delta_update and sdpa
An unwrapped run gives the reference step time and the profiling overhead.

Prefill mirrors generate_step: the first N-1 prompt tokens are processed and
only the cache is evaluated (no final norm, no lm_head). Decode is M single
token steps with fixed token ids and no sampler.

    python bench/layer_profile.py --model <path> --prompt-tokens 512 \\
        --decode-tokens 32 --json out.json [--gputrace DIR]
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# Metal capture must be enabled before the Metal device is created.
if "--gputrace" in sys.argv:
    os.environ.setdefault("MTL_CAPTURE_ENABLED", "1")

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import load
from mlx_lm.models import qwen3_5, qwen3_next
from mlx_lm.models.cache import make_prompt_cache

LEVELS = ("layers", "blocks", "ops")


def arrays_in(obj):
    """Collect the arrays in call arguments, including cache states."""
    if isinstance(obj, mx.array):
        return [obj]
    if isinstance(obj, nn.Module):
        return []
    if isinstance(obj, (list, tuple)):
        return [a for o in obj for a in arrays_in(o)]
    if isinstance(obj, dict):
        return arrays_in(list(obj.values()))
    if hasattr(obj, "state"):
        return arrays_in(obj.state)
    return []


class Timer:
    """Records the wall time of module or function calls, forcing evaluation."""

    def __init__(self):
        self.times = defaultdict(list)
        self.restores = []

    def timed(self, fn, name):
        times = self.times

        def call(*args, **kwargs):
            mx.eval(arrays_in(args), arrays_in(kwargs))
            tic = time.perf_counter()
            out = fn(*args, **kwargs)
            mx.eval(out, arrays_in(args), arrays_in(kwargs))
            times[name].append(time.perf_counter() - tic)
            return out

        return call

    def wrap_module(self, module, name, method="__call__"):
        cls = type(module)
        call = self.timed(getattr(cls, method), name)
        module.__class__ = type(cls.__name__, (cls,), {method: call})
        self.restores.append(lambda: setattr(module, "__class__", cls))

    def wrap_function(self, namespace, attr, name):
        fn = getattr(namespace, attr)
        setattr(namespace, attr, self.timed(fn, name))
        self.restores.append(lambda: setattr(namespace, attr, fn))

    def unwrap(self):
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()


def kind(layer):
    return "gdn" if layer.is_linear else "attn"


def mixer(layer):
    if layer.is_linear:
        return "linear_attn", layer.linear_attn
    return "self_attn", layer.self_attn


def wrap_layers(timer, model):
    for layer in model.layers:
        timer.wrap_module(layer, f"{kind(layer)}_layer")


def wrap_blocks(timer, model):
    for layer in model.layers:
        k = kind(layer)
        name, mod = mixer(layer)
        timer.wrap_module(layer.input_layernorm, f"{k}.input_layernorm")
        timer.wrap_module(mod, f"{k}.{name}")
        timer.wrap_module(
            layer.post_attention_layernorm, f"{k}.post_attention_layernorm"
        )
        timer.wrap_module(layer.mlp, f"{k}.mlp")


def wrap_ops(timer, model):
    for layer in model.layers:
        k = kind(layer)
        name, mod = mixer(layer)
        for child, sub in mod.children().items():
            timer.wrap_module(sub, f"{k}.{name}.{child}")
        for child, sub in layer.mlp.children().items():
            timer.wrap_module(sub, f"{k}.mlp.{child}")
    timer.wrap_function(
        qwen3_5, "gated_delta_update", "gdn.linear_attn.gated_delta_update"
    )
    timer.wrap_function(
        qwen3_next, "scaled_dot_product_attention", "attn.self_attn.sdpa"
    )


WRAPPERS = {"layers": wrap_layers, "blocks": wrap_blocks, "ops": wrap_ops}


def wrap(timer, model, level, head):
    timer.wrap_module(model.model.embed_tokens, "embed")
    WRAPPERS[level](timer, model)
    if head:
        text = model.language_model
        timer.wrap_module(model.model.norm, "final_norm")
        if text.args.tie_word_embeddings:
            timer.wrap_module(model.model.embed_tokens, "lm_head", "as_linear")
        else:
            timer.wrap_module(text.lm_head, "lm_head")


def prefill(model, cache, prompt):
    model(prompt[:-1][None], cache=cache)
    mx.eval([c.state for c in cache])


def decode_step(model, cache, tok):
    mx.eval(model(tok, cache=cache))


def summarize(times, steps, step_ms):
    rows = []
    for name, ts in times.items():
        count = len(ts) / steps
        total_ms = sum(ts) * 1e3 / steps
        rows.append(
            {
                "component": name,
                "count": count,
                "total_ms": total_ms,
                "mean_ms": total_ms / count,
                "share": total_ms / step_ms,
            }
        )
    return rows


def run(model, prompt, tokens, level=None):
    """Prefill then decode with modules wrapped at `level` (None: unwrapped)."""
    cache = make_prompt_cache(model)
    phases = {}

    timer = Timer()
    if level:
        wrap(timer, model, level, head=False)
    tic = time.perf_counter()
    prefill(model, cache, prompt)
    step_ms = (time.perf_counter() - tic) * 1e3
    phases["prefill"] = {"step_ms": step_ms, "rows": summarize(timer.times, 1, step_ms)}
    timer.unwrap()

    timer = Timer()
    if level:
        wrap(timer, model, level, head=True)
    tic = time.perf_counter()
    for tok in tokens:
        decode_step(model, cache, tok)
    step_ms = (time.perf_counter() - tic) * 1e3 / len(tokens)
    phases["decode"] = {
        "step_ms": step_ms,
        "rows": summarize(timer.times, len(tokens), step_ms),
    }
    timer.unwrap()
    return phases


def sync_floor(n=200):
    """Mean wall time of a trivial timed call: the floor of every measurement."""
    timer = Timer()
    call = timer.timed(lambda a: a + 1, "floor")
    x = mx.ones((16,))
    for _ in range(n):
        call(x)
    return sum(timer.times["floor"]) * 1e3 / n


def print_table(title, phase):
    rows, step_ms = phase["rows"], phase["step_ms"]
    print(f"\n{title}: step {step_ms:.3f} ms")
    print(
        f"{'component':44} {'n/step':>6} {'total ms':>10} {'mean ms':>10} {'share':>7}"
    )
    for r in rows:
        print(
            f"{r['component']:44} {r['count']:6.0f} {r['total_ms']:10.3f} "
            f"{r['mean_ms']:10.3f} {100 * r['share']:6.1f}%"
        )
    covered = sum(r["total_ms"] for r in rows)
    print(
        f"{'sum of rows':44} {'':6} {covered:10.3f} {'':10} {100 * covered / step_ms:6.1f}%"
    )


def report(results, n_prompt, n_decode):
    ref = results["unwrapped"]
    print(
        f"\nprefill ({n_prompt - 1} tokens): unwrapped {ref['prefill']['step_ms']:.1f} ms "
        f"= {(n_prompt - 1) / ref['prefill']['step_ms'] * 1e3:.0f} tok/s"
    )
    print(
        f"decode ({n_decode} steps): unwrapped {ref['decode']['step_ms']:.3f} ms/token "
        f"= {1e3 / ref['decode']['step_ms']:.1f} tok/s"
    )
    for level in LEVELS:
        pre, dec = results[level]["prefill"], results[level]["decode"]
        print(
            f"  wrapped[{level}]: prefill {pre['step_ms']:.1f} ms "
            f"({pre['step_ms'] / ref['prefill']['step_ms']:.2f}x), "
            f"decode {dec['step_ms']:.3f} ms/token = {1e3 / dec['step_ms']:.1f} tok/s "
            f"({dec['step_ms'] / ref['decode']['step_ms']:.2f}x)"
        )
    for level in LEVELS:
        for phase in ("prefill", "decode"):
            print_table(f"{phase} / {level}", results[level][phase])


def capture(path, fn):
    shutil.rmtree(path, ignore_errors=True)
    try:
        mx.metal.start_capture(str(path))
    except Exception as e:
        print(f"gputrace: skipped {path.name}: {e}")
        return
    try:
        fn()
    finally:
        mx.metal.stop_capture()
    print(f"gputrace: wrote {path}")


def capture_traces(model, prompt, tokens, out_dir):
    if not hasattr(mx, "metal"):
        print("gputrace: no Metal backend, skipped")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = make_prompt_cache(model)
    prefill(model, cache, prompt)
    for tok in tokens[:2]:
        decode_step(model, cache, tok)
    tok = tokens[-1]
    capture(out_dir / "decode_step.gputrace", lambda: decode_step(model, cache, tok))
    x = model.model.embed_tokens(tok)
    mx.eval(x)
    for name, is_linear in (("gdn_layer", True), ("attn_layer", False)):
        i = next(i for i, l in enumerate(model.layers) if l.is_linear == is_linear)
        layer, c = model.layers[i], cache[i]
        capture(
            out_dir / f"{name}.gputrace", lambda: mx.eval(layer(x, mask=None, cache=c))
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument(
        "--warmup", type=int, default=4, help="untimed decode steps before measuring"
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--gputrace", type=Path, help="write Metal captures into this directory"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    model, _ = load(args.model)
    vocab = model.language_model.args.vocab_size
    mx.random.seed(args.seed)
    prompt = mx.random.randint(0, vocab, (args.prompt_tokens,))
    ids = mx.random.randint(0, vocab, (args.decode_tokens - 1,)).tolist()
    tokens = [prompt[-1:][None]] + [mx.array([[t]]) for t in ids]
    mx.eval(prompt, tokens)

    cache = make_prompt_cache(model)
    prefill(model, cache, prompt)
    for tok in tokens[: args.warmup]:
        decode_step(model, cache, tok)

    results = {"unwrapped": run(model, prompt, tokens)}
    for level in LEVELS:
        results[level] = run(model, prompt, tokens, level)
    floor_ms = sync_floor()

    print(f"model: {args.model}")
    print(
        f"layers: {len(model.layers)} ({sum(l.is_linear for l in model.layers)} gdn, "
        f"{sum(not l.is_linear for l in model.layers)} attn), "
        f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB, "
        f"sync floor per timed call {floor_ms * 1e3:.0f} us"
    )
    report(results, args.prompt_tokens, args.decode_tokens)

    if args.gputrace:
        capture_traces(model, prompt, tokens, args.gputrace)
    if args.json:
        out = {
            "model": str(args.model),
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "sync_floor_ms": floor_ms,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "results": results,
        }
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
