"""Where the time goes inside an MTP speculative-decoding cycle.

Re-runs the loop of ``speculative_generate_step`` for an MTP head with an
``mx.eval`` and a wall clock around every phase, so the per-cycle cost splits
into draft steps, the verify forward, host readback, cache rollback and the
Python remainder. Greedy sampling only.

    python bench/mtp_profile.py --model <target> --draft-model <sidecar|bundled> \\
        --num-draft-tokens 2 --prompt "text" --max-tokens 128 [--verify-layers] [--json out.json]
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from layer_profile import Timer, kind

from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, make_prompt_cache, trim_prompt_cache
from mlx_lm.models.qwen3_5_mtp import load_bundled


class Phase:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def __call__(self, name, fn, *outs):
        tic = time.perf_counter()
        result = fn()
        mx.eval(result if outs == () else outs)
        self.times[name] += time.perf_counter() - tic
        self.counts[name] += 1
        return result


def argmax_tokens(logits):
    return mx.argmax(logits[0], axis=-1).astype(mx.uint32)


def plain_decode(model, prompt, n):
    cache = make_prompt_cache(model)
    model(prompt[None, :-1], cache=cache)
    mx.eval([c.state for c in cache])
    y = prompt[-1:]
    times = []
    for _ in range(n):
        tic = time.perf_counter()
        y = argmax_tokens(model(y[None], cache=cache))
        mx.eval(y)
        times.append(time.perf_counter() - tic)
    return sum(times[4:]) / len(times[4:])


def run(model, draft, prompt, k, max_tokens, timer=None):
    draft.bind(model)
    cache, draft_cache = make_prompt_cache(model), draft.make_cache()
    phase = Phase()
    # Prefill target with hidden states, seed the drafter with (token, previous hidden)
    y = prompt
    _, h = model(y[None, :-1], cache=cache, return_hidden=True)
    draft(y[None, 1:-1], h[:, :-1], cache=draft_cache)
    hidden = h[:, -1:]
    mx.eval([c.state for c in cache], [c.state for c in draft_cache], hidden)
    y = draft_y = prompt[-1:]
    for c in cache:
        if isinstance(c, ArraysCache):
            c.keep_states = True

    accepted_at = defaultdict(int)
    produced, cycles, tokens_out = 0, 0, []
    while produced < max_tokens:
        cycle_tic = time.perf_counter()
        drafts, hd = [], hidden
        for i in range(k):
            inp = draft_y[None] if i == 0 else drafts[-1][None]
            logits, hd = phase(f"draft_{i}", lambda: draft(inp, hd, cache=draft_cache))
            hd = hd[:, -1:]
            drafts.append(argmax_tokens(logits[:, -1:]))
            mx.eval(drafts[-1])
        draft_tokens = mx.concatenate(drafts) if drafts else mx.array([], mx.uint32)
        inputs = mx.concatenate([y, draft_tokens])

        logits, hidden_out = phase(
            "verify", lambda: model(inputs[None], cache=cache, return_hidden=True)
        )
        tokens = phase("sample", lambda: argmax_tokens(logits[:, -(k + 1) :]))
        d, t = phase("readback", lambda: (draft_tokens.tolist(), tokens.tolist()))
        n = 0
        while n < k and t[n] == d[n]:
            n += 1
        for i in range(n):
            accepted_at[i] += 1
        got = t[: n + 1]
        tokens_out += got
        produced += len(got)
        cycles += 1

        y = draft_y = mx.array(t[n : n + 1], mx.uint32)
        if n == k:
            draft_y = mx.concatenate([mx.array(d[-1:], mx.uint32), y])
        hidden = hidden_out[:, n + 1 - draft_y.size : n + 1]
        phase(
            "rollback",
            lambda: (
                trim_prompt_cache(cache, k - n),
                trim_prompt_cache(draft_cache, max(k - n - 1, 0)),
            ),
            [],
        )
        phase.times["cycle"] += time.perf_counter() - cycle_tic
        phase.counts["cycle"] += 1
    return phase, accepted_at, cycles, produced, tokens_out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model", required=True, help="sidecar path or 'bundled'")
    ap.add_argument("--num-draft-tokens", "-k", type=int, default=2)
    ap.add_argument(
        "--prompt",
        default="Write a quicksort in Python with type hints and a short explanation.",
    )
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        help="random prompt of this length instead of --prompt",
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--verify-layers",
        action="store_true",
        help="per-layer-type time inside the verify forward",
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    model, tok = load(args.model)
    draft = (
        load_bundled(args.model)
        if args.draft_model == "bundled"
        else load(args.draft_model)[0]
    )
    if args.prompt_tokens:
        vocab = model.language_model.args.vocab_size
        prompt = mx.random.randint(0, vocab, (args.prompt_tokens,))
    else:
        prompt = mx.array(
            tok.apply_chat_template(
                [{"role": "user", "content": args.prompt}], add_generation_prompt=True
            )
        )
    mx.eval(prompt)

    run(model, draft, prompt, args.num_draft_tokens, 16)  # warm up
    plain_ms = plain_decode(model, prompt, 24) * 1e3
    timer = Timer() if args.verify_layers else None
    if timer:
        for layer in model.layers:
            timer.wrap_module(layer, f"{kind(layer)}_layer")
        if not model.language_model.args.tie_word_embeddings:
            timer.wrap_module(model.language_model.lm_head, "lm_head")
    phase, accepted_at, cycles, produced, out = run(
        model, draft, prompt, args.num_draft_tokens, args.max_tokens, timer
    )
    if timer:
        timer.unwrap()

    k = args.num_draft_tokens
    cycle_ms = phase.times["cycle"] * 1e3 / cycles
    rows = []
    for name in [f"draft_{i}" for i in range(k)] + [
        "verify",
        "sample",
        "readback",
        "rollback",
    ]:
        ms = phase.times[name] * 1e3 / cycles
        rows.append((name, ms, ms / cycle_ms))
    rest = cycle_ms - sum(r[1] for r in rows)
    rows.append(("python remainder", rest, rest / cycle_ms))
    print(
        f"model {args.model}, draft {args.draft_model}, k={k}, {cycles} cycles, {produced} tokens"
    )
    print(f"plain decode {plain_ms:.2f} ms/token = {1e3 / plain_ms:.1f} tok/s")
    print(
        f"MTP cycle {cycle_ms:.2f} ms, {produced / cycles:.2f} tok/cycle = {produced / cycles / cycle_ms * 1e3:.1f} tok/s "
        f"({produced / cycles / cycle_ms * plain_ms:.2f}x plain)"
    )
    print(f"{'phase':18} {'ms/cycle':>9} {'share':>7}")
    for name, ms, share in rows:
        print(f"{name:18} {ms:9.2f} {100 * share:6.1f}%")
    print(
        "acceptance: "
        + ", ".join(
            f"P(d{i + 1} ok | prev ok)={accepted_at[i] / (accepted_at[i - 1] if i else cycles):.2f}"
            for i in range(k)
        )
    )
    if timer:
        print(f"\nverify forward by layer type (S={k + 1}, sync per layer):")
        for name, ts in timer.times.items():
            print(
                f"  {name:12} n={len(ts) / cycles:5.1f}  {sum(ts) * 1e3 / cycles:8.2f} ms/cycle"
            )
    print("\n" + tok.decode(out[:48]).replace("\n", " ")[:200])
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "draft": args.draft_model,
                    "k": k,
                    "cycles": cycles,
                    "tokens": produced,
                    "plain_ms": plain_ms,
                    "cycle_ms": cycle_ms,
                    "phases": {n: ms for n, ms, _ in rows},
                    "accepted_at": dict(accepted_at),
                    "verify_layers": (
                        {n: sum(ts) * 1e3 / cycles for n, ts in timer.times.items()}
                        if timer
                        else None
                    ),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
