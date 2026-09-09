"""Where the time goes inside an MTP speculative-decoding cycle.

Re-runs the loop of ``speculative_generate_step`` for an MTP head with an
``mx.eval`` and a wall clock around every phase, so the per-cycle cost splits
into draft steps, the verify forward, host readback, cache rollback and the
Python remainder. Greedy sampling only. The draft stop rule, the candidate-set
head, its fallback and the sibling rows follow ``speculative_generate_step``;
``--check-candidates`` also runs the full head on every candidate draft and
counts the misses.

    python bench/mtp_profile.py --model <target> --draft-model <sidecar|bundled> \\
        --num-draft-tokens 2 --prompt "text" --max-tokens 128 [--verify-layers] [--json out.json]
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
from layer_profile import Timer, kind

from mlx_lm import load
from mlx_lm.generate import sibling_rows
from mlx_lm.models.cache import ArraysCache, make_prompt_cache, trim_prompt_cache
from mlx_lm.models.qwen3_5_mtp import load_bundled


class Phase:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def __call__(self, name, fn, *outs):
        tic = time.perf_counter()
        result = fn()
        if outs == () and hasattr(result, "rows"):
            outs = (result.first, *result.rows)
        mx.eval(result if outs == () else outs)
        self.times[name] += time.perf_counter() - tic
        self.counts[name] += 1
        return result


def argmax_tokens(logits):
    return mx.argmax(logits[0], axis=-1).astype(mx.uint32)


def warm_shapes(model, prompt, max_rows):
    """Compile the verify kernels of every row count before the timed runs."""
    cache = make_prompt_cache(model)
    model(prompt[None, :-1], cache=cache)
    mx.eval([c.state for c in cache])
    for c in cache:
        if isinstance(c, ArraysCache):
            c.keep_states = True
    for s in range(1, max_rows + 1):
        mx.eval(model(mx.tile(prompt[-1:], s)[None], cache=cache))
        trim_prompt_cache(cache, s)


def plain_decode(model, prompt, n):
    """Mean step time after warm-up and the greedy tokens."""
    cache = make_prompt_cache(model)
    model(prompt[None, :-1], cache=cache)
    mx.eval([c.state for c in cache])
    y = prompt[-1:]
    times, out = [], []
    for _ in range(n):
        tic = time.perf_counter()
        y = argmax_tokens(model(y[None], cache=cache))
        mx.eval(y)
        times.append(time.perf_counter() - tic)
        out.append(y.item())
    return sum(times[4:]) / len(times[4:]), out


def run(
    model,
    draft,
    prompt,
    k,
    max_tokens,
    timer=None,
    stop=0.0,
    candidates=0,
    fallback=0.0,
    check=False,
    siblings=False,
    points=None,
):
    draft.bind(model)
    cache, draft_cache = make_prompt_cache(model), draft.make_cache()
    phase = Phase()
    stats = Counter()
    # Prefill target with hidden states, seed the drafter with (token, previous hidden)
    y = prompt
    logits, h = model(y[None, :-1], cache=cache, return_hidden=True)
    draft(y[None, 1:-1], h[:, :-1], cache=draft_cache)
    hidden = h[:, -1:]
    cands = None
    if candidates:
        cands = draft.candidates(candidates)
        cands.extend(prompt)
        cands.observe(logits[0, -64:])
    mx.eval([c.state for c in cache], [c.state for c in draft_cache], hidden)
    y = draft_y = prompt[-1:]
    recent = ()
    for c in cache:
        if isinstance(c, ArraysCache):
            c.keep_states = True

    accepted_at = defaultdict(int)
    produced, cycles, tokens_out, per_cycle = 0, 0, [], []
    while produced < max_tokens:
        cycle_tic = time.perf_counter()
        head = None
        if cands is not None:
            head = phase("cand_build", lambda: cands.make_head(*recent))
            stats["cand_size"] += head.ids.size
        drafts, alts, hd, ps, q, i = [], [], hidden, [], 1.0, 0
        fused = hasattr(draft, "sample") and not (check or siblings)
        while i < k:
            inp = draft_y[None] if i == 0 else drafts[-1][None]
            if fused:
                # The generation loop's path: one kernel samples the draft after the head
                tok2, stat, hd_new = phase(
                    f"draft_{i}", lambda: draft.sample(inp, hd, cache=draft_cache, head=head)
                )
                hd_new = hd_new[:, -1:]
                tok, p, margin = tok2[:1], stat[:1], stat[1:].item()
                if head is not None:
                    stats["cand_steps"] += 1
                    if fallback and margin < fallback:
                        stats["fallback"] += 1
                        full = phase("fallback", lambda: draft.lm_head(hd_new)[0, -1])
                        tok = mx.argmax(full, keepdims=True).astype(mx.uint32)
                        p = mx.max(mx.softmax(full.astype(mx.float32)))
                mx.eval(tok, p)
                drafts.append(tok)
                ps.append(p.item())
                hd = hd_new
                if 0 < i < k - 1:
                    q *= ps[i - 1]
                    if q < stop:
                        trim_prompt_cache(draft_cache, 1)
                        drafts.pop()
                        break
                i += 1
                continue
            logits, hd_new = phase(
                f"draft_{i}", lambda: draft(inp, hd, cache=draft_cache, head=head)
            )
            hd_new = hd_new[:, -1:]
            l = logits[0, -1].astype(mx.float32)
            tok = mx.argmax(l, keepdims=True).astype(mx.uint32)
            p = mx.max(mx.softmax(l))
            if head is not None:
                tok = head.ids[tok]
                margin = mx.abs(mx.diff(mx.topk(l, 2))).item()
                stats["cand_steps"] += 1
                if check:
                    full = phase("check", lambda: draft.lm_head(hd_new)[0, -1])
                    stats["miss"] += int(mx.argmax(full).item() != tok.item())
                    p_full = mx.max(mx.softmax(full.astype(mx.float32))).item()
                    stats["p_diff"] += p.item() - p_full
                    stats["p_lower"] += int(p.item() < p_full - 1e-4)
                if fallback and margin < fallback:
                    stats["fallback"] += 1
                    full = phase("fallback", lambda: draft.lm_head(hd_new)[0, -1])
                    tok = mx.argmax(full, keepdims=True).astype(mx.uint32)
                    p = mx.max(mx.softmax(full.astype(mx.float32)))
            if siblings:
                # The second choice is the sibling row of this position
                src = full if (fallback and margin < fallback) else l
                alt = mx.argmax(mx.put_along_axis(src, mx.argmax(src, keepdims=True), mx.array(-mx.inf), -1), keepdims=True)
                if head is not None and src is l:
                    alt = head.ids[alt]
                alts.append(alt.astype(mx.uint32))
            mx.eval(tok, p, *alts[-1:])
            drafts.append(tok)
            ps.append(p.item())
            hd = hd_new
            # The stop rule of the generation loop: the previous draft's p is read one late
            if 0 < i < k - 1:
                q *= ps[i - 1]
                if q < stop:
                    trim_prompt_cache(draft_cache, 1)
                    drafts.pop()
                    alts = alts[: len(drafts)]
                    break
            i += 1
        kd = len(drafts)
        n_sib = sibling_rows(kd) if siblings else 0
        chain = kd + 1 if n_sib else None
        draft_tokens = mx.concatenate(drafts) if drafts else mx.array([], mx.uint32)
        inputs = mx.concatenate([y, draft_tokens] + ([mx.concatenate(alts[:n_sib])] if chain else []))
        rows = inputs.size

        logits, hidden_out = phase(
            "verify",
            lambda: model(inputs[None], cache=cache, return_hidden=True, chain=chain),
        )
        tokens = phase("sample", lambda: argmax_tokens(logits[:, -rows:]))
        if points is not None and drafts:
            # Per draft position: the target's logprob of the draft, its top logprob and its entropy
            lp = logits[0, -rows:-1].astype(mx.float32)
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            at = mx.take_along_axis(lp, draft_tokens[:, None], axis=-1)[:, 0]
            ent = -mx.sum(mx.exp(lp) * lp, axis=-1)
            phase("points", lambda: points.append((at, lp.max(axis=-1), ent, cycles)), [])
        if cands is not None:
            phase("cand_score", lambda: cands.observe(logits[0, -rows:]), [])
            mx.eval(cands.score)
        d, t = phase("readback", lambda: (draft_tokens.tolist(), tokens.tolist()))
        n = 0
        while n < kd and t[n] == d[n]:
            n += 1
        for i in range(n):
            accepted_at[i] += 1
        sib = bool(chain) and n < n_sib and alts[n].item() == t[n]
        stats["sib_tries"] += int(bool(chain) and n < n_sib)
        stats["sib_rows"] += n_sib
        stats["sib_hits"] += int(sib)
        got = t[: n + 1] + ([t[chain + n]] if sib else [])
        tokens_out += got
        produced += len(got)
        cycles += 1
        stats["drafted"] += kd
        per_cycle.append((n, kd))

        y = draft_y = mx.array(got[-1:], mx.uint32)
        if n == kd:
            draft_y = mx.concatenate([mx.array(d[-1:], mx.uint32), y])
            hidden = hidden_out[:, n - 1 : n + 1]
        elif sib:
            draft_y = mx.concatenate([mx.array(t[n : n + 1], mx.uint32), y])
            hidden = mx.concatenate(
                [hidden_out[:, n : n + 1], hidden_out[:, chain + n : chain + n + 1]], axis=1
            )
        else:
            hidden = hidden_out[:, n : n + 1]
        if cands is not None:
            cands.extend(mx.array(got, mx.uint32))
            recent = (draft_tokens, tokens)
        keep = 1 + n + int(sib)

        def rollback():
            if sib:
                for c in cache:
                    if isinstance(c, ArraysCache):
                        c.stage(n + 1, chain + n)
                    else:
                        c.move_row(rows - chain - n, rows - n - 1)
            trim_prompt_cache(cache, rows - keep)
            trim_prompt_cache(draft_cache, max(kd - n - 1, 0))

        phase("rollback", rollback, [])
        phase.times["cycle"] += time.perf_counter() - cycle_tic
        phase.counts["cycle"] += 1
    run.stats = stats
    run.per_cycle = per_cycle
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
    ap.add_argument("--draft-stop-prob", type=float, default=0.0)
    ap.add_argument("--draft-candidates", type=int, default=0)
    ap.add_argument("--draft-fallback-margin", type=float, default=0.0)
    ap.add_argument("--draft-siblings", action="store_true")
    ap.add_argument(
        "--check-candidates",
        action="store_true",
        help="run the full head on every candidate draft and count the misses",
    )
    ap.add_argument(
        "--verify-layers",
        action="store_true",
        help="per-layer-type time inside the verify forward",
    )
    ap.add_argument("--json", type=Path)
    ap.add_argument("--points", type=Path, help="save (logprob of draft, top logprob, entropy, cycle) per draft position")
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

    opts = dict(
        stop=args.draft_stop_prob,
        candidates=args.draft_candidates,
        fallback=args.draft_fallback_margin,
        check=args.check_candidates,
        siblings=args.draft_siblings,
    )
    points = [] if args.points else None
    k = args.num_draft_tokens
    warm_shapes(model, prompt, 2 * k + 1 if args.draft_siblings else k + 1)
    run(model, draft, prompt, args.num_draft_tokens, 16, **opts)  # warm up
    plain_ms, plain_out = plain_decode(model, prompt, max(24, args.max_tokens))
    plain_ms *= 1e3
    timer = Timer() if args.verify_layers else None
    if timer:
        for layer in model.layers:
            timer.wrap_module(layer, f"{kind(layer)}_layer")
        if not model.language_model.args.tie_word_embeddings:
            timer.wrap_module(model.language_model.lm_head, "lm_head")
    phase, accepted_at, cycles, produced, out = run(
        model, draft, prompt, args.num_draft_tokens, args.max_tokens, timer, points=points, **opts
    )
    if points:
        mx.eval([x for p in points for x in p[:3]])
        cols = [mx.concatenate([p[i] for p in points]) for i in range(3)]
        cycle = mx.concatenate([mx.full(p[0].shape, p[3]) for p in points])
        mx.savez(str(args.points), draft=cols[0], top=cols[1], entropy=cols[2], cycle=cycle)
    stats = run.stats
    if timer:
        timer.unwrap()

    k = args.num_draft_tokens
    cycle_ms = phase.times["cycle"] * 1e3 / cycles
    rows = []
    for name in [f"draft_{i}" for i in range(k)] + [
        "cand_build",
        "cand_score",
        "check",
        "fallback",
        "verify",
        "sample",
        "readback",
        "rollback",
    ]:
        if name not in phase.times:
            continue
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
    print(
        f"drafts per cycle {stats['drafted'] / cycles:.2f}, accepted per position "
        + "/".join(f"{accepted_at[i] / cycles:.2f}" for i in range(k))
    )
    if stats["sib_tries"]:
        print(
            f"sibling rows: {stats['sib_tries']} rejections, {stats['sib_hits']} held the correction "
            f"({stats['sib_hits'] / stats['sib_tries']:.2f}), {stats['sib_hits'] / cycles:.2f} extra tokens per cycle, "
            f"{stats['sib_rows'] / cycles:.2f} sibling rows per cycle"
        )
    if stats["cand_steps"]:
        print(
            f"candidate drafts {stats['cand_steps']}, |C| mean {stats['cand_size'] / cycles:.0f}, "
            f"fallback rate {stats['fallback'] / stats['cand_steps']:.3f}, "
            f"miss rate {stats['miss'] / stats['cand_steps']:.3f}"
        )
        if args.check_candidates:
            print(
                f"p_C - p_full mean {stats['p_diff'] / stats['cand_steps']:+.4f}, "
                f"p_C < p_full on {stats['p_lower']} steps"
            )
    if timer:
        print(f"\nverify forward by layer type (S={k + 1}, sync per layer):")
        for name, ts in timer.times.items():
            print(
                f"  {name:12} n={len(ts) / cycles:5.1f}  {sum(ts) * 1e3 / cycles:8.2f} ms/cycle"
            )
    first = next((i for i, (a, b) in enumerate(zip(out, plain_out)) if a != b), None)
    print(f"first token different from plain greedy: {first} (of {min(len(out), len(plain_out))})")
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
                    "stats": dict(stats),
                    "first_diff": first,
                    "tokens_out": out,
                    "per_cycle": run.per_cycle,
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
