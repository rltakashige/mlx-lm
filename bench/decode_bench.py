"""Throughput harness: prompt and generation tok/s with an optional MTP or draft model.

    python bench/decode_bench.py --model <path> [--draft-model <sidecar|bundled>] \\
        [--num-draft-tokens 2] [--prompt "text" | --prompt-tokens 512] [--max-tokens 256] \\
        [--repeats 3] [--json out.json]
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_lm import load, stream_generate
from mlx_lm.models.qwen3_5_mtp import load_bundled


def run(model, tok, prompt, max_tokens, draft, k):
    tokens, accepted = [], 0
    for r in stream_generate(
        model, tok, prompt, max_tokens=max_tokens, draft_model=draft, num_draft_tokens=k
    ):
        tokens.append(r.token)
        accepted += int(r.from_draft)
    forwards = len(tokens) - accepted
    return {
        "tokens": len(tokens),
        "prompt_tps": r.prompt_tps,
        "generation_tps": r.generation_tps,
        "peak_memory_gb": r.peak_memory,
        "accepted": accepted,
        "acceptance": accepted / max(len(tokens), 1),
        "tokens_per_forward": len(tokens) / max(forwards, 1),
        "text": tok.decode(tokens),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--draft-model", help="sidecar path, 'bundled', or omit for plain decoding"
    )
    ap.add_argument("--num-draft-tokens", "-k", type=int, default=2)
    ap.add_argument("--prompt", help="chat prompt text")
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="random prompt length when --prompt is not given",
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    model, tok = load(args.model)
    draft = None
    if args.draft_model == "bundled":
        draft = load_bundled(args.model)
    elif args.draft_model:
        draft = load(args.draft_model)[0]
    if args.prompt:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True
        )
    else:
        vocab = model.language_model.args.vocab_size
        prompt = mx.random.randint(0, vocab, (args.prompt_tokens,)).tolist()
    tok._eos_token_ids = {}

    run(model, tok, prompt, 32, draft, args.num_draft_tokens)
    results = [
        run(model, tok, prompt, args.max_tokens, draft, args.num_draft_tokens)
        for _ in range(args.repeats)
    ]
    keys = [
        "prompt_tps",
        "generation_tps",
        "peak_memory_gb",
        "acceptance",
        "tokens_per_forward",
    ]
    for i, r in enumerate(results):
        print(f"trial {i + 1}: " + ", ".join(f"{k}={r[k]:.3f}" for k in keys))
    best = max(results, key=lambda r: r["generation_tps"])
    print("best: " + ", ".join(f"{k}={best[k]:.3f}" for k in keys))
    if args.json:
        args.json.write_text(
            json.dumps(
                {"args": vars(args) | {"json": str(args.json)}, "trials": results},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
