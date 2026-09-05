# bench

Measurement tools used for the Qwen3.8-27B-4bit work. All take `--model <path or repo>`.

- `layer_profile.py`: per-layer-type time breakdown (embed, GDN layers, attention layers, norms,
  lm_head, and the blocks and ops inside each layer) for prefill and single-token decode, measured
  with synchronous timers at three nesting levels; reports the unwrapped step time so the timing
  overhead is visible. `--gputrace DIR` writes Metal captures (`mx.metal.start_capture`) of one
  decode step, one GDN layer and one attention layer for Xcode.
- `decode_bench.py`: prompt and generation tok/s with an optional MTP head (`--draft-model
  <sidecar>` or `bundled`), acceptance rate and tokens per target forward.
- `mtp_profile.py`: where the time goes inside an MTP cycle (each draft step, the verify forward,
  sampling, readback, rollback, Python remainder), per-position acceptance, and with
  `--verify-layers` the layer-type split of the verify forward.

```
python bench/layer_profile.py --model mlx-community/Qwen3.8-27B-4bit -p 512 --decode-tokens 32
python bench/decode_bench.py --model mlx-community/Qwen3.8-27B-4bit --draft-model mlx-community/Qwen3.8-27B-MTP-4bit -k 2
python bench/mtp_profile.py --model mlx-community/Qwen3.8-27B-4bit --draft-model mlx-community/Qwen3.8-27B-MTP-4bit -k 2
```

## Profiling notes

- Per-kernel GPU timings: capture with `layer_profile.py --gputrace DIR` (uses
  `mx.metal.start_capture`; the environment variable `MTL_CAPTURE_ENABLED=1` is set by the script)
  and open the `.gputrace` bundle in Xcode's GPU debugger. Captures include every resident buffer,
  so a 27B capture is about 16 GB.
- Instruments (`xctrace record --template 'Metal System Trace'`) exports only command-buffer
  level compute intervals (`metal-gpu-intervals`), no kernel names; the shader-profiler table
  stays empty outside the Xcode GUI. It answers "GPU busy vs idle", not "which kernel".
- The synchronous timers in `layer_profile.py` inflate the step (about 160 us per timed call),
  so read the per-type shares and the unwrapped step time, not the wrapped totals.
