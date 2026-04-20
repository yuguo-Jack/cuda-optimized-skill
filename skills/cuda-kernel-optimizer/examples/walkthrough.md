# Example Walkthrough

A hypothetical session optimizing `gemm.cu` against `ref.py` for 3 iterations on an H100.

## Layout before the run

```
~/work/
├── gemm.cu          ← baseline (has `extern "C" void solve(float*, float*, float*, int, int, int)`)
└── ref.py           ← defines `reference(A, B, C, M, N, K)`  + `atol = 1e-3`
```

> `benchmark.py` is bundled inside the skill at `scripts/benchmark.py` — no need to keep a separate copy.

## Command chain (Claude-driven)

### Step 0–2 — Setup

```bash
python cuda-kernel-optimizer/scripts/orchestrate.py setup \
  --baseline ~/work/gemm.cu \
  --ref      ~/work/ref.py \
  --iterations 3 \
  --ncu-num 5 \
  --dims '{"M":4096,"N":4096,"K":4096}'
```

Output (abridged):
```json
{
  "run_dir": "/home/user/work/run_20260418_143022",
  "state":   "/home/user/work/run_20260418_143022/state.json",
  "env":     "/home/user/work/env.json",
  "next_step": "Claude should now read iterv1/ncu_top.json..."
}
```

### Step 3b–c (iter 1) — Claude reasons and writes code

Claude inspects:

- `run_.../iterv1/ncu_top.json` — sees `hmma_cycles_active = 6.1%` → compute axis cold
- `run_.../iterv1/ncu_top.json` — sees `long_scoreboard stalls = 58%` → latency bound on global loads
- `run_.../iterv1/ncu_top.json` — sees `L2 hit = 22%` → poor reuse

Claude picks:

| Axis    | Method id                              |
|---------|----------------------------------------|
| compute | `compute.tensor_cores_mma_sync`        |
| memory  | `memory.smem_swizzle_xor`              |
| latency | `latency.multi_stage_pipeline`         |

Claude writes (into `iterv1/`):
- `kernel.cu` — updated source applying all three
- `methods.json` — matches the schema in `templates/methods.schema.json`
- `analysis.md` — filled-in `templates/iteration_report.md`

### Step 3d–f (iter 1) — Validate, update, re-profile

```bash
python cuda-kernel-optimizer/scripts/orchestrate.py close-iter \
  --run-dir ~/work/run_20260418_143022 \
  --iter 1
```

Happy path output:
```json
{
  "iter": 1,
  "status": "closed",
  "best_ms": 2.132,
  "next_iter": 2,
  "state": "/home/user/work/run_20260418_143022/state.json"
}
```

Validation-failure path:
```json
{
  "iter": 1,
  "status": "validation_failed",
  "bench_json": ".../iterv1/bench.json",
  "stderr_log": ".../iterv1/bench.stderr.txt",
  "guidance": "Claude should read bench.json['correctness'] and bench.stderr.txt, ..."
}
```

Claude reads the diff, edits `iterv1/kernel.cu` (preserving `methods.json`), and re-runs `close-iter`. Up to 3 retries per iteration, then give up and move on.

### Steps 3b–f (iter 2, iter 3) — same loop

Each iteration picks **different** methods; by now `selected_methods` has 3 entries (the picks of iter 1), and iter 2's picks must avoid them. If iter 1 was effective, Claude composes orthogonal picks that build on top; if iter 1 was ineffective, Claude picks a different axis emphasis.

### Step 4 — Finalize

```bash
python cuda-kernel-optimizer/scripts/orchestrate.py finalize \
  --run-dir ~/work/run_20260418_143022
```

Produces `summary.md` with headline speedup, timeline table, and the effective/ineffective method catalogs. Claude then appends a short retrospective paragraph at the bottom.

## Final layout

```
run_20260418_143022/
├── state.json
├── env.json
├── summary.md
├── baseline/
│   ├── gemm.cu
│   └── bench.json
├── iterv1/
│   ├── kernel.cu
│   ├── methods.json
│   ├── analysis.md
│   ├── best_input.ncu-rep     (profile of baseline)
│   ├── ncu_top.json
│   ├── kernel.ncu-rep         (profile of iter-1 kernel)
│   ├── bench.json
│   └── bench.stderr.txt
├── iterv2/
│   └── ... (same pattern, best_input.ncu-rep is the iter-1 kernel's rep)
└── iterv3/
    └── ...
```

## Tip: `--dry-run`-style usage when ncu isn't available

If `env.json` says `ncu.available == false` (e.g. in a container without perf-counter access), the profiler still produces `ncu_top.json` but with `"degraded": true`. Claude should then rely on:
- static source inspection of `best_file` (what tiling, what pipeline depth, what tensor-core usage)
- the reference implementation (what shape, what dtype)
- arch-specific defaults from `references/optimization_catalog.md`

This loses the evidence-based ranking but preserves the 3-methods-per-iter discipline.
