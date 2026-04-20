---
name: cuda-kernel-optimizer
description: Iteratively optimize a CUDA/CUTLASS/Triton kernel against a reference implementation using ncu-guided reasoning. Use this skill whenever the user asks to optimize, speed up, or improve the performance of a .cu kernel (CUDA or CUTLASS) or a Triton/Python kernel file, especially when they provide a baseline operator and a reference, mention "ncu", "Nsight Compute", "iterative optimization", "kernel tuning", or ask Claude to "make this kernel faster". The skill drives a multi-iteration optimization loop: profile with ncu → pick top compute/memory/latency metrics → propose three complementary optimization methods (one per axis) → generate a new kernel → validate + benchmark → update global state. Each iteration's artifacts (kernel, CoT analysis, ncu-rep) are persisted under a timestamped run folder, and a final summary is emitted.
---

# CUDA Kernel Iterative Optimizer

## What this skill does

Given:
- a **baseline kernel file** (`.cu` for CUDA / CUTLASS, or `.py` for Triton),
- a **reference** Python file (exposes `reference(**kwargs)` — same contract as `benchmark.py --ref`),
- kernel dimension arguments (e.g. `--M=4096 --N=4096 --K=4096`),
- optional iteration count `N` (default **3**) and `ncu_num` (default **5**),

the skill runs an **ncu-guided iterative optimization loop** and produces a timestamped directory of per-iteration artifacts plus a final summary.

## Inputs the skill expects from the user

Before starting, confirm you have:

1. **Baseline operator file**, e.g. `./gemm.cu` or `./gemm_triton.py`
2. **Reference file**, e.g. `./ref.py` (required — correctness validation depends on it)
3. **Dimensions** — kernel-signature scalars like `--M=4096 --N=4096 --K=4096`
4. **Iteration count `N`** (default 3)
5. **`ncu_num`** — how many top metrics to extract per axis (default 5)

> `benchmark.py` is bundled at `scripts/benchmark.py`; all scripts default to it automatically. Override with `--benchmark <path>` only if you have a custom version.

If any of these are missing, ask the user once — briefly — then proceed.

## The loop at a glance

```
0. check_env        → env.json (GPU, nvcc, CUTLASS, ncu)
1. init run folder  → run_YYYYMMDD_HHMMSS/
2. copy baseline    → baseline/ + bench once to seed `best`
3. for i in 1..N:
     a. profile best_kernel with ncu        → iterv{i}/best_input.ncu-rep
     b. extract top compute/mem/latency     → ncu_top.json
     c. Claude picks 3 methods (1 per axis) → analysis.md (CoT)
     d. Claude writes new kernel            → iterv{i}/kernel.{cu|py}
     e. benchmark.py --ref ...              → validate + time
     f. if FAIL: regenerate with reason (max 3 retries)
     g. if PASS: profile new kernel, update state
4. emit summary.md
```

Steps (a), (b), (e), (f) are **deterministic** — run them via `scripts/`. Steps (c) and (d) are **where Claude thinks** — follow the reasoning rules in `references/optimization_catalog.md` and `references/ncu_metrics_guide.md`.

---

## Step 0 — Check local environment

Run the env probe **before** doing anything else:

```bash
python <skill>/scripts/check_env.py --out ./env.json
```

It records: GPU name + compute capability (SM arch), nvcc path + version, ncu path + version, CUTLASS include dir (if detectable), CUDA driver, torch + triton versions. If **ncu is not available** or the user is not running as root / lacks `--access=all` perf counters, warn the user explicitly — the skill can degrade to benchmark-only mode, but ncu-guided reasoning is significantly weaker without it.

## Step 0b — Preflight the baseline + ref contract

```bash
python <skill>/scripts/preflight.py \
  --baseline ./gemm.cu \
  --ref      ./ref.py \
  --dims     '{"M":4096,"N":4096,"K":4096}'
```

Validates, before any compilation or profiling: baseline has `extern "C" void solve(...)` (CUDA/CUTLASS) *or* `setup(**kwargs)` + `run_kernel(**kwargs)` (Triton); ref has `reference(**kwargs)`; every `int`/`long` parameter of `solve` has a value supplied in `--dims`. On failure the script prints exactly what's missing — surface that directly to the user instead of pushing forward. `orchestrate.py setup` runs this automatically.

## Step 1 — Initialize the run folder

```bash
python <skill>/scripts/state.py init \
  --baseline ./gemm.cu \
  --ref ./ref.py \
  --iterations 3 \
  --ncu-num 5 \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --env ./env.json
```

This creates `./run_YYYYMMDD_HHMMSS/` next to the baseline file, copies the baseline into `baseline/`, and writes `state.json` with:

```jsonc
{
  "run_dir": "...",
  "baseline_file": "...",
  "ref_file": "...",
  "best_file": "<baseline>",        // updated as we find better ones
  "best_metric_ms": null,           // filled by first bench
  "best_ncu_rep": null,
  "env": {...},
  "iterations_total": 3,
  "ncu_num": 5,
  "selected_methods": [],           // ${当前已经选择过的优化方法}
  "effective_methods": [],          // ${当前有效的优化方法}
  "ineffective_methods": [],        // ${当前无效的优化方法}
  "dims": {...},
  "history": []                      // per-iteration records
}
```

## Step 2 — Seed `best` with a baseline benchmark

```bash
python <skill>/scripts/run_iteration.py seed-baseline \
  --state ./run_*/state.json
```

This runs `benchmark.py` against the baseline with `--ref` (correctness must pass), stores the JSON metrics, and records `best_metric_ms` in `state.json`. If the **baseline itself fails validation**, halt and report to the user — something is wrong with the input pair.

---

## Step 3 — Iteration loop (repeat for i = 1..N)

### 3a. Profile the current `best` with ncu

```bash
python <skill>/scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter $i \
  --which best_input
```

This writes:
- `iterv{i}/best_input.ncu-rep` — full ncu report
- `iterv{i}/ncu_top.json` — top-`ncu_num` metrics per axis (compute / memory / latency), chosen using the rubric in `references/ncu_metrics_guide.md`

The profiler uses `ncu --set full --launch-count 3 -k solve` for CUDA/CUTLASS, and `--launch-count 3` with no `-k` filter for Triton (kernel names are not stable, so we let ncu capture all launches inside the first few benchmark reps). If ncu is unavailable, the script writes `ncu_top.json` with a `"degraded": true` flag — Claude should then reason from static code features only.

### 3b. Decide on 3 optimization methods (Claude reasons here)

**Read** (in this order):
1. `references/optimization_catalog.md` — **the catalog is priority-ordered; selection MUST follow priority**
2. `iterv{i}/ncu_top.json` — current bottleneck metrics (the trigger evidence)
3. `state.json` — `best_file`, `selected_methods`, `effective_methods`, `ineffective_methods`, `env`
4. The current `best_file` source code
5. `references/ncu_metrics_guide.md` — metric → root cause mapping

**Selection rule — STRICT PRIORITY SCAN**:
For each axis (compute / memory / latency), scan the catalog **from P1 downward**. For each priority level, check:
1. Is `method.id` already in `selected_methods`? → skip (already tried)
2. Does the detected `sm_arch` meet the method's arch requirement? → skip if not
3. Does the method's **skip condition** apply? → skip (record reason in analysis.md)
4. Does the method's **trigger condition** match the ncu evidence? → skip if no bottleneck here

Select the **first method that passes all four checks**. **Do NOT skip a higher-priority applicable method to try a lower-priority one.** If you must skip a high-priority method, document the specific reason in analysis.md's "排除候选" section.

**Produce** exactly **three methods** (one per axis). For each, write a Chain-of-Thought: specific ncu metric value → root-cause hypothesis → proposed method (citing catalog id + priority) → expected metric shifts.

**Hard constraints**:
1. If `memory.multi_stage_pipeline` (P5) and `latency.async_pipeline` (P3) are both selected, they count as one optimization — replace one with the next applicable method on that axis.
2. Methods in `ineffective_methods` are **blocked** unless the ncu bottleneck profile has fundamentally changed (must cite specific metric deltas in analysis.md).
3. All three methods must be **arch-compatible** and **mutually orthogonal** (see catalog's Combining Rules).

Save the decision to `iterv{i}/analysis.md` using the template in `templates/iteration_report.md`.

> **Enforcement**: `methods.json` is automatically validated by `scripts/validate_methods.py` when `state.py update` is called (via `orchestrate.py close-iter`). The validator checks against `references/method_registry.json` and rejects submissions where:
> - Any method id is not in the registry
> - Submitted priority / axis don't match the registry
> - A method's priority is > 1 but `skipped_higher` doesn't account for every higher-priority method on that axis
> - Skip reason codes are not in `{already_selected, arch_incompatible, skip_condition, no_trigger}`
> - Both sides of a coupled pair are selected (e.g. `memory.multi_stage_pipeline` + `latency.async_pipeline`)
> - The method is already in `selected_methods`, or in `ineffective_methods` without `--allow-ineffective` override
> - The method's `min_sm` exceeds the detected `sm_arch`
>
> On rejection, the iteration is NOT recorded to state and Claude must fix `methods.json` and retry. This is the mechanism that makes priority discipline enforceable rather than advisory.

### 3c. Generate the new kernel (Claude writes code)

Take `best_file` as the starting point and apply all 3 methods to produce `iterv{i}/kernel.<ext>`. Preserve the `extern "C" void solve(...)` signature (for CUDA/CUTLASS) or the `setup(**kwargs)` + `run_kernel(**kwargs)` contract (for Triton). Do not change the public signature — `benchmark.py` relies on it.

### 3d. Validate + benchmark

```bash
python <skill>/scripts/run_iteration.py benchmark \
  --state ./run_*/state.json \
  --iter $i
```

This calls `benchmark.py solution_file --ref ref.py --json-out iterv{i}/bench.json <dims>`. Outcomes:

- **Validation fail** → read `bench.json["correctness"]["passed"] == false` and the stderr capture. Go to 3e.
- **Validation pass** → read `bench.json["kernel"]["average_ms"]`. Compare to `state.best_metric_ms`. Go to 3f.

### 3e. Repair on validation failure (up to 3 retries per iteration)

If correctness failed, read the validation diff (max|delta|, first-bad-idx, previews from `bench.json`). Claude rewrites `iterv{i}/kernel.<ext>` incorporating the failure reason, **without abandoning the 3 chosen methods** (unless one is fundamentally incompatible — then log that as the reason). Re-run 3d. After 3 retries, mark the iteration as a failed attempt in `state.history`, move on, and **do not** add the methods to any list.

### 3f. Update global state

On success:

```bash
python <skill>/scripts/state.py update \
  --state ./run_*/state.json \
  --iter $i \
  --kernel iterv{i}/kernel.<ext> \
  --bench iterv{i}/bench.json \
  --methods-json iterv{i}/methods.json
```

Rules implemented by the script:
- `selected_methods += 3 methods` (always, since they were actually tried)
- If `new_ms < best_ms` by more than **noise_threshold (default 2%)** → methods go to `effective_methods`, `best_file` := `iterv{i}/kernel.<ext>`, `best_metric_ms` := `new_ms`, also save `iterv{i}/kernel.ncu-rep` as the new `best_ncu_rep`.
- Otherwise → methods go to `ineffective_methods`. `best` stays the same.
- Append a record to `state.history` with {iter, methods, ms, speedup, status}.

Then profile the new kernel so the next iteration has fresh data even if it didn't become `best`:

```bash
python <skill>/scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter $i \
  --which kernel
```

(If the new kernel *is* the new best, the profiler reuses that report for next iteration's 3a — skipping re-profiling.)

---

## Step 4 — Final summary

After the loop:

```bash
python <skill>/scripts/summarize.py \
  --state ./run_*/state.json \
  --out ./run_*/summary.md
```

The summary includes: env snapshot, per-iteration timeline (methods + ms + speedup + status), final best kernel + final speedup vs baseline, consolidated effective/ineffective method catalogs, and a short Claude-written retrospective (what worked, what didn't, what to try next).

---

## Reasoning references

- **`references/optimization_catalog.md`** — Catalog of CUDA / CUTLASS / Triton optimization methods organized by axis (compute / memory / latency). Use this as the menu when selecting the 3 methods. Each entry notes arch requirements and typical ncu signatures.
- **`references/ncu_metrics_guide.md`** — How to read ncu output, which metrics matter per axis, and the mapping from bottleneck signature → probable optimization.

Read both before the first iteration and re-consult per-axis sections in later iterations.

---

## Failure modes to watch for

- **Benchmark crashes, not just fails validation** → the kernel has UB or a launch error. Check the `bench.json` `"error"` field; the captured process stderr often lives next to the JSON.
- **ncu reports all-zero metrics** → likely means `ncu` didn't attach (permissions) or the launch-count filter missed the kernel. Re-run with `--no-kernel-filter` (the profile script exposes this).
- **`can_read_counters: false` in env.json** → container / non-root setup blocks perf counters. ncu output will be scalar-only or empty. Tell the user; don't silently pretend the `ncu_top.json` is trustworthy. The skill flags this by setting `"degraded": true` in `ncu_top.json` on zero-metric collections.
- **ncu older than 2022.1** → the CSV column names differ slightly (capitalization, unit suffix). `profile_ncu.py` handles both, but if you see `metric_count_collected: 0` with `degraded: false`, check `*.ncu.log` and pass `--no-kernel-filter` or bump ncu.
- **Triton + `@triton.autotune`** → autotuning **under ncu** triggers a profile per config, which can balloon from seconds to many minutes and sometimes time out. Before the first profile, either (a) hard-code the most promising config and remove `autotune`, or (b) call `--launch-count 1 --warmup 0 --repeat 1` to limit profiled launches. Note this in `analysis.md`.
- **Massive speedup claims with small absolute ms** → measurement noise. The skill's 2% threshold guards against this, but call it out in `analysis.md` if the best time is < 50 µs.
- **Triton + CUTLASS mix** — unusual but legal. If the user swaps backends mid-loop, note it in `analysis.md`; the benchmark script auto-detects backend per file.
- **Methods chosen but the generated kernel has the same SASS** — can happen when Claude "intends" an optimization (e.g. `unroll_inner_loop`) but nvcc was already unrolling. Confirm by diffing `iterv{i}/kernel.ncu-rep` against `iterv{i}/best_input.ncu-rep` metric-by-metric — if nothing moved on the targeted axis, mark that method ineffective in the retrospective.

---

## Output contract (what the user gets)

```
<baseline-dir>/run_YYYYMMDD_HHMMSS/
├── env.json
├── state.json
├── baseline/
│   ├── <baseline>           (copied)
│   └── bench.json
├── iterv1/
│   ├── kernel.<ext>
│   ├── analysis.md          (ncu metrics + 3 methods + CoT)
│   ├── methods.json
│   ├── best_input.ncu-rep   (profile of best going INTO this iter)
│   ├── ncu_top.json
│   ├── kernel.ncu-rep       (profile of the new kernel)
│   └── bench.json
├── iterv2/...
├── iterv3/...
└── summary.md
```

Announce the `run_dir` path at the end so the user can open any iteration directly.
