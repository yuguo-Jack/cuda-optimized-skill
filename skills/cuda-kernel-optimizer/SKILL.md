---
name: cuda-kernel-optimizer
description: Iteratively optimize a CUDA/CUTLASS/Triton kernel against a Python reference with NCU evidence, correctness gates, deterministic single-GPU measurement, and bounded parallel CPU compilation.
---

# CUDA Kernel Iterative Optimizer (v3)

## What this skill does

Given:
- a **baseline kernel file** (`.cu` for CUDA / CUTLASS, or `.py` for Triton),
- a **reference** Python file (exposes `reference(**kwargs)` — same contract as `benchmark.py --ref`),
- optional `atol`/`rtol` and `workload_model(**dims) -> {flops, bytes_min}` in the reference,
- kernel dimension arguments (e.g. `--M=4096 --N=4096 --K=4096`),
- optional iteration count `N` (default **3**), `ncu_num` (default **5**), and `branches` (default **4**),
- optional `--compile-jobs auto|N` (default `auto`) and `--numerics-mode reference|strict|approximate` (default `reference`),

the skill runs a **roofline-guided, branch-and-select iterative optimization loop** and produces a timestamped directory of per-iteration artifacts plus a final summary.

## Key point

1. **Evidence-driven axis budget**: use a real workload roofline when `ref.py` exposes `workload_model`; otherwise label the result as a bottleneck-gap heuristic and do not claim `near_peak`.
2. **Branch-and-Select**: each iteration generates K candidate kernels (hyperparameter/implementation variants), benchmarks all, selects champion. 
3. **Ablation attribution**: after selecting champion, each method is individually ablated to determine its actual contribution.
4. **SASS verification**: `cuobjdump --dump-sass` confirms claimed optimizations actually appear in generated code.
5. **Every iteration produces an adaptive ncu report** on the champion kernel; only CPU compilation is parallel.

## Inputs the skill expects from the user

Before starting, confirm you have:

1. **Baseline operator file**, e.g. `./gemm.cu` or `./gemm_triton.py`
2. **Reference file**, e.g. `./ref.py` (required — correctness validation depends on it)
3. **Dimensions** — kernel-signature scalars like `--M=4096 --N=4096 --K=4096`
4. **Iteration count `N`** (default 3)
5. **`ncu_num`** — how many top metrics to extract per axis (default 5)
6. **`branches`** — how many hyperparameter variants per iteration (default 4)

> `benchmark.py` is bundled at `scripts/benchmark.py`; all scripts default to it automatically.

If any of these are missing, ask the user once — briefly — then proceed.

## The loop at a glance

```
0. check_env          → env.json (GPU, nvcc, CUTLASS, ncu)
1. init run folder    → run_YYYYMMDD_HHMMSS/
2. copy baseline      → baseline/ + bench once to seed `best`
3. for i in 1..N:
     a. profile best_kernel with adaptive ncu (full/light by duration)  → iterv{i}/best_input.ncu-rep
     b. extract top compute/mem/latency            → ncu_top.json
     c. roofline.py: compute real roofline or bottleneck gaps → roofline.json + axis_budget
     d. Claude picks methods (b_axis per axis, cap=2) → analysis.md (CoT)
     e. Claude writes K branch kernels (same methods, diff hyperparams)
     f. contract gate, then branch_explore.py: parallel CPU compile, serial GPU bench all K → select champion
     g. if champion FAIL: regenerate (max 3 retries)
     h. ncu profile champion (adaptive, same bundle) → iterv{i}/kernel.ncu-rep
     i. ablate.py: single-method rollback bench    → attribution.json
     j. sass_check.py: verify SASS signatures      → sass_check.json
     k. update state with attribution + SASS results
4. emit summary.md
```

Steps (a), (b), (c), (f), (h), (i), (j) are scripted and reproducible; GPU timing itself is noisy.
Steps (d) and (e) are **where Claude thinks** — follow the reasoning rules in `references/optimization_catalog.md` and `references/ncu_metrics_guide.md`.

---

## Step 0 — Check local environment

Run the env probe **before** doing anything else:

```bash
python <skill>/scripts/check_env.py --out ./env.json
```

It records: GPU name + compute capability (SM arch), nvcc path + version, ncu path + version, CUTLASS include dir (if detectable), CUDA driver, torch + triton versions, GPU peak FLOPS and bandwidth (for roofline). If **ncu is not available** or the user is not running as root / lacks `--access=all` perf counters, warn the user explicitly — the skill can degrade to benchmark-only mode, but ncu-guided reasoning is significantly weaker without it.

## Step 0b — Preflight the baseline + ref contract

```bash
python <skill>/scripts/preflight.py \
  --baseline ./gemm.cu \
  --ref      ./ref.py \
  --dims     '{"M":4096,"N":4096,"K":4096}'
```

Validates baseline and reference contracts. On failure, surface errors directly to the user. `orchestrate.py setup` runs this automatically.

## Step 1 — Initialize the run folder

```bash
python <skill>/scripts/state.py init \
  --baseline ./gemm.cu \
  --ref ./ref.py \
  --iterations 3 \
  --ncu-num 5 \
  --branches 4 \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --env ./env.json
```

Creates `./run_YYYYMMDD_HHMMSS/` next to the baseline file and writes `state.json`:

```jsonc
{
  "run_dir": "...",
  "baseline_file": "...",
  "ref_file": "...",
  "best_file": "<baseline>",
  "best_metric_ms": null,
  "best_ncu_rep": null,
  "env": {...},
  "iterations_total": 3,
  "ncu_num": 5,
  "branches": 4,
  "selected_methods": [],
  "effective_methods": [],
  "ineffective_methods": [],
  "implementation_failed_methods": [],
  "dims": {...},
  "history": [],
  "roofline_history": [],
  "frontier": []
}
```

## Step 2 — Seed `best` with a baseline benchmark

```bash
python <skill>/scripts/run_iteration.py seed-baseline \
  --state ./run_*/state.json
```

## Step 3 — Iteration loop (repeat for i = 1..N)

### 3a. Profile the current `best` with ncu (adaptive report)

```bash
python <skill>/scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter $i \
  --which best_input
```

The default policy uses `full` below 10 ms and the light/basic metric bundle
at 10 ms and above. A full replay that times out or exits with code 11 is
retried with the light bundle. `ncu_top.json` records the selected set,
duration, reason, and all attempts.

### 3b. Compute roofline gaps and axis budgets

```bash
python <skill>/scripts/roofline.py \
  --state ./run_*/state.json \
  --iter $i
```

Reads `ncu_top.json` + `env.json`, computes:
- `Δ_c` = compute utilization gap
- `Δ_m` = bandwidth utilization gap
- `Δ_l` = max stall percentage

Writes `iterv{i}/roofline.json`:
```jsonc
{
  "delta_compute": 0.85,
  "delta_memory": 0.60,
  "delta_latency": 0.55,
  "bound": "compute",
  "near_peak": false,
  "axis_budget": {"compute": 1, "memory": 1, "latency": 1}
}
```

**Budget allocation rule**: proportional to known Δ values, rounded, cap per axis = 2, total = 3. `near_peak`/early stop is allowed only when all three gaps are known, a workload model is present, and all Δ < 0.15.

### 3c. Select methods (Claude reasons here)

**Read** (in this order):
1. `references/method_registry.json` — canonical IDs, priorities, capabilities and relations
2. `references/metric_registry.json` — versioned NCU aliases, units and missing-value semantics
3. `references/optimization_catalog.md` — only the relevant method/backend cards and conditional archetype packs
4. `iterv{i}/roofline.json` — axis budgets and bound classification
5. `iterv{i}/ncu_top.json` — current bottleneck metrics
6. `state.json` — method history and `numerics_mode`
7. The current `best_file` source code
8. `references/ncu_metrics_guide.md` — only metrics relevant to the observed bottleneck

**Selection rule — BUDGET-AWARE PRIORITY SCAN**:

For each axis with `b_axis > 0`, scan the catalog **from P1 downward**. For each priority level, check:
1. Is `method.id` already in `selected_methods`? → skip (already tried)
2. Does the detected `sm_arch` meet the method's arch requirement? → skip if not
3. Does the method's **skip condition** apply? → skip (record reason in analysis.md)
4. Does the method's **trigger condition** match the ncu evidence? → skip if no bottleneck here

Select methods in priority order until `b_axis` eligible methods are found. If fewer candidates pass all gates, leave the budget under-filled and record the reason; never add an untriggered or unsafe filler.

Produce up to **B methods** (sum of axis budgets, typically 3). Record concise evidence and decision rationale; do not emit private Chain-of-Thought.

**Hard constraints**:
1. Apply typed registry relations: conflicts reject a pair, complements are allowed, and same-budget groups consume one slot.
2. Methods in `ineffective_methods` are **blocked** unless ncu bottleneck has fundamentally changed.
3. Methods in `implementation_failed_methods` require explicit acknowledgment of the prior failure.
4. All methods must pass backend, capability, toolchain and numerical-semantic gates.
5. **Per-axis cap is 2** — no axis can receive more than 2 methods.

Save to `iterv{i}/analysis.md` using the template in `templates/iteration_report.md`.

### 3d. Generate K branch kernels (Claude writes code)

All K branches share the **same method combination** from step 3c. They differ in **hyperparameters and implementation details**:
- Tile sizes (BLOCK_M, BLOCK_N, BLOCK_K)
- Pipeline stage count (num_stages)
- Warp count (num_warps)
- Implementation variant within a method (e.g., swizzle mode, MMA atom selection)

Write K kernels under `iterv{i}/branches/b{1..K}/kernel.<ext>`.

### 3e. Branch explore: compile + benchmark all K

```bash
python <skill>/scripts/branch_explore.py \
  --state ./run_*/state.json \
  --iter $i
```

For the bundled benchmark, compiles CUDA/CUTLASS branches in a bounded CPU pool, waits for all builds, then benchmarks them serially on the ranking GPU in `b1..bK` order. Triton and unsupported custom benchmarks use the original serial path. Selects champion by `(average_ms, branch_index)`; non-champions are saved to `state.frontier`.

### 3f. Repair on validation failure (up to 3 retries per iteration)

If champion fails correctness, Claude rewrites and re-runs 3e.

Before GPU work, CUDA branches pass `contract_check.py`. Its independent
`compile_pass`, `contract_pass`, `correctness_pass`, `race_safe`, and
`timing_valid` states are persisted; a failed contract cannot be timed or
selected. Branch results retain requested/realized shapes, OOM attempts,
robust CV, and kernel-only/end-to-end timing fields.

### 3g. Profile champion with ncu (same adaptive bundle)

```bash
python <skill>/scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter $i \
  --which kernel
```

Writes `iterv{i}/kernel.ncu-rep`. The selected metric bundle is kept identical
to the first profile in the run so baseline/champion deltas remain comparable.

### 3h. Ablation attribution

```bash
python <skill>/scripts/ablate.py \
  --state ./run_*/state.json \
  --iter $i
```

For each pre-generated ablation kernel, compile CUDA/CUTLASS versions in the same bounded pool, then benchmark them serially. Missing or failed ablations are inconclusive. Computes attribution:
```
attribution(m) = ms_without_m - ms_champion
```
Positive attribution = the method contributed positively. Near-zero or negative = the method was not helpful.

Writes `iterv{i}/attribution.json`.

### 3i. SASS verification

```bash
python <skill>/scripts/sass_check.py \
  --state ./run_*/state.json \
  --iter $i
```

Runs the declared verifier on the compiled champion. SASS status is `pass|fail|inconclusive|not_applicable|tool_error`; empty patterns, Triton and unavailable artifacts are not success. Writes `iterv{i}/sass_check.json`.

### 3j. Update global state

```bash
python <skill>/scripts/state.py update \
  --state ./run_*/state.json \
  --iter $i \
  --kernel iterv{i}/kernel.<ext> \
  --bench iterv{i}/bench.json \
  --methods-json iterv{i}/methods.json \
  --attribution iterv{i}/attribution.json \
  --sass-check iterv{i}/sass_check.json
```

Rules:
- `selected_methods += all methods` (always)
- Method enters `effective_methods` **only if**: attribution > noise_threshold **AND** SASS verified
- Method enters `implementation_failed_methods` only on a strong method-specific verification failure
- Method enters `ineffective_methods` if attribution ≤ noise_threshold and verification is conclusive
- Method enters `unverified_methods` if ablation or verification is inconclusive/unavailable
- If `new_ms < best_ms` by more than noise_threshold → `best_file` updated
- Append record to `state.history` and `state.roofline_history`

---

## Step 4 — Final summary

```bash
python <skill>/scripts/summarize.py \
  --state ./run_*/state.json \
  --out ./run_*/summary.md
```

## Compilation and numerical policy

`--compile-jobs auto` caps workers at half the CPUs, four total workers, and
available-memory estimates (2 GiB/job for CUDA, 4 GiB/job for CUTLASS), retaining
2 GiB. Resource/OOM failures retry once serially. The build manifest covers the
effective source, compiler/toolchain, architecture, flags and include/link
inputs; only a matching manifest may be reused. The run-local cache is never a
timing cache.

`numerics_mode=reference` keeps the reference's effective `atol/rtol` contract.
`strict` admits only bitwise-preserving or explicitly preconditioned methods;
`approximate` requires explicit opt-in and still runs all correctness checks.
Unknown NCU metrics and SASS/tool errors are inconclusive, never zero or pass.

---

## Reasoning references

- **`references/optimization_catalog.md`** — Catalog of optimization methods by axis, with algorithmic methods section.
- **`references/ncu_metrics_guide.md`** — How to read ncu output and map bottleneck signatures.
- **`references/sass_signatures.json`** — Expected SASS instruction patterns per method.

---

## Failure modes to watch for

- **Benchmark crashes** → check `bench.json` `"error"` field.
- **ncu reports all-zero metrics** → permissions issue or launch filter miss.
- **`can_read_counters: false` in env.json** → warn user; degrade gracefully.
- **Triton + `@triton.autotune`** → hard-code config before profiling.
- **Champion chosen but all methods have near-zero attribution** → the speedup came from hyperparameter change, not methods. Record in analysis.md.
- **SASS signature missing but kernel is faster** → nvcc took a different path. Mark the method `unverified` unless a strong, method-specific failure is proven; keep the kernel if it's faster.
- **Branch explore: all K branches fail validation** → Claude must rewrite with different approach.
- **Early stop triggered** → all Δ < 0.15, kernel is near roofline. Report to user.

---

## Output contract

```
<baseline-dir>/run_YYYYMMDD_HHMMSS/
├── env.json
├── state.json
├── baseline/
│   ├── <baseline>           (copied)
│   └── bench.json
├── iterv1/
│   ├── kernel.<ext>          (champion)
│   ├── analysis.md           (roofline + methods + CoT)
│   ├── methods.json
│   ├── roofline.json
│   ├── best_input.ncu-rep    (profile of best going INTO this iter)
│   ├── ncu_top.json
│   ├── kernel.ncu-rep        (profile of champion — ALWAYS present)
│   ├── attribution.json
│   ├── sass_check.json
│   ├── bench.json
│   └── branches/
│       ├── b1/ ... b4/       (all branch candidates)
├── iterv2/...
├── iterv3/...
└── summary.md
```
