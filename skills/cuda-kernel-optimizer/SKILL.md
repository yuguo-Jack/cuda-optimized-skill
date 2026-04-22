---
name: cuda-kernel-optimizer
description: Iteratively optimize a CUDA/CUTLASS/Triton kernel against a reference implementation using ncu-guided reasoning. Use this skill whenever the user asks to optimize, speed up, or improve the performance of a .cu kernel (CUDA or CUTLASS) or a Triton/Python kernel file, especially when they provide a baseline operator and a reference, mention "ncu", "Nsight Compute", "iterative optimization", "kernel tuning", or ask Claude to "make this kernel faster". The skill drives a multi-iteration roofline-guided optimization loop: profile with ncu → compute roofline gaps → allocate axis budgets → pick methods by priority scan → generate K branch candidates → validate + benchmark → select champion → ablation attribution → SASS verification → update global state. Each iteration's artifacts (kernel, CoT analysis, ncu-rep) are persisted under a timestamped run folder, and a final summary is emitted.
---

# CUDA Kernel Iterative Optimizer (v2 — Roofline-Driven)

## What this skill does

Given:
- a **baseline kernel file** (`.cu` for CUDA / CUTLASS, or `.py` for Triton),
- a **reference** Python file (exposes `reference(**kwargs)` — same contract as `benchmark.py --ref`),
- kernel dimension arguments (e.g. `--M=4096 --N=4096 --K=4096`),
- optional iteration count `N` (default **3**), `ncu_num` (default **5**), and `branches` (default **4**),

the skill runs a **roofline-guided, branch-and-select iterative optimization loop** and produces a timestamped directory of per-iteration artifacts plus a final summary.

## Key point

1. **Roofline-driven axis budget**: compute/memory/latency axis budgets are allocated proportionally to measured Δ gaps, with a per-axis cap of 2. 
2. **Branch-and-Select**: each iteration generates K candidate kernels (hyperparameter/implementation variants), benchmarks all, selects champion. 
3. **Ablation attribution**: after selecting champion, each method is individually ablated to determine its actual contribution.
4. **SASS verification**: `cuobjdump --dump-sass` confirms claimed optimizations actually appear in generated code.
5. **Every iteration produces a full ncu report** on the champion kernel.

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
     a. profile best_kernel with ncu (--set full)  → iterv{i}/best_input.ncu-rep
     b. extract top compute/mem/latency            → ncu_top.json
     c. roofline.py: compute Δ_c, Δ_m, Δ_l        → roofline.json + axis_budget
        if near_peak (all Δ < 0.15) → early stop
     d. Claude picks methods (b_axis per axis, cap=2) → analysis.md (CoT)
     e. Claude writes K branch kernels (same methods, diff hyperparams)
     f. branch_explore.py: compile + bench all K   → select champion
     g. if champion FAIL: regenerate (max 3 retries)
     h. ncu profile champion (--set full)          → iterv{i}/kernel.ncu-rep
     i. ablate.py: single-method rollback bench    → attribution.json
     j. sass_check.py: verify SASS signatures      → sass_check.json
     k. update state with attribution + SASS results
4. emit summary.md
```

Steps (a), (b), (c), (f), (h), (i), (j) are **deterministic** — run via scripts.
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

### 3a. Profile the current `best` with ncu (FULL report — mandatory)

```bash
python <skill>/scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter $i \
  --which best_input
```

Writes `iterv{i}/best_input.ncu-rep` (full ncu report) and `iterv{i}/ncu_top.json`.

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

**Budget allocation rule**: proportional to Δ, rounded, cap per axis = 2, total = 3. If all Δ < 0.15 → `near_peak: true` → **early stop**.

### 3c. Select methods (Claude reasons here)

**Read** (in this order):
1. `references/optimization_catalog.md` — priority-ordered catalog
2. `iterv{i}/roofline.json` — axis budgets and bound classification
3. `iterv{i}/ncu_top.json` — current bottleneck metrics
4. `state.json` — `best_file`, `selected_methods`, `effective_methods`, `ineffective_methods`
5. The current `best_file` source code
6. `references/ncu_metrics_guide.md` — metric → root cause mapping

**Selection rule — BUDGET-AWARE PRIORITY SCAN**:

For each axis with `b_axis > 0`, scan the catalog **from P1 downward**. For each priority level, check:
1. Is `method.id` already in `selected_methods`? → skip (already tried)
2. Does the detected `sm_arch` meet the method's arch requirement? → skip if not
3. Does the method's **skip condition** apply? → skip (record reason in analysis.md)
4. Does the method's **trigger condition** match the ncu evidence? → skip if no bottleneck here

Select the **first method that passes all four checks**. Continue scanning until `b_axis` methods are selected for that axis. If `b_axis >= 2`, after collecting all candidates that pass checks, rank by **trigger strength** and take the top `b_axis`.

**Produce** exactly **B methods** (sum of axis budgets, typically 3). For each, write Chain-of-Thought.

**Hard constraints**:
1. If `memory.multi_stage_pipeline` (P5) and `latency.async_pipeline` (P3) are both selected, they count as one — replace one with next applicable method on that axis.
2. Methods in `ineffective_methods` are **blocked** unless ncu bottleneck has fundamentally changed.
3. Methods in `implementation_failed_methods` require explicit acknowledgment of the prior failure.
4. All methods must be **arch-compatible** and **mutually orthogonal** (see catalog's Combining Rules).
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

Compiles and benchmarks all K branches (no ncu). Selects champion = fastest valid branch. Non-champions saved to `state.frontier`.

### 3f. Repair on validation failure (up to 3 retries per iteration)

If champion fails correctness, Claude rewrites and re-runs 3e.

### 3g. Profile champion with ncu (FULL report — mandatory)

```bash
python <skill>/scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter $i \
  --which kernel
```

Writes `iterv{i}/kernel.ncu-rep` — **every iteration must have a full ncu report on the champion**.

### 3h. Ablation attribution

```bash
python <skill>/scripts/ablate.py \
  --state ./run_*/state.json \
  --iter $i
```

For each method, generates an ablated kernel (champion minus that one method), benchmarks it. Computes attribution:
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

Runs `cuobjdump --dump-sass` on the compiled champion and greps for expected instruction patterns from `references/sass_signatures.json`. Writes `iterv{i}/sass_check.json`.

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
- Method enters `implementation_failed_methods` if: SASS check says signature missing
- Method enters `ineffective_methods` if: attribution ≤ noise_threshold but SASS was fine
- If `new_ms < best_ms` by more than noise_threshold → `best_file` updated
- Append record to `state.history` and `state.roofline_history`

---

## Step 4 — Final summary

```bash
python <skill>/scripts/summarize.py \
  --state ./run_*/state.json \
  --out ./run_*/summary.md
```

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
- **SASS signature missing but kernel is faster** → nvcc took a different path. Mark method as `implementation_failed` but keep the kernel if it's faster.
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
