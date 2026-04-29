# Example Walkthrough (DCU/HIP, Roofline-Driven, Branch-and-Select)

This walkthrough is intentionally long. It is not a marketing example; it is an operator runbook for an agent. It shows which files exist, which deterministic scripts run, what the agent must read, what it must write, and how `hipprof` plus `dccobjdump` evidence affects `state.json`.

The example uses a simple HIP kernel for one to three iterations on Hygon DCU. Real GEMM, attention, convolution, MoE, or normalization kernels follow the same loop, but will usually select CK Tile, MMAC, MLS, LDS, or SQTT-guided methods instead of the simple memory-access methods below.

## Layout Before The Run

```text
<case-dir>/
  kernel.hip
  ref.py
```

The baseline exports a host-side entry:

```cpp
extern "C" void solve(const float* x, float* y, int N) {
    constexpr int block_size = 128;
    int grid_size = (N + block_size - 1) / block_size;
    hipLaunchKernelGGL(my_kernel, dim3(grid_size), dim3(block_size), 0, 0, x, y, N);
}
```

The reference exposes:

```python
def reference(x, y, N):
    y.copy_(x * 2.0 + 1.0)
```

## Remote Execution Rule

All DCU validation must follow the target project's own remote workflow. Do not reuse this source repository's remote assumptions in another project. First read the target project's `AGENTS.md`, `.codex/skills/`, `.agents/skills/`, README, or local runbooks to determine whether validation uses a login node, compute node, Docker container, scheduler, module environment, or direct shell.

```text
target project workspace -> target project remote workflow -> Hygon DCU execution environment
```

In the target DCU execution environment the repository path must be whatever the target project's workflow defines:

```text
<target-remote-repo>
```

Temporary verification artifacts may stay under repository-root:

```text
hygon_tmp/
```

`hygon_tmp/` is scratch space only. Do not make exact filenames inside it part of the workflow contract. Do not edit remote environment files. Load modules and activate conda for the command session only.

## Step 0: Environment Check

The skill starts by detecting DTK and DCU tools:

```bash
python3 <optimizer-skill>/scripts/check_env.py \
  --out <scratch-dir>/env.json
```

Expected fields:

```json
{
  "gpu": "BW500SM",
  "gfx_arch": "gfx936",
  "hipcc": "/opt/dtk-25.04.4/bin/hipcc",
  "hipprof": true,
  "hipprof_pmc": true,
  "dccobjdump": true,
  "ck_tile": false,
  "torch": "2.9.0"
}
```

If `ck_tile` is false, CK Tile examples should be treated as source patterns only until headers are installed.

## Step 1: Preflight

Preflight validates the kernel entry point and Python reference:

```bash
python3 <optimizer-skill>/scripts/preflight.py \
  --baseline <case-dir>/kernel.hip \
  --ref <case-dir>/ref.py \
  --dims '{"N":1048576}'
```

The kernel must expose `extern "C" void solve(...)`. The reference must expose `reference(...)`. Pointer arguments are allocated as ROCm PyTorch GPU tensors by `benchmark.py`.

## Step 2: Setup

```bash
cd <target-remote-repo>

HIP_VISIBLE_DEVICES=0 python3 <optimizer-skill>/scripts/orchestrate.py setup \
  --baseline <case-dir>/kernel.hip \
  --ref <case-dir>/ref.py \
  --iterations 1 \
  --branches 2 \
  --dims '{"N":1048576}' \
  --ptr-size 1048576 \
  --warmup 2 \
  --repeat 5
```

`setup` performs:

1. `check_env.py`
2. `preflight.py`
3. `state.py init`
4. `run_iteration.py seed-baseline`
5. `profile_hipprof.py --which best_input` using `--pmc-mode all` by default, which merges `--pmc`, `--pmc-read`, and `--pmc-write` CSVs and runs `--codeobj-analyze` when the compiled object is available.
6. `roofline.py`

For this smoke test the baseline is about `0.0367 ms`, and `roofline.json` reports:

```json
{
  "bound": "bandwidth",
  "axis_budget": {
    "compute": 1,
    "memory": 2,
    "latency": 0
  }
}
```

Interpretation: the agent must pick one compute method and two memory methods.

## Plateau SQTT Triage

When three consecutive iterations fail to produce material additional improvement over the previous best, run SQTT and the analysis tools before choosing more source changes. Material improvement means above the run's noise threshold and supported by profiler/ISA evidence, not just a single timing fluctuation. You may also run this earlier for ambiguous hardware errors or unexplained regressions.

```bash
HIP_VISIBLE_DEVICES=0 python3 <optimizer-skill>/scripts/profile_hipprof.py \
  --state <run-dir>/state.json \
  --iter <plateau_iter> \
  --which kernel \
  --pmc-mode none \
  --sqtt-type 1 \
  --sqtt-output-type 0 \
  --sqtt-data-dir <run-dir>/iterv<plateau_iter>/sqtt_json/ \
  --kernel-name <kernel-filter> \
  --no-codeobj-analyze
```

Then analyze the artifacts:

```bash
python3 <optimizer-skill>/scripts/analyze_sqtt.py \
  <run-dir>/iterv<plateau_iter> \
  --out <run-dir>/iterv<plateau_iter>/sqtt_analysis.json

python3 <optimizer-skill>/scripts/analyze_perfetto_trace.py \
  <run-dir>/iterv<plateau_iter> \
  --max-files 4 \
  --out <run-dir>/iterv<plateau_iter>/perfetto_analysis.json
```

If only `.sqtt.csv` is produced and no `thread_trace_*.json` appears, inspect `kernel.hipprof.log`. DTK hipprof may need `llvm-objdump` in the SQTT subprocess `PATH` to export JSON. This is a hipprof export dependency only; final ISA verification still uses `dccobjdump`.

## Step 3: Agent Reads Evidence

The agent reads only the files needed for the next decision:

```text
run_*/state.json
run_*/iterv1/roofline.json
run_*/iterv1/dcu_top.json
skills/hyhon-hip-kernel-optimizer/references/optimization_catalog.md
skills/hyhon-hip-kernel-optimizer/references/dcu_metrics_guide.md
skills/hyhon-hip-kernel-optimizer/references/method_registry.json
skills/hyhon-hip-kernel-optimizer/templates/methods.schema.json
skills/hyhon-hip-kernel-optimizer/templates/iteration_report.md
```

For this example, the selected methods are:

| Axis | Budget | Method id | Why |
| --- | ---: | --- | --- |
| compute | 1 | `compute.launch_config_wave64` | Try a block size aligned with DCU wave64 execution. |
| memory | 2 | `memory.coalesced_access` | Preserve contiguous lane-to-address mapping. |
| memory | - | `memory.vectorized_global_access` | Test packed `float4` global load/store. |

Higher-priority compute methods such as MMAC and FP8/BF8 are skipped because this is an elementwise kernel, not a matrix/tensor-core workload.

## Step 4: Agent Writes Methods

The agent writes `iterv1/methods.json`:

```json
{
  "iter": 1,
  "methods": [
    {
      "id": "compute.launch_config_wave64",
      "name": "Launch configuration for DCU wave64",
      "axis": "compute",
      "priority": 3,
      "description": "Use a wave64-friendly block size.",
      "skipped_higher": [
        {"id": "compute.mmac_tensor_core", "priority": 1, "reason": "skip_condition"},
        {"id": "compute.mixed_precision_fp8_bf8", "priority": 2, "reason": "skip_condition"}
      ]
    },
    {
      "id": "memory.coalesced_access",
      "name": "Coalesced global memory access",
      "axis": "memory",
      "priority": 1,
      "description": "Keep adjacent lanes on adjacent elements.",
      "skipped_higher": []
    },
    {
      "id": "memory.vectorized_global_access",
      "name": "Vectorized global load/store",
      "axis": "memory",
      "priority": 2,
      "description": "Use aligned float4 loads/stores where legal.",
      "skipped_higher": []
    }
  ]
}
```

It also writes `iterv1/analysis.md`, using `templates/iteration_report.md` as a checklist.

## Step 5: Agent Writes Branch Kernels

The agent writes K branch kernels:

```text
run_*/iterv1/branches/b1/kernel.hip
run_*/iterv1/branches/b2/kernel.hip
```

Example branch choices:

| Branch | Change | Expected effect |
| --- | --- | --- |
| b1 | Scalar kernel, block size `256` | Better wave64-friendly block shape. |
| b2 | `float4` load/store, block size `256` | Fewer memory instructions per processed element. |

Every branch must preserve the `extern "C" void solve(...)` contract.

## Step 6: Close Iteration

```bash
HIP_VISIBLE_DEVICES=0 python3 <optimizer-skill>/scripts/orchestrate.py close-iter \
  --run-dir <run-dir> \
  --iter 1 \
  --warmup 2 \
  --repeat 5
```

`close-iter` performs:

1. `branch_explore.py`: compile and benchmark every branch.
2. Promote the fastest correct branch to `iterv1/kernel.hip`.
3. `profile_hipprof.py --which kernel`: profile the champion.
4. `sass_check.py`: collect DCU ISA/resource/symbol evidence with `dccobjdump`.
5. `state.py update`: update method buckets and best kernel.
6. `state.py set-best-hipprof-output`: record the champion profile when it becomes best.

Example branch result:

```json
{
  "champion": {
    "branch_index": 2,
    "ms": 0.00784
  },
  "branches": [
    {"branch_index": 1, "passed": true, "ms": 0.0197},
    {"branch_index": 2, "passed": true, "ms": 0.00784}
  ]
}
```

## DCU ISA Verification Details

`sass_check.py` follows the `dccobjdump` guide:

```bash
dccobjdump --inputs=kernel.so \
  --show-sass \
  --show-instruction-encoding \
  --separate-functions \
  --output=<existing-dir>

dccobjdump --inputs=kernel.so --show-all-fatbin --output=<existing-dir>
dccobjdump --inputs=kernel.so --show-symbols --show-resource-usage --show-kernel-descriptor --output=<existing-dir>
dccobjdump --inputs=kernel.so --list-elf
dccobjdump --inputs=kernel.so --extract-elf=all --output=<existing-dir>
```

Important detail: `dccobjdump --output=<dir>` expects the output directory to already exist. If the directory is missing, it may print only an ELF header and fail to create `.ISA` files.

The DCU ISA patterns use Hygon/AMD-style lowercase mnemonics:

```text
global_load_dword
global_load_dwordx2
global_load_dwordx4
global_store_dwordx4
buffer_load_dword*
buffer_store_dword*
flat_load_dword
flat_store_dword
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
ds_read*
ds_write*
v_mmac*
matrix_load*
```

For a controlled no-tail `float4` probe, the ISA check finds:

```json
{
  "vmem_instruction_count": 6,
  "checks": [
    {"method_id": "memory.coalesced_access", "verified": true},
    {"method_id": "memory.vectorized_global_access", "verified": true},
    {"method_id": "latency.waitcnt_pipeline", "verified": true}
  ]
}
```

For a tiny scalar-tail kernel, DTK `dccobjdump` can produce ISA files but no vector/global memory instruction lines. In that case the check is marked:

```json
{
  "inconclusive": true,
  "note": "dccobjdump produced no vector/global memory instructions; dump may be incomplete for this code object"
}
```

This is intentionally not classified as `implementation_failed_methods`.

## Step 7: Optional Ablation

When ablation kernels exist under:

```text
run_*/iterv1/ablations/<method_id_with_underscores>/kernel.hip
```

run:

```bash
HIP_VISIBLE_DEVICES=0 python3 <optimizer-skill>/scripts/ablate.py \
  --state <run-dir>/state.json \
  --iter 1
```

The output `attribution.json` estimates causal contribution:

```text
attribution_ms = ablated_ms - champion_ms
```

Positive attribution means removing the method slowed the kernel down.

## Step 8: Finalize

```bash
python3 <optimizer-skill>/scripts/orchestrate.py finalize \
  --run-dir <run-dir>
```

This writes `summary.md` with:

- environment summary,
- baseline and best runtime,
- best kernel and best `hipprof` output,
- roofline history,
- per-iteration timeline,
- effective methods,
- ineffective methods,
- implementation-failed methods,
- frontier branch candidates.

Example headline:

```text
Baseline time: 0.0367 ms
Best time:     0.0078 ms
Speedup:       4.72x
```

## Final Layout

```text
run_YYYYMMDD_HHMMSS/
  state.json
  summary.md
  baseline/
    kernel.hip
    kernel.so
    bench.json
  iterv1/
    kernel.hip
    kernel.so
    methods.json
    analysis.md
    roofline.json
    dcu_top.json
    best_input.hipprof.csv
    best_input.hipprof.log
    best_input.hipprof.codeobj_analyze.log
    best_input.hipprof.sqtt_analysis.json      # present only when SQTT was requested
    kernel.hipprof.csv
    kernel.hipprof.log
    kernel.hipprof.codeobj_analyze.log
    kernel.hipprof.sqtt_analysis.json          # present only when SQTT was requested
    isa_check.json
    bench.json
    branch_results.json
    attribution.json
    branches/
      b1/
        kernel.hip
        kernel.so
        bench.json
      b2/
        kernel.hip
        kernel.so
        bench.json
```

## What Changes For Real Kernels

| Kernel shape | First DCU methods to consider |
| --- | --- |
| GEMM / attention matmul | CK Tile, `compute.mmac_tensor_core`, `memory.matrix_load_mls`, LDS layout, waitcnt pipeline |
| FP8/BF8-tolerant matmul | `compute.mixed_precision_fp8_bf8`, gfx938 builtins, CK Tile FP8/BF8 paths |
| LDS-heavy tile kernel | `memory.lds_tiling`, `memory.lds_bank_conflict`, `latency.waitcnt_pipeline` |
| Wave-level exchange | `latency.wavefront_shuffle_ds_bpermute` |
| Compiler emits weak ISA | `compute.inline_asm_builtin`, verified by `dccobjdump` |

If the compiler does not emit the intended data path, inspect `rocminfo` and `state.json` for the exact `gfx` target, consult the `gfx936/gfx938` ISA notes, and use inline asm or `__builtin_hcu_*` only after a source-level HIP or CK Tile path cannot express the optimization.

## Key Differences From CUDA Walkthrough

| CUDA skill | DCU/HIP skill |
| --- | --- |
| `nvcc` | `hipcc` / DTK `dcc` |
| Nsight Compute `ncu` | `hipprof --pmc`, `--pmc-read`, `--pmc-write`, optional `hipprof --sqtt` |
| `cuobjdump` / SASS patterns | `dccobjdump` and DCU ISA mnemonics |
| CUTLASS templates | CK Tile templates |
| Warp size 32 assumptions | Wavefront size 64 assumptions |
| SASS signatures like HMMA/CP.ASYNC | DCU signatures like `v_mmac`, `global_load_dwordx4`, `ds_read*`, `matrix_load*`, `s_waitcnt` |
