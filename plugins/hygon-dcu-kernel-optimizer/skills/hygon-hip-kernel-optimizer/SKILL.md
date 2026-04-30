---
name: hygon-hip-kernel-optimizer
description: Iteratively optimize Hygon DCU HIP / CK Tile kernels against a Python reference using hipprof, DTK tools, dccobjdump ISA verification, roofline-style budgeting, branch selection, ablation attribution, and gfx936/gfx938-aware optimization references. Use when the user asks to optimize HIP kernels, Hygon DCU kernels, gfx936/gfx938 kernels, CK Tile kernels, port CUDA kernel tuning workflows to DCU, validate DCU ISA patterns, or reason about DCU-specific inline assembly and source-backed HCU/AMDGPU builtins.
---

# Hygon HIP Kernel Iterative Optimizer

## What this skill does

Optimize a Hygon DCU HIP or CK Tile kernel against a Python reference by running a measured loop:

1. validate environment and baseline/reference contract,
2. profile the current best kernel with `hipprof`,
3. classify compute, memory, and latency gaps into an axis budget,
4. select optimization methods from the DCU method registry,
5. generate K branch kernels with different implementation parameters,
6. compile, validate, and benchmark branches,
7. profile the champion, ablate selected methods, and verify ISA with `dccobjdump`,
8. update state and emit a final summary.

Use deterministic scripts for environment checks, profiling, benchmarking, ablation, ISA checks, state updates, and summaries. Use agent reasoning only for method selection, code changes, and repair.

## Key points

1. **Roofline-style axis budget**: allocate compute, memory, and latency method slots from measured DCU counters and timing.
2. **Branch-and-select**: generate several variants for the same method set, benchmark all valid branches, and keep the fastest champion.
3. **Ablation attribution**: keep a method only when removing it measurably hurts the champion.
4. **DCU ISA verification**: use `dccobjdump` patterns from `references/dcu_isa_signatures.json`; final proof is generated ISA, not source intent.
5. **Source-backed builtin discipline**: HCU or AMD-named builtins are candidates only when the exact call shape is backed by DCU KB source, a compile probe, or existing project code. `__has_builtin` failure alone is not enough to reject a source-backed builtin.
6. **CK Tile first**: prefer CK Tile for GEMM/conv/norm/MoE template work; do not port CUTLASS assumptions directly.
7. **Deep search on ambiguity**: for unclear hardware errors, unexplained performance regressions, or compiler/tool behavior that does not match expectation, search the local DCU knowledge base and source-backed reference projects before guessing. Web search is allowed when local references are insufficient.

## Inputs

Have these before starting:

- baseline kernel file: `.hip`, `.cu`, `.cpp`, `.cc`, `.cxx`, or `.py`
- Python reference file exposing `reference(**kwargs)`
- dimension JSON such as `{"N":1048576}` or `{"M":4096,"N":4096,"K":4096}`

If the user only provides a reference file and shape, first use the sibling `hygon-hip-baseline-generator` skill to generate and correctness-validate `kernel.hip` plus the benchmark-compatible `ref.py`. Do not begin optimization iterations until that generated baseline passes preflight and benchmark correctness.

Optional:

- iterations, default `3`
- `--ncu-num`, retained as the top-K DCU metric count for CUDA skill compatibility
- branches per iteration, default `4`
- `--ptr-size` when benchmark allocation needs an explicit element count
- warmup and repeat counts for benchmark stability

If a required input is missing and cannot be inferred, ask once briefly.

## Environment

Run DCU validation through the target project's own remote workflow when the target DCU is remote. This plugin intentionally does not ship a fixed remote-work skill because login nodes, compute nodes, Docker usage, module setup, and sync rules vary by project. Read the target repository's `AGENTS.md`, `.codex/skills/`, `.agents/skills/`, README, or other local runbooks before remote execution.

When this skill is loaded from the `hygon-dcu-kernel-optimizer` plugin in another project, treat the current working directory as the target project root. Before creating probes, logs, or temporary validation cases, ensure `<project-root>/hygon_tmp/` exists. Prefer running the plugin helper:

```bash
python <plugin-root>/scripts/ensure_hygon_workspace.py --root <project-root>
```

If the helper path is not obvious, create `hygon_tmp/` with normal filesystem tools and add `hygon_tmp/` to the target project's `.gitignore` unless the user says not to.

`hygon_tmp/` is only a temporary scratch area for ad-hoc probes, smoke-test cases, generated traces, pulled logs, and other validation artifacts. Do not make fixed filenames under `hygon_tmp/` part of the skill contract, and do not treat any generated run there as a project source asset.

Required or expected tools:

- `hipcc`
- `hipprof`
- `dccobjdump`
- `rocminfo`
- `rocm-smi`
- Python 3.10+ with the project benchmark dependencies
- CK Tile headers for CK Tile kernels
- optional DTK analysis tools under `/opt/dtk`, including PMC/SQTT-related tools when available

Probe the environment first:

```bash
python <skill>/scripts/check_env.py --out ./env.json
```

The probe records gfx target, DTK tools, `hipprof` availability, `dccobjdump`, CK Tile include discovery, and degraded profiling flags. If counters are unavailable, continue with timing, source inspection, SQTT when useful, code-object resource analysis, and ISA evidence, but tell the user that profiling is degraded.

## Fast path commands

Use `orchestrate.py` for normal runs:

```bash
python <skill>/scripts/orchestrate.py setup \
  --baseline ./kernel.hip \
  --ref ./ref.py \
  --iterations 2 \
  --branches 2 \
  --ptr-size 1048576 \
  --dims '{"N":1048576}'

python <skill>/scripts/orchestrate.py open-iter \
  --run-dir ./run_YYYYMMDD_HHMMSS \
  --iter 1

# Agent writes iterv1/methods.json, analysis.md, and branch kernels.

python <skill>/scripts/orchestrate.py close-iter \
  --run-dir ./run_YYYYMMDD_HHMMSS \
  --iter 1

python <skill>/scripts/orchestrate.py finalize \
  --run-dir ./run_YYYYMMDD_HHMMSS
```

`setup` runs environment/preflight/state initialization and seeds the baseline. `open-iter` profiles the current best and writes `roofline.json`. `close-iter` validates methods, explores branches, profiles the champion, ablates, runs ISA checks, and updates state. `finalize` writes `summary.md`.

## Detailed loop

### Step 0: preflight

Use:

```bash
python <skill>/scripts/preflight.py \
  --baseline ./kernel.hip \
  --ref ./ref.py \
  --dims '{"N":1048576}'
```

Surface contract failures directly. Do not begin optimization if the reference cannot run or the baseline cannot be compiled/benchmarked.

### Step 1: initialize and seed baseline

Normal path:

```bash
python <skill>/scripts/orchestrate.py setup --baseline ... --ref ... --dims ...
```

Manual path:

```bash
python <skill>/scripts/state.py init --baseline ./kernel.hip --ref ./ref.py --dims '{"N":1048576}' --env ./env.json
python <skill>/scripts/run_iteration.py seed-baseline --state ./run_*/state.json
```

The run folder is `run_YYYYMMDD_HHMMSS/` beside the baseline and contains `state.json`, copied baseline artifacts, and benchmark results.

### Step 2: profile current best and budget methods

Use:

```bash
python <skill>/scripts/profile_hipprof.py \
  --state ./run_*/state.json \
  --iter 1 \
  --which best_input \
  --pmc-mode all

python <skill>/scripts/roofline.py \
  --state ./run_*/state.json \
  --iter 1
```

Read:

- `iterv{i}/dcu_top.json`
- `iterv{i}/roofline.json`
- `state.json`
- current `best_file`

If all gaps are near peak, stop early and summarize. Otherwise use the `axis_budget` to decide how many methods to select per axis.

If the optimizer has completed **three consecutive iterations** without a material additional improvement over the previous best, trigger an SQTT/tooling triage before selecting more source changes. "Material" normally means exceeding the configured noise threshold in `state.json` (default 2%) and being explainable by profiler/ISA evidence, not just a single noisy timing sample. The triage should:

1. run `profile_hipprof.py` on the current best or latest champion with `--pmc-mode none --sqtt-type 1 --sqtt-output-type 0 --sqtt-data-dir <itervN>/sqtt_json/`;
2. analyze artifacts with `scripts/analyze_sqtt.py`;
3. if `perfetto` is available locally or remotely, analyze representative `thread_trace_*.json` files with `scripts/analyze_perfetto_trace.py`;
4. use the SQTT/Perfetto evidence to decide whether the next methods should target waitcnt placement, issue stalls, branch divergence, LDS/bank behavior, cache/global memory pressure, or whether the kernel is already near the practical ceiling.

You may run SQTT earlier for ambiguous hardware errors or unexplained regressions, but do not make it a mandatory every-iteration cost.

### Step 3: select methods

Read these references in order, loading only the needed parts:

1. `references/optimization_catalog.md` for method intent, triggers, skip rules, and combining rules.
2. `references/method_registry.json` for machine-validated method ids, axes, priorities, requirements, and expected ISA signatures.
3. `references/dcu_metrics_guide.md` for metric-to-cause mapping and hipprof/PMC interpretation.
4. `references/dcu_isa_signatures.json` for final dccobjdump pattern names.

Selection rule:

1. For each axis with positive budget, scan methods by priority.
2. Skip methods already tried unless the bottleneck has changed materially.
3. Skip methods blocked by target architecture, datatype, layout, or previous implementation failure.
4. Prefer methods with direct evidence in `dcu_top.json`, `roofline.json`, or source inspection.
5. Select exactly `sum(axis_budget)` methods unless no valid method exists; if fewer are available, explain every missing slot in `analysis.md`.
6. Keep selected methods mutually compatible. Avoid choosing two methods that are just the same pipeline or tiling change in different words.

Write:

- `iterv{i}/methods.json` matching `templates/methods.schema.json`
- `iterv{i}/analysis.md` following `templates/iteration_report.md`

Validate before generating branches:

```bash
python <skill>/scripts/validate_methods.py \
  --methods ./run_*/iterv1/methods.json \
  --state ./run_*/state.json
```

### Step 4: generate branch kernels

Generate K branches under `iterv{i}/branches/b1..bK/`. All branches should implement the same selected method set, but vary implementation details:

- tile sizes and vector width,
- wave/block mapping,
- LDS layout and bank-conflict strategy,
- pipeline stage count,
- CK Tile policy names and template parameters,
- direct load/store path versus LDS staging,
- inline asm or builtin form when compiler output must be forced.

For GEMM/conv/norm/MoE-style kernels, prefer CK Tile strategies and known fast paths such as `TLS`, `MLS`, `WASP`, `cshuffle`, `wavelet`, `persistent`, `split-k`, `preshuffle`, and DS-read matrix variants when the operation shape fits.

### Step 5: branch explore and repair

Use:

```bash
python <skill>/scripts/branch_explore.py \
  --state ./run_*/state.json \
  --iter 1
```

or let `orchestrate.py close-iter` run it. If all branches fail, inspect branch `bench.json`, `bench.stderr.txt`, compiler logs, and validation errors. Repair the branch sources and rerun. Do not mark a method ineffective when the branch never compiled or never passed correctness.

### Step 6: profile champion, ablate, verify ISA

Use:

```bash
python <skill>/scripts/profile_hipprof.py \
  --state ./run_*/state.json \
  --iter 1 \
  --which kernel

python <skill>/scripts/ablate.py \
  --state ./run_*/state.json \
  --iter 1

python <skill>/scripts/sass_check.py \
  --state ./run_*/state.json \
  --iter 1
```

`sass_check.py` is named for CUDA compatibility, but on this skill it runs DCU ISA verification with `dccobjdump` and DCU signature patterns.

### Step 7: update state and summarize

Use:

```bash
python <skill>/scripts/state.py update \
  --state ./run_*/state.json \
  --iter 1 \
  --kernel ./run_*/iterv1/kernel.hip \
  --bench ./run_*/iterv1/bench.json \
  --methods-json ./run_*/iterv1/methods.json \
  --attribution ./run_*/iterv1/attribution.json \
  --sass-check ./run_*/iterv1/isa_check.json

python <skill>/scripts/summarize.py \
  --state ./run_*/state.json \
  --out ./run_*/summary.md
```

State rules:

- Add every attempted method to `selected_methods`.
- Add a method to `effective_methods` only when attribution is positive beyond noise and expected ISA evidence is present.
- Add a method to `ineffective_methods` when ISA evidence is present but attribution is not positive.
- Add a method to `implementation_failed_methods` when the code compiled but expected ISA evidence is missing from a relevant dump.
- If a branch is faster but ISA evidence for a claimed method is missing, keep the faster kernel if correct, but record that method as implementation-failed.

## Hygon-specific hard rules

- Treat wavefront size as 64. Recheck every CUDA warp-size assumption.
- Use CK Tile instead of CUTLASS for DCU template kernels.
- Use `hipprof --pmc --pmc-type 3` for regular PMC-style data when available. Use SQTT/stat-stall tooling when PMC cannot explain stalls and the environment supports it.
- Use `hipprof --pmc-read --pmc-type 3` and `hipprof --pmc-write --pmc-type 3` in addition to regular `--pmc` when memory-read/write behavior matters. The bundled profiler defaults to `--pmc-mode all` and merges those CSVs into `dcu_top.json`.
- Use `hipprof --codeobj-analyze <elf-or-so-file>` after compilation to inspect VGPR/SGPR/LDS pressure. Treat high register pressure as a first-class signal for register control, occupancy, and latency decisions.
- Use SQTT for ambiguous stalls or instruction-flow questions, and automatically consider it after three consecutive no-material-improvement iterations: `hipprof --sqtt --sqtt-type 1`, `stat_stall`, `stat_valu`, or `all` depending on trace size. Prefer `--sqtt-output-type 0` for JSON and `--sqtt-data-dir <dir>/` when traces are large. Analyze generated `thread_trace_*.json` with `scripts/analyze_sqtt.py`; when the Python `perfetto` package is available, use `scripts/analyze_perfetto_trace.py` for PerfettoSQL summaries. Keep large temporary traces under `hygon_tmp/` when they are diagnostic probes rather than run artifacts.
- SQTT export may require `llvm-objdump` in `PATH` because `hipprof` calls it internally while creating trace JSON. This is not a replacement for `dccobjdump`: the optimizer's DCU ISA verification and pattern checks must still use DTK `dccobjdump`.
- Use `dccobjdump --inputs=<binary> --show-sass --show-instruction-encoding --separate-functions` plus resource/symbol dumps when instruction, register, LDS, or occupancy evidence is needed.
- Treat dump files with no relevant vector/global/matrix instructions as inconclusive, not immediate implementation failure.
- When a hardware-related error message, profiler symptom, compiler lowering choice, waitcnt hazard, or performance degradation is unclear, use deep DCU KB search to find matching reference projects and inspect how their kernels implement the same pattern. If the local KB is insufficient, search the web for ROCm/AMD/CK Tile/HIP material and treat it as analogy until Hygon compilation and ISA verification confirm it.
- For memory methods, look for DCU global/buffer/flat load/store families, vector widths, LDS paths, coalescing symptoms, and `buffer_load_*_lds` or `raw_buffer_load_lds` when staging through LDS.
- For matrix or tensor paths, remember Hygon tensorcore-related instructions diverge from AMD naming. Use AMD/ROCm/MFMA material only as analogy unless `dccobjdump` proves the final Hygon `v_mmac` or matrix instruction.
- Do not introduce FP4 strategies; current Hygon DCU target does not expose an FP4 hardware path for this workflow.
- For gfx938, source-backed `__builtin_hcu_*` conversion, MMAC, matrix-load, and DS-read helpers may be used only with exact signatures from DCU KB or existing source examples. Compile-probe before relying on them.
- For gfx936, AMD-named `__builtin_amdgcn_*` MMAC forms and inline asm patterns may be candidates only when source-backed or probe-backed, then verified by final ISA.
- Do not invent builtin names from AMD documents, spreadsheet rows, or mnemonic guesses.
- If compiler scheduling or lowering blocks an optimization, use inline asm as a last resort and add the required `s_waitcnt`, `s_barrier`, and hazard handling.
- For global-to-LDS and LDS-to-compute pipelines, check `s_waitcnt vmcnt(0)` for global-load consumers and `s_waitcnt lgkmcnt(0)` for LDS/scalar consumers.
- For known matrix/LDS patterns, useful final-ISA families include `ds_read_m32x16_b16`, `ds_read_m32x16_b16_alt`, `ds_read_m32x32_b8`, `ds_read_m32x64_b4`, `ds_read_m32x8_b32`, `ds_permute_b32`, `ds_bpermute_b32`, `matrix_load`, `v_mmac`, `v_pk_*`, VOP3R/VOP3P, and resource wait instructions. Use the JSON signatures for exact matching.

## Builtin and asm verification workflow

When a method needs a builtin or inline asm:

1. Search DCU KB first for the exact gfx target, builtin name, call signature, and source example.
2. If uncertain, create or update a minimal probe under a task-specific scratch directory such as `hygon_tmp/<probe-name>/`.
3. Run the probe remotely with the target `--offload-arch`, using the actual probe path you just created, for example:

```bash
python3 <probe-path> --arch gfx938
```

4. Remove unsupported call forms from the method implementation or mark them unavailable for that target.
5. Verify the final optimized kernel with `dccobjdump`; compile success alone is not enough.

Use this hierarchy for evidence:

1. benchmark correctness and timing,
2. `hipprof`/PMC/SQTT bottleneck evidence,
3. source-backed compile probe for builtin or asm availability,
4. final `dccobjdump` ISA and resource evidence.

## Ambiguous hardware or performance behavior

When the optimizer hits unclear DCU behavior, do not stop at generic GPU advice. Continue investigation in this order:

1. Search the local DCU KB for the exact gfx target, tool output, mnemonic, builtin, compiler diagnostic, profiler counter, or CK Tile path.
2. Inspect source-backed reference projects found by the KB and copy only patterns whose call signatures, target guards, layout contracts, and wait rules are visible in source.
3. If local evidence is insufficient, search the web for ROCm, AMD GPU, HIP, LLVM AMDGPU, or CK Tile references. Mark those findings as analogies until Hygon `hipcc` and `dccobjdump` confirm them.
4. Build a minimal compile or runtime probe under a task-specific scratch directory in repository-root `hygon_tmp/`. Keep probe inputs, source, logs, PMC read/write outputs, SQTT JSON/HTML/stat files, code-object analysis logs, dumps, and summaries there, but do not reference those scratch filenames as stable workflow entry points.
5. Feed confirmed findings back into the branch implementation, method notes, or reference files. Remove or quarantine unsupported assumptions.

Use this path for unclear errors, unexpected slowdowns, profiler/tool contradictions, unsupported intrinsic questions, codegen surprises, and suspected waitcnt/LDS/MMAC hazards.

## Failure modes

- `hipprof` writes degraded or empty metrics: continue with timing and ISA evidence, but disclose degraded profiling.
- Hardware-specific errors or unclear performance regressions: search DCU KB and reference projects deeply, optionally use web sources as analogies, then create a minimal probe in `hygon_tmp/` before changing the main kernel.
- `dccobjdump` cannot find a binary or relevant function: inspect compile artifacts and symbol names before declaring a method failed.
- Expected ISA pattern is absent from a relevant dump: mark the method implementation-failed, even if the branch is fast.
- All branches fail correctness or compilation: repair source and retry; do not update method attribution from failed branches.
- Champion speedup comes only from hyperparameters: record methods with low attribution as ineffective even if the kernel is faster.
- Builtin exists in AMD material but not DCU source/probe: do not use it as a DCU claim.
- `__has_builtin` reports missing for source-backed HCU builtins: treat that probe as inconclusive and compile the exact call shape instead.
- Remote DCU access is unavailable: prepare local scripts/probes under `hygon_tmp/` and ask the user to run them remotely, then process returned logs.

## References

Load references only when needed:

- `references/optimization_catalog.md`: human-readable optimization catalog, triggers, skip rules, combining guidance, and DCU-specific strategies.
- `references/method_registry.json`: structured method ids, priorities, requirements, signatures, and validation metadata.
- `references/dcu_metrics_guide.md`: hipprof/PMC/SQTT metrics, bottleneck interpretation, and tool usage notes.
- `references/dcu_isa_signatures.json`: dccobjdump pattern groups for DCU ISA verification.
- `examples/walkthrough.md`: full walkthrough for debugging or demonstrating the workflow.

## Output contract

Each run creates:

```text
<baseline-dir>/run_YYYYMMDD_HHMMSS/
|-- env.json
|-- state.json
|-- baseline/
|   |-- <baseline copy>
|   `-- bench.json
|-- iterv1/
|   |-- analysis.md
|   |-- methods.json
|   |-- dcu_top.json
|   |-- roofline.json
|   |-- best_input.hipprof.csv
|   |-- best_input.hipprof.log
|   |-- best_input.hipprof.codeobj_analyze.log
|   |-- best_input.hipprof.sqtt_analysis.json
|   |-- kernel.<ext>
|   |-- kernel.hipprof.csv
|   |-- kernel.hipprof.log
|   |-- kernel.hipprof.codeobj_analyze.log
|   |-- kernel.hipprof.sqtt_analysis.json
|   |-- bench.json
|   |-- attribution.json
|   |-- isa_check.json
|   `-- branches/
|       |-- b1/
|       |-- b2/
|       `-- ...
|-- iterv2/
`-- summary.md
```

The final user-facing answer should report best speedup, champion path, effective methods, implementation-failed methods, profiling/ISA caveats, and any remote validation that could not be run.
