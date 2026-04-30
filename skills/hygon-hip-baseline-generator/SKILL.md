---
name: hygon-hip-baseline-generator
description: Generate a Hygon DCU HIP/C++ baseline kernel and correctness harness from a Torch, Triton, TileLang, Python, or CUDA/C++ reference plus shape JSON, including evidence-backed CUDA-to-HIP/DCU conversion, then hand the validated baseline to the Hygon HIP kernel optimizer. Use when the user has no initial HIP/C++ kernel, asks to start from a ref implementation, provides only ref.py and shape/dims, needs CUDA source ported to HIP/DCU, or wants automatic baseline generation before iterative DCU optimization.
---

# Hygon HIP Baseline Generator

## Purpose

Create the missing baseline stage before `hygon-hip-kernel-optimizer` runs. This skill turns a reference implementation plus shape into a case directory containing:

- `kernel.hip`: conservative HIP/C++ baseline exposing `extern "C" void solve(...)`;
- `ref.py`: benchmark-compatible reference wrapper exposing `reference(...)`;
- `baseline_manifest.json`: detected operation, signature, assumptions, and next commands;
- optional debug logs from compile/correctness repair.

It also handles CUDA-to-HIP/DCU baseline conversion when the user provides CUDA `.cu`, `.cuh`, C++ extension, or CUDA-library code instead of a Torch/Triton/TileLang/Python reference. In that mode, the output is still a correctness-first HIP baseline plus benchmark-compatible reference, but the conversion must be evidence-backed rather than a blind rename pass.

Only start the iterative performance optimizer after this baseline passes correctness against the reference.

Path rule: never assume `skills/...` exists under the target project. Resolve scripts from the loaded skill file:

- `<baseline-skill>` is the directory containing this `SKILL.md`.
- `<optimizer-skill>` is the sibling directory `<baseline-skill>/../hygon-hip-kernel-optimizer`.
- Before running commands, verify these files exist: `<baseline-skill>/scripts/inspect_ref.py`, `<baseline-skill>/scripts/generate_baseline.py`, `<optimizer-skill>/scripts/preflight.py`, `<optimizer-skill>/scripts/benchmark.py`, and `<optimizer-skill>/scripts/orchestrate.py`.

## Inputs

Required:

- reference or source file: usually `.py`, containing Torch, Triton, TileLang, or mixed Python code; or CUDA/C++ source such as `.cu`, `.cuh`, `.cpp`, `.cc`, `.cxx`, or extension code that needs HIP/DCU conversion;
- shape JSON, for example `{"N":1048576}` or `{"M":1024,"N":1024,"K":1024}`.

Optional:

- case output directory;
- dtype/tolerance assumptions;
- expected output name when the reference returns a tensor;
- remote DCU device id for validation.

If the ref file does not clearly expose a runnable Python oracle, create a wrapper around the most trustworthy path first. Triton and TileLang kernels are often implementation references rather than correctness oracles; prefer a Torch equivalent in the same file when available.

If the input is CUDA source, preserve the original file and create converted HIP artifacts beside the generated case. Do not overwrite user CUDA sources. Prefer names such as `kernel_original.cu`, `kernel.hip`, `hipify_report.md`, and `cuda_to_hip_manifest.json` or include the same information in `baseline_manifest.json`.

## Workflow

### 1. Inspect the reference

Run:

```bash
python <baseline-skill>/scripts/inspect_ref.py \
  --ref <ref-file> \
  --dims '<shape-json>' \
  --out <case-dir>/ref_analysis.json
```

On PowerShell, prefer `--dims-file <case-dir>/shape.json` or escape JSON quotes explicitly.

Read `ref_analysis.json` before writing kernel code. It identifies imports, functions, decorators, likely reference function, tensor parameters, in-place output parameters, return style, and operation family.

Operation-family hints are intentionally conservative:

- `matmul`: `torch.matmul`, `@`, `tl.dot`, GEMM-like TileLang calls, or dims containing `M,N,K`;
- `elementwise`: vector-shaped refs with simple arithmetic, activations, masks, or one-output maps;
- `reduction`: `sum`, `max`, `mean`, norm, softmax-like patterns;
- `unknown`: requires agent-written baseline from source inspection.

### 2. Convert CUDA inputs when needed

When the source is CUDA/C++ rather than a direct Python/Torch/Triton/TileLang oracle, create a HIP/DCU baseline before normal scaffold/validation. Use this evidence order:

1. Search the local DCU RAG KB first. Start with the AMD/NVIDIA comparison entries and CUDA-to-HIP mapping tables, especially:
   - `comparisons/hip-cuda-programming-comparison.md`
   - `comparisons/rocm-vs-cuda.md`
   - `amd-knowledge-base/layer-2-compute-stack/hip/cuda-to-hip-porting.md`
   - `amd-knowledge-base/layer-6-extended/optimize-guides/L2-optional/cuda-runtime-api-hip.md`
   - library-specific tables such as `cublas-api-hip.md`, `cusparse-api-hip.md`, `curand`, `cufft`, `cusolver`, and `cub` mappings when those APIs appear.
2. Use ROCm HIPIFY documentation as the upstream rule source. Prefer `hipify-clang` for production or complex C++ because it parses CUDA with Clang and reports conversion failures; use `hipify-perl` only for quick/simple code. For PyTorch CUDA extensions, CMake-based PyTorch submodules, or projects with custom include rewriting, consult and follow the official `ROCm/hipify_torch` repository patterns and custom mapping support.
3. Run an automatic hipify pass when tools are available in the target environment, keeping logs:
   - `hipify-clang <file.cu> --cuda-path=<cuda-path> --print-stats -- <includes-and-defines>`
   - or `python hipify_cli.py --config-json <config>` for `hipify_torch` style projects.
   If `compile_commands.json` exists, prefer it so include paths and macros match the real build.
4. Review every unconverted symbol, warning, include, macro, launch wrapper, library call, and device intrinsic manually. HIPIFY is a starting point, not proof of correctness.

For missing or uncertain mappings, search the remote DTK/ROCm installation before deciding the API is unsupported. Some CUDA-like functions are not present in the KB or public docs but do exist in the installed DTK headers or libraries.

Use the target project's remote workflow to inspect the actual DCU environment. Search likely roots such as `/opt/dtk`, `/opt/rocm`, `/usr/include`, project-provided CK/hip headers, and active conda or module paths. Prefer `rg` when available:

```bash
rg -n "cudaFunction|cuFunction|hipFunction|rocFunction" /opt/dtk /opt/rocm /usr/include 2>/dev/null
```

If the exact CUDA name is absent, try systematic substitutions and library-family variants:

- `cuda*` -> `hip*`, `cuda*` constants/enums -> `hip*`
- `cu*` driver APIs -> `hip*` module/driver-style APIs where available
- `cublas*` -> `hipblas*` first, then `rocblas*`
- `cusparse*` -> `hipsparse*` first, then `rocsparse*`
- `curand*` -> `hiprand*` first, then `rocrand*`
- `cufft*` -> `hipfft*` first, then `rocfft*`
- `cusolver*` -> `hipsolver*` first, then `rocsolver*`
- `cub::` -> `hipcub::` first, then `rocprim::`
- CUDA headers such as `cuda_runtime.h`, `cuda_fp16.h`, and `cuda_bf16.h` -> HIP/ROCm equivalents such as `hip/hip_runtime.h`, `hip/hip_fp16.h`, and available bf16 headers verified in DTK.

Do not invent mappings from naming symmetry alone. A mapping is trusted only after at least one of these is true:

- the KB or ROCm/HIPIFY table documents it;
- `hipify-clang` or `hipify_torch` converts it and no later review contradicts it;
- the DTK/ROCm install contains a matching declaration, wrapper, sample, or library symbol;
- a minimal compile probe with `hipcc` succeeds on the target DCU toolchain.

Record the evidence for each non-obvious mapping in the manifest or report: original CUDA symbol, chosen HIP/DCU symbol, evidence source, and any caveat. If no credible mapping exists, keep a small compatibility wrapper or rewrite the operation using supported HIP/ROCm primitives, then validate correctness before optimization.

### 3. Scaffold the baseline case

Run:

```bash
python <baseline-skill>/scripts/generate_baseline.py \
  --analysis <case-dir>/ref_analysis.json \
  --out-dir <case-dir> \
  --op auto
```

The generator handles common flat elementwise and naive matmul baselines. For unsupported or ambiguous refs, let the script produce the wrapper and manifest, then edit `kernel.hip` manually using the analysis.

Generated code is deliberately simple. It should be correct and easy to debug, not fast.

### 4. Validate locally when possible, remotely when DCU is required

Use the Hygon optimizer preflight and benchmark scripts:

```bash
python <optimizer-skill>/scripts/preflight.py \
  --baseline <case-dir>/kernel.hip \
  --ref <case-dir>/ref.py \
  --dims '<shape-json>' \
  --out <case-dir>/preflight.json

HIP_VISIBLE_DEVICES=<device> python <optimizer-skill>/scripts/benchmark.py \
  <case-dir>/kernel.hip \
  --ref <case-dir>/ref.py \
  --ptr-size <num-elements> \
  --warmup 2 \
  --repeat 5 \
  --json-out <case-dir>/baseline_bench.json \
  --N=<N>
```

For matrix shapes, pass `--M=<M> --N=<N> --K=<K>` and set `--ptr-size` large enough for the largest flat input/output tensor, normally `max(M*K, K*N, M*N)`.

If validation must run on remote DCU, use the repository `remote-ssh-docker-workflow` skill. Keep generated cases in a normal case directory if they are project artifacts. Use repository-root `hygon_tmp/` only for scratch probes and temporary logs.

### 5. Debug correctness before optimization

Do not start performance iteration while baseline correctness is failing.

Use this repair order:

1. inspect `preflight.json`, `baseline_bench.json`, and benchmark stderr;
2. confirm `solve(...)` argument names match `ref.py reference(...)`;
3. confirm output parameters are non-const pointer args and reference writes them in-place;
4. confirm flat tensor views match shape: elementwise `N`, matmul `A[M,K]`, `B[K,N]`, `C[M,N]`;
5. reduce to a tiny shape and add temporary debug prints or CPU-side reference checks;
6. only after correctness passes, remove debug code and continue.

Use `references/ref_to_baseline_patterns.md` for framework-specific conversion patterns.

### 6. Hand off to the optimizer

After `baseline_bench.json` reports correctness passed:

```bash
python <optimizer-skill>/scripts/orchestrate.py setup \
  --baseline <case-dir>/kernel.hip \
  --ref <case-dir>/ref.py \
  --dims '<shape-json>' \
  --ptr-size <num-elements> \
  --iterations <user-selected-iterations> \
  --branches <branches-per-iteration>
```

The iteration count is not chosen by this skill. Ask the user if it was not supplied.

## Rules

- Treat reference semantics as the source of truth; baseline speed is secondary.
- Keep generated HIP baseline boring: one thread per output element, no aggressive tiling, no inline asm, no CK Tile unless the baseline cannot be expressed simply.
- Preserve the original reference file; generate an adapter `ref.py` instead of rewriting the user file.
- If the original reference returns a tensor, generated `ref.py` must copy it into an output tensor so the existing benchmark can compare outputs.
- If the original reference mutates output tensors, preserve those names in `solve(...)`.
- For Triton/TileLang files, do not assume the decorated kernel is the oracle. Prefer a plain Torch `reference`, `torch_ref`, `ref`, `forward`, or `golden` function when present.
- For CUDA inputs, use HIPIFY/KB/DTK evidence to create a conservative HIP baseline before performance work. Keep original CUDA files, record mapping evidence, and compile-probe uncertain conversions.
- Record unsupported assumptions in `baseline_manifest.json`; do not silently invent dtype, layout, broadcasting, or reduction semantics.
- Do not use `hygon_tmp/` as a committed interface. It is scratch only.

## Resources

- `scripts/inspect_ref.py`: AST-based reference inspection and operation classification.
- `scripts/generate_baseline.py`: wrapper and conservative HIP baseline scaffolding.
- `references/ref_to_baseline_patterns.md`: practical conversion and debugging guidance for Torch, Triton, and TileLang refs.
