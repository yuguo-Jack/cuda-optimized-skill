---
name: hygon-hip-baseline-generator
description: Generate a Hygon DCU HIP/C++ baseline kernel and correctness harness from only a Torch, Triton, TileLang, or Python reference file plus shape JSON, then hand the validated baseline to the Hygon HIP kernel optimizer. Use when the user has no initial HIP/C++ kernel, asks to start from a ref implementation, provides only ref.py and shape/dims, or wants automatic baseline generation before iterative DCU optimization.
---

# Hygon HIP Baseline Generator

## Purpose

Create the missing baseline stage before `hyhon-hip-kernel-optimizer` runs. This skill turns a reference implementation plus shape into a case directory containing:

- `kernel.hip`: conservative HIP/C++ baseline exposing `extern "C" void solve(...)`;
- `ref.py`: benchmark-compatible reference wrapper exposing `reference(...)`;
- `baseline_manifest.json`: detected operation, signature, assumptions, and next commands;
- optional debug logs from compile/correctness repair.

Only start the iterative performance optimizer after this baseline passes correctness against the reference.

## Inputs

Required:

- reference file: usually `.py`, containing Torch, Triton, TileLang, or mixed Python code;
- shape JSON, for example `{"N":1048576}` or `{"M":1024,"N":1024,"K":1024}`.

Optional:

- case output directory;
- dtype/tolerance assumptions;
- expected output name when the reference returns a tensor;
- remote DCU device id for validation.

If the ref file does not clearly expose a runnable Python oracle, create a wrapper around the most trustworthy path first. Triton and TileLang kernels are often implementation references rather than correctness oracles; prefer a Torch equivalent in the same file when available.

## Workflow

### 1. Inspect the reference

Run:

```bash
python <skill>/scripts/inspect_ref.py \
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

### 2. Scaffold the baseline case

Run:

```bash
python <skill>/scripts/generate_baseline.py \
  --analysis <case-dir>/ref_analysis.json \
  --out-dir <case-dir> \
  --op auto
```

The generator handles common flat elementwise and naive matmul baselines. For unsupported or ambiguous refs, let the script produce the wrapper and manifest, then edit `kernel.hip` manually using the analysis.

Generated code is deliberately simple. It should be correct and easy to debug, not fast.

### 3. Validate locally when possible, remotely when DCU is required

Use the Hygon optimizer preflight and benchmark scripts:

```bash
python skills/hyhon-hip-kernel-optimizer/scripts/preflight.py \
  --baseline <case-dir>/kernel.hip \
  --ref <case-dir>/ref.py \
  --dims '<shape-json>' \
  --out <case-dir>/preflight.json

HIP_VISIBLE_DEVICES=<device> python skills/hyhon-hip-kernel-optimizer/scripts/benchmark.py \
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

### 4. Debug correctness before optimization

Do not start performance iteration while baseline correctness is failing.

Use this repair order:

1. inspect `preflight.json`, `baseline_bench.json`, and benchmark stderr;
2. confirm `solve(...)` argument names match `ref.py reference(...)`;
3. confirm output parameters are non-const pointer args and reference writes them in-place;
4. confirm flat tensor views match shape: elementwise `N`, matmul `A[M,K]`, `B[K,N]`, `C[M,N]`;
5. reduce to a tiny shape and add temporary debug prints or CPU-side reference checks;
6. only after correctness passes, remove debug code and continue.

Use `references/ref_to_baseline_patterns.md` for framework-specific conversion patterns.

### 5. Hand off to the optimizer

After `baseline_bench.json` reports correctness passed:

```bash
python skills/hyhon-hip-kernel-optimizer/scripts/orchestrate.py setup \
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
- Record unsupported assumptions in `baseline_manifest.json`; do not silently invent dtype, layout, broadcasting, or reduction semantics.
- Do not use `hygon_tmp/` as a committed interface. It is scratch only.

## Resources

- `scripts/inspect_ref.py`: AST-based reference inspection and operation classification.
- `scripts/generate_baseline.py`: wrapper and conservative HIP baseline scaffolding.
- `references/ref_to_baseline_patterns.md`: practical conversion and debugging guidance for Torch, Triton, and TileLang refs.
