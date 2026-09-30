#!/usr/bin/env python3
"""Generic operator benchmark with optional correctness validation.

Supported backends:
  - cuda: raw CUDA `.cu` kernels exposing `extern "C" void solve(...)`
  - cutlass: CUTLASS-based `.cu` kernels exposing the same `solve(...)` entry
  - triton: Python modules exposing `setup(...)` + `run_kernel(...)`

When --ref is provided:
  1. Validates kernel correctness against the reference implementation.
     Exits immediately if validation fails.
  2. Benchmarks the reference implementation.
  3. Benchmarks the target kernel/module.
  4. Prints a combined summary with speedup.

When --ref is omitted:
  Benchmarks the target kernel/module only.
"""

import re
import hashlib
import os
import sys
try:
    from gpu_lock import gpu_lock
except ImportError:  # allow import via importlib in registry/unit tests
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gpu_lock import gpu_lock
import json
import copy
import glob
import subprocess
import ctypes
import argparse
import importlib.util
import math
import statistics
from pathlib import Path
import torch
from workload_matrix import generate_workload_matrix
from contract_check import check_contract

# ---------------------------------------------------------------------------
# Type tables
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = {
    "float*": ("float*", ctypes.c_void_p),
    "double*": ("double*", ctypes.c_void_p),
    "unsigned char*": ("unsigned char*", ctypes.c_void_p),
    "unsigned short*": ("unsigned short*", ctypes.c_void_p),
    "unsigned int*": ("unsigned int*", ctypes.c_void_p),
    "char*": ("char*", ctypes.c_void_p),
    "short*": ("short*", ctypes.c_void_p),
    "long*": ("long*", ctypes.c_void_p),
    "int*": ("int*", ctypes.c_void_p),
    "int": ("int", ctypes.c_int),
    "long": ("long", ctypes.c_long),
    "size_t": ("size_t", ctypes.c_size_t),
    "unsigned int": ("unsigned int", ctypes.c_uint),
    "unsigned short": ("unsigned short", ctypes.c_ushort),
    "unsigned char": ("unsigned char", ctypes.c_ubyte),
    "char": ("char", ctypes.c_char),
    "short": ("short", ctypes.c_short),
}

DTYPE_MAP = {
    "float*": torch.float32,
    "double*": torch.float64,
    "int*": torch.int32,
    "long*": torch.int64,
    "short*": torch.int16,
    "char*": torch.int8,
    "unsigned char*": torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*": getattr(torch, "uint32", torch.int32),
}

INT_TYPES = {"int", "long", "size_t", "unsigned int"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_solve_signature(cu_file: str):
    """Extract parameter list from `extern "C" void solve(...)` in a .cu file."""
    with open(cu_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f'Cannot find \'extern "C" void solve(...)\' in {cu_file}')

    raw = match.group(1)
    raw = re.sub(r"/\*.*?\*/", "", raw)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())

    params = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        is_const = "const" in token
        token_clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
        matched = False
        for key in sorted(SUPPORTED_TYPES.keys(), key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            m = re.match(rf"({base})\s+(\w+)", token_clean)
            if m:
                params.append((key, m.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token.strip()}'")

    return params



def detect_arch(device_index: int | None = None) -> str:
    """Auto-detect GPU compute capability and return sm_XX string."""
    if torch.cuda.is_available():
        if device_index is None:
            device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        return f"sm_{major}{minor}"
    return "sm_80"


_STRIP_INCLUDES = re.compile(r'^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$', re.MULTILINE)



def _preprocess_cu(cu_file: str) -> str:
    """Strip clang-specific includes that break nvcc. Returns path to clean file."""
    with open(cu_file, "r", encoding="utf-8") as f:
        src = f.read()
    cleaned = _STRIP_INCLUDES.sub("", src)
    if cleaned == src:
        return cu_file
    tmp = cu_file + ".nvcc_clean.cu"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return tmp



def find_cutlass_include_dir() -> str:
    """Find a CUTLASS include directory containing both `cutlass/` and `cute/`."""
    candidates = []

    env_root = os.environ.get("CUTLASS_PATH", "").strip()
    env_include = os.environ.get("CUTLASS_INCLUDE_DIR", "").strip()

    if env_root:
        candidates.extend([env_root, os.path.join(env_root, "include")])
    if env_include:
        candidates.append(env_include)

    candidates.extend(sorted(glob.glob("/usr/local/cutlass*/include")))
    candidates.extend([
        "/usr/local/cutlass/include",
        "/opt/cutlass/include",
    ])

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        resolved = os.path.abspath(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if os.path.isdir(os.path.join(resolved, "cutlass")) and os.path.isdir(os.path.join(resolved, "cute")):
            return resolved

    return ""



def compile_cu(cu_file: str, output_so: str, arch: str, nvcc_bin: str, backend: str = "cuda"):
    """Compile .cu to a shared library."""
    clean_file = _preprocess_cu(cu_file)
    try:
        source = Path(cu_file).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        source = ""
    cmd = [nvcc_bin]
    if os.name != "nt":
        cmd.extend(["-Xcompiler", "-fPIC"])
    else:
        cmd.extend([
            "-allow-unsupported-compiler",
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        ])

    if backend == "cutlass":
        cutlass_include_dir = find_cutlass_include_dir()
        if not cutlass_include_dir:
            if clean_file != cu_file and os.path.exists(clean_file):
                os.remove(clean_file)
            print(
                "Compilation failed:\n"
                "Cannot locate CUTLASS headers. Set CUTLASS_PATH or CUTLASS_INCLUDE_DIR, "
                "or install CUTLASS under a standard path such as /usr/local/cutlass*/include.",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd.extend(["-I", cutlass_include_dir])

    if "#include <cublas_v2.h>" in source or "#include <cublasLt.h>" in source:
        cmd.extend(["-lcublas", "-lcublasLt"])

    cmd.extend(["-shared", "-std=c++17", f"-arch={arch}", "-O3", "-o", output_so, clean_file])
    print(f"[compile] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        if clean_file != cu_file and os.path.exists(clean_file):
            os.remove(clean_file)
        print(f"Compilation failed:\n{exc}", file=sys.stderr)
        sys.exit(1)
    if clean_file != cu_file and os.path.exists(clean_file):
        os.remove(clean_file)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[compile] -> {output_so}")



def load_python_module(module_file: str, module_name: str):
    """Import a Python module from a file path and return its module object."""
    if not os.path.exists(module_file):
        raise FileNotFoundError(f"Module file not found: {module_file}")
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {module_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def load_reference(ref_file: str):
    """Import a Python reference file and return its module."""
    mod = load_python_module(ref_file, "_ref_module")
    if not hasattr(mod, "reference"):
        raise AttributeError(f"'{ref_file}' must define a `reference(**kwargs)` function.")
    return mod



def infer_backend(solution_file: str, backend: str) -> str:
    if backend != "auto":
        return backend
    suffix = os.path.splitext(solution_file)[1].lower()
    if suffix == ".py":
        return "triton"
    if suffix == ".cu":
        try:
            text = Path(solution_file).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return "cuda"
        stripped = re.sub(r"//.*", "", text)
        stripped = re.sub(r"/\*[\s\S]*?\*/", "", stripped)
        if re.search(r"#\s*include\s*<\s*(cutlass|cute)/", stripped) or re.search(r"\b(cutlass|cute)::", stripped):
            return "cutlass"
        return "cuda"
    return "cuda"



def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return copy.deepcopy(value)



def _reset_tensor_inputs(state):
    """Restore tensor_inputs to their pristine initial state in-place.

    Uses torch.Tensor.copy_() so that the underlying data_ptr() stays valid —
    critical for the CUDA/CUTLASS path, where kernel_call_args holds raw pointers
    that were captured at setup time. Does nothing if pristine snapshots are
    absent (backwards-compat with any caller that doesn't populate them).
    """
    pristine = state.get("pristine_tensors") or {}
    for name, tensor in state["tensor_inputs"].items():
        snap = pristine.get(name)
        if snap is not None:
            tensor.copy_(snap)



def _regenerate_pristine(state, new_seed):
    """Re-draw random values into pristine snapshots under a new seed.

    This also refreshes reference_inputs (for the tensor entries). Used by the
    multi-seed validation path — each seed gets a fresh independent random draw
    so seed-specific bugs surface.

    Signature info cached at setup time tells us which tensors are const inputs
    (re-draw random) versus outputs (zero-init). Non-tensor entries in
    reference_inputs are left alone.
    """
    torch.manual_seed(new_seed)
    sig = state.get("signature") or []
    const_names = {item["name"] for item in sig if item.get("is_const")}

    for name, snap in state["pristine_tensors"].items():
        if name in const_names:
            if snap.dtype.is_floating_point:
                snap.normal_()
            else:
                snap.random_()
        else:
            snap.zero_()
        # Keep reference_inputs in sync with pristine tensors (fresh clones).
        if name in state["reference_inputs"] and isinstance(state["reference_inputs"][name], torch.Tensor):
            state["reference_inputs"][name].copy_(snap)



def _determine_ptr_elems(int_values: list, ptr_size_override: int) -> int:
    """Calculate number of elements for pointer buffers from dimension values."""
    if ptr_size_override > 0:
        ptr_elems = ptr_size_override
    elif len(int_values) == 0:
        ptr_elems = 1024 * 1024
    elif len(int_values) == 1:
        ptr_elems = int_values[0]
    else:
        sv = sorted(int_values, reverse=True)
        ptr_elems = sv[0] * sv[1]
    return min(ptr_elems, 256 * 1024 * 1024)



def _fmt_vals(vals, width=10):
    """Format a list of numeric values for compact display."""
    return "[" + ", ".join(f"{v:>{width}.4f}" for v in vals) + "]"



def _color(text: str, ok: bool) -> str:
    """ANSI color: green for pass, red for fail (only when stdout is a tty)."""
    if not sys.stdout.isatty():
        return text
    code = "\033[92m" if ok else "\033[91m"
    return f"{code}{text}\033[0m"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_iterations(fn, warmup: int, repeat: int, batches: int = 1) -> list:
    """Run independent CUDA-event measurements.

    The previous implementation timed one aggregate window and duplicated its
    average, which made variance and stability appear perfect.  A separate
    event pair per invocation gives real samples while preserving GPU
    serialization and the existing warmup semantics.
    """
    samples = []
    for _ in range(max(1, int(batches))):
        for _ in range(max(0, int(warmup))):
            fn()
        torch.cuda.synchronize()
        for _ in range(max(1, int(repeat))):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            end_event.synchronize()
            samples.append(float(start_event.elapsed_time(end_event)))
    return samples



def _stats(times_ms: list):
    if not times_ms:
        raise ValueError("timing produced no samples")
    avg = statistics.mean(times_ms)
    med = statistics.median(times_ms)
    return avg, med, min(times_ms), max(times_ms)



def _bootstrap_ci(times_ms: list[float], statistic, confidence: float = 0.95, resamples: int = 400, seed: int = 17):
    """Return a deterministic percentile bootstrap CI for a timing statistic."""
    values = [float(x) for x in times_ms if math.isfinite(float(x))]
    if len(values) < 2:
        value = float(statistic(values)) if values else None
        return {"low_ms": value, "high_ms": value, "confidence": confidence, "resamples": 0}
    # A local deterministic PRNG avoids changing caller/global random state.
    import random
    rng = random.Random(seed)
    estimates = []
    n = len(values)
    for _ in range(max(50, int(resamples))):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(float(statistic(sample)))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = estimates[max(0, min(len(estimates) - 1, int(alpha * len(estimates))))]
    hi = estimates[max(0, min(len(estimates) - 1, int((1.0 - alpha) * len(estimates)) - 1))]
    return {"low_ms": lo, "high_ms": hi, "confidence": confidence, "resamples": len(estimates)}


def _robust_cv(times_ms: list[float], median_ms: float | None = None) -> float | None:
    values = [float(x) for x in times_ms if math.isfinite(float(x))]
    if not values:
        return None
    med = float(median_ms if median_ms is not None else statistics.median(values))
    if med <= 0:
        return None
    mad = statistics.median(abs(x - med) for x in values)
    return float(1.4826 * mad / med)


def _stats_dict(times_ms: list):
    avg, med, mn, mx = _stats(times_ms)
    ordered = sorted(times_ms)
    def percentile(p):
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * p
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    stdev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    cv = stdev / avg if avg > 0 else None
    robust_cv = _robust_cv(times_ms, med)
    # Use ordinary CV for compatibility, while robust CV is the stability gate
    # for short/noisy kernels where one outlier should not dominate selection.
    stability_cv = robust_cv if robust_cv is not None else cv
    return {
        "average_ms": avg,
        "median_ms": med,
        "min_ms": mn,
        "max_ms": mx,
        "p10_ms": percentile(0.10),
        "p90_ms": percentile(0.90),
        "stddev_ms": stdev,
        "cv": cv,
        "robust_cv": robust_cv,
        "confidence_interval_95": _bootstrap_ci(times_ms, statistics.median),
        "samples": len(times_ms),
        "stability": ("stable" if stability_cv is not None and stability_cv <= 0.05 else
                       "borderline" if stability_cv is not None and stability_cv <= 0.10 else "unstable"),
    }



def _write_json_out(path: str, payload: dict):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------


def _print_results(label, avg, med, mn, mx, total_ptr_bytes, ptr_elems, solution_file, dim_values, arch, ref_avg=None):
    """Print benchmark results table; append speedup line when ref_avg is given."""
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print()
    print("=" * 55)
    print(f"  {label}")
    print(f"  Target       : {os.path.basename(solution_file)}")
    print(f"  GPU          : {gpu_name}")
    print(f"  Arch         : {arch}")
    print(f"  Dims         : {dim_values}")
    print(f"  Buf/ptr      : {ptr_elems} elems")
    print("-" * 55)
    print(f"  Average      : {avg:>10.4f} ms")
    print(f"  Median       : {med:>10.4f} ms")
    print(f"  Min          : {mn:>10.4f} ms")
    print(f"  Max          : {mx:>10.4f} ms")
    if avg > 0:
        bw = total_ptr_bytes / (avg / 1000) / 1e9
        print(f"  ~Bandwidth   : {bw:>10.2f} GB/s  (all tensors, rough)")
    if ref_avg is not None and avg > 0:
        speedup = ref_avg / avg
        print(f"  Speedup      : {speedup:>10.2f}x  vs reference")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_outputs(kernel_tensors, ref_tensors, output_params, atol, rtol):
    """Compare kernel and reference output tensors. Returns True if all pass."""
    PREVIEW = 8
    print(f"\n[validate] {len(output_params)} output tensor(s)\n")

    all_pass = True
    for pname, ptype in output_params:
        kt = kernel_tensors[pname].float()
        rt = ref_tensors[pname].float()

        match = torch.allclose(kt, rt, atol=atol, rtol=rtol)
        if not match:
            all_pass = False

        max_diff = (kt - rt).abs().max().item()
        mean_diff = (kt - rt).abs().mean().item()
        rel_err = ((kt - rt).abs() / rt.abs().clamp(min=1e-8)).mean().item()

        status_str = _color("PASS" if match else "FAIL", match)
        print(f"  [{status_str}]  {pname}  ({ptype})")
        print(f"         max |delta|   = {max_diff:.6e}")
        print(f"         mean |delta|  = {mean_diff:.6e}")
        print(f"         mean rel  = {rel_err:.6e}")

        if not match:
            diff_mask = ~torch.isclose(kt, rt, atol=atol, rtol=rtol)
            bad_idx = diff_mask.nonzero()
            n_bad = bad_idx.shape[0]
            print(f"         mismatches: {n_bad} / {kt.numel()}")
            if n_bad > 0:
                coord = tuple(int(v) for v in bad_idx[0].tolist())
                flat_idx = int(torch.tensor(coord).dot(torch.tensor(kt.stride())).item()) if kt.ndim > 1 else coord[0]
                print(
                    f"         first bad   @ idx={flat_idx} coord={coord}:  "
                    f"kernel={kt[coord].item():.6f}  ref={rt[coord].item():.6f}"
                )

        k_preview = kernel_tensors[pname].reshape(-1)[:PREVIEW].float().cpu().tolist()
        r_preview = ref_tensors[pname].reshape(-1)[:PREVIEW].float().cpu().tolist()
        print(f"         kernel[:{PREVIEW}] = {_fmt_vals(k_preview)}")
        print(f"         ref   [:{PREVIEW}] = {_fmt_vals(r_preview)}")
        print()

    return all_pass


# ---------------------------------------------------------------------------
# Setup: compile + allocate buffers
# ---------------------------------------------------------------------------


def _load_artifact_manifest(path: str, solution_file: str, arch: str, backend: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    artifact = manifest.get("artifact", "")
    source_hash = hashlib.sha256(Path(solution_file).read_bytes()).hexdigest()
    if (manifest.get("backend") != backend or manifest.get("arch") != arch
            or manifest.get("source_sha256") != source_hash
            or not os.path.isfile(artifact)):
        raise ValueError("artifact manifest does not match backend/arch or artifact is missing")
    return artifact


def _setup_cuda(solution_file, dim_values, ptr_size_override, arch, nvcc_bin, seed=None,
                backend_name="cuda", artifact_manifest=""):
    params = parse_solve_signature(solution_file)
    sig_str = ", ".join(f"{'const ' if c else ''}{t} {n}" for t, n, c in params)
    print(f"[signature] solve({sig_str})\n")

    lib_ext = ".dll" if os.name == "nt" else ".so"
    so_file = os.path.splitext(solution_file)[0] + lib_ext
    if artifact_manifest:
        so_file = _load_artifact_manifest(artifact_manifest, solution_file, arch, backend_name)
    else:
        compile_cu(solution_file, so_file, arch, nvcc_bin, backend=backend_name)
    lib = ctypes.CDLL(so_file)

    for ptype, pname, _ in params:
        if ptype in INT_TYPES and pname not in dim_values:
            raise ValueError(f"Missing dimension: --{pname}=<value>  (required by kernel signature)")

    int_vals = [dim_values[pname] for ptype, pname, _ in params if ptype in INT_TYPES]
    ptr_elems = _determine_ptr_elems(int_vals, ptr_size_override)

    if seed is not None:
        torch.manual_seed(seed)

    tensor_inputs = {}        # mutable; kernel operates on these via data_ptr
    reference_inputs = {}      # pristine kwargs for ref_fn; tensors are INDEPENDENT clones
    pristine_tensors = {}      # in-memory snapshot for resetting tensor_inputs
    output_specs = []
    kernel_call_args = []
    argtypes = []

    print("[buffers]")
    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            if is_const:
                # Input (const) buffer: random, representative of real data distribution
                if dtype.is_floating_point:
                    tensor = torch.randn(ptr_elems, device="cuda", dtype=dtype)
                else:
                    tensor = torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_()
            else:
                # Output (non-const) buffer: ZERO-initialized to match real-world
                # callers that typically hand the kernel a freshly-allocated/zeroed
                # output tensor. Random-init here masks RMW / read-output bugs.
                tensor = torch.zeros(ptr_elems, device="cuda", dtype=dtype)
            tensor_inputs[pname] = tensor
            # Keep an independent pristine snapshot so we can reset between phases.
            pristine_tensors[pname] = tensor.clone()
            # Reference gets its OWN clone — decoupled from tensor_inputs so kernel
            # mutations can never leak into the reference path.
            reference_inputs[pname] = tensor.clone()
            if not is_const:
                output_specs.append((pname, ptype))
            kernel_call_args.append(ctypes.c_void_p(tensor.data_ptr()))
            argtypes.append(ctypes.c_void_p)
            role = "input" if is_const else "output"
            eb = tensor.element_size()
            print(
                f"  {pname:>10s} : {ptype:<16s} [{role:>6s}] "
                f"{ptr_elems} elems  ({ptr_elems * eb / 1024 / 1024:.1f} MB)"
            )
        elif ptype in SUPPORTED_TYPES:
            _, ctype = SUPPORTED_TYPES[ptype]
            val = dim_values[pname]
            reference_inputs[pname] = val
            kernel_call_args.append(ctype(val))
            argtypes.append(ctype)
            print(f"  {pname:>10s} : {ptype:<16s} = {val}")

    lib.solve.restype = None
    lib.solve.argtypes = argtypes

    total_ptr_bytes = sum(t.nelement() * t.element_size() for t in tensor_inputs.values())

    return {
        "backend": backend_name,
        "signature": [
            {"type": ptype, "name": pname, "is_const": is_const}
            for ptype, pname, is_const in params
        ],
        "callable": lambda: lib.solve(*kernel_call_args),
        "tensor_inputs": tensor_inputs,
        "reference_inputs": reference_inputs,
        "pristine_tensors": pristine_tensors,
        "output_specs": output_specs,
        "ptr_elems": ptr_elems,
        "total_ptr_bytes": total_ptr_bytes,
        "preview_tensors": [
            {
                "name": pname,
                "type": ptype,
                "role": "input" if is_const else "output",
                "tensor": tensor_inputs[pname],
            }
            for ptype, pname, is_const in params if ptype in DTYPE_MAP
        ],
    }



def _setup_triton(solution_file, dim_values, seed=None):
    module = load_python_module(solution_file, "_triton_kernel_module")
    if not hasattr(module, "setup"):
        raise AttributeError(f"'{solution_file}' must define a `setup(**kwargs)` function for Triton benchmarking.")
    if not hasattr(module, "run_kernel"):
        raise AttributeError(f"'{solution_file}' must define a `run_kernel(**kwargs)` function for Triton benchmarking.")

    setup_kwargs = dict(dim_values)
    setup_kwargs["seed"] = seed
    prepared = module.setup(**setup_kwargs)
    if not isinstance(prepared, dict):
        raise TypeError("Triton setup() must return a dict")

    setup_inputs = prepared.get("inputs")
    outputs = prepared.get("outputs")
    if not isinstance(setup_inputs, dict):
        raise TypeError("Triton setup() must return dict['inputs'] as a mapping")
    if not isinstance(outputs, (list, tuple)):
        raise TypeError("Triton setup() must return dict['outputs'] as a list/tuple")

    for name in outputs:
        if name not in setup_inputs:
            raise ValueError(f"Triton output '{name}' not found in setup()['inputs']")
        if not isinstance(setup_inputs[name], torch.Tensor):
            raise TypeError(f"Triton output '{name}' must be a torch.Tensor")

    # Zero out output tensors to match real-world callers (setup() cannot be
    # trusted to do this reliably — user code may have pre-filled them).
    outputs_set = set(outputs)
    for name in outputs:
        setup_inputs[name].zero_()

    # tensor_inputs: the tensors the kernel actually operates on (may be mutated).
    # Keep references into setup_inputs so run_kernel(**kernel_kwargs) works.
    tensor_inputs = {
        name: value for name, value in setup_inputs.items() if isinstance(value, torch.Tensor)
    }

    # pristine_tensors: independent snapshots taken BEFORE any kernel runs.
    pristine_tensors = {
        name: value.clone() for name, value in tensor_inputs.items()
    }

    # reference_inputs: INDEPENDENT kwargs dict for ref_fn. Tensors are clones,
    # non-tensors are deep-copied. Kernel mutations cannot leak into this path.
    reference_inputs = {
        name: (value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
        for name, value in setup_inputs.items()
    }

    # kernel_kwargs: what run_kernel() gets called with. Points at tensor_inputs
    # (mutable) for tensors; non-tensor args are shared as-is.
    kernel_kwargs = {
        name: (tensor_inputs[name] if name in tensor_inputs else value)
        for name, value in setup_inputs.items()
    }

    ptr_elems = sum(value.numel() for value in tensor_inputs.values())
    total_ptr_bytes = sum(value.nelement() * value.element_size() for value in tensor_inputs.values())

    preview_tensors = []
    for name, value in tensor_inputs.items():
        preview_tensors.append(
            {
                "name": name,
                "type": str(value.dtype).replace("torch.", ""),
                "role": "output" if name in outputs_set else "input",
                "tensor": value,
            }
        )

    signature = []
    for name, value in setup_inputs.items():
        if isinstance(value, torch.Tensor):
            signature.append({
                "type": f"tensor[{str(value.dtype).replace('torch.', '')}]",
                "name": name,
                "is_const": name not in outputs_set,
            })
        elif isinstance(value, int):
            signature.append({"type": "int", "name": name, "is_const": True})
        elif isinstance(value, float):
            signature.append({"type": "float", "name": name, "is_const": True})
        else:
            signature.append({"type": type(value).__name__, "name": name, "is_const": True})

    print("[triton setup]")
    for item in signature:
        role = "input" if item["is_const"] else "output"
        print(f"  {item['name']:>10s} : {item['type']:<24s} [{role:>6s}]")

    output_specs = []
    for name in outputs:
        dtype_name = str(setup_inputs[name].dtype).replace("torch.", "")
        output_specs.append((name, dtype_name))

    return {
        "backend": "triton",
        "signature": signature,
        "callable": lambda: module.run_kernel(**kernel_kwargs),
        "tensor_inputs": tensor_inputs,
        "reference_inputs": reference_inputs,
        "pristine_tensors": pristine_tensors,
        "output_specs": output_specs,
        "ptr_elems": ptr_elems,
        "total_ptr_bytes": total_ptr_bytes,
        "preview_tensors": preview_tensors,
    }



def _setup_backend(solution_file, backend, dim_values, ptr_size_override, arch, nvcc_bin, seed=None,
                   artifact_manifest=""):
    if backend == "triton":
        return _setup_triton(solution_file, dim_values, seed=seed)
    if backend in {"cuda", "cutlass"}:
        return _setup_cuda(
            solution_file,
            dim_values,
            ptr_size_override,
            arch,
            nvcc_bin,
            seed=seed,
            backend_name=backend,
            artifact_manifest=artifact_manifest,
        )
    raise ValueError(f"Unsupported backend: {backend}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(solution_file, ref_file, dim_values, warmup, repeat, ptr_size_override, arch, atol, rtol, seed,
        json_out="", nvcc_bin="nvcc", backend="auto", validation_seeds=None,
        artifact_manifest="", numerics_mode="reference", timing_batches=1,
        workload_meta=None):
    """Main benchmark pipeline."""
    if json_out:
        Path(json_out).unlink(missing_ok=True)
    resolved_backend = infer_backend(solution_file, backend)
    has_ref = bool(ref_file)

    ref_fn = None
    _atol = atol
    _rtol = rtol

    if has_ref:
        ref_mod = load_reference(ref_file)
        ref_fn = ref_mod.reference
        _atol = float(getattr(ref_mod, "atol", atol))
        _rtol = float(getattr(ref_mod, "rtol", rtol))
        print(f"[reference] {ref_file}  (atol={_atol}, rtol={_rtol})\n")

    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    result = {
        "solution_file": os.path.abspath(solution_file),
        "cu_file": os.path.abspath(solution_file),
        "backend": resolved_backend,
        "ref_file": os.path.abspath(ref_file) if has_ref else "",
        "has_reference": has_ref,
        "dims": dim_values,
        "workload": workload_meta or {"scale": "default", "requested_dims": dim_values, "realized_dims": dim_values},
        "warmup": warmup,
        "repeat": repeat,
        "ptr_size_override": ptr_size_override,
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "arch": arch,
        "seed": seed,
        "artifact_manifest": os.path.abspath(artifact_manifest) if artifact_manifest else "",
        "numerics_mode": numerics_mode,
        "correctness": {
            "checked": has_ref,
            "passed": None,
            "atol": _atol if has_ref else None,
            "rtol": _rtol if has_ref else None,
            "output_tensor_count": 0,
        },
        "kernel": None,
        "reference": None,
        "speedup_vs_reference": None,
        "error": None,
        "states": {
            "compile_pass": "inconclusive" if resolved_backend == "triton" else "fail",
            "contract_pass": "inconclusive",
            "correctness_pass": "inconclusive",
            "race_safe": "inconclusive",
            "timing_valid": "inconclusive",
        },
    }

    # Reject malformed ABI/dimension contracts before compiling or allocating
    # CUDA buffers. This keeps direct benchmark invocations consistent with
    # branch_explore's post-build contract gate.
    if resolved_backend in {"cuda", "cutlass"}:
        pre_contract = check_contract(solution_file, dims=dim_values, compile=False)
        result["contract"] = pre_contract
        result["states"]["contract_pass"] = pre_contract.get("contract_pass", "inconclusive")
        if pre_contract.get("contract_pass") == "fail":
            result["correctness"]["passed"] = False if has_ref else None
            result["error"] = {"code": "contract_failed", "stage": "contract_check",
                                "message": "; ".join(pre_contract.get("errors", []))}
            _write_json_out(json_out, result)
            raise SystemExit(1)

    try:
        state = _setup_backend(
            solution_file,
            resolved_backend,
            dim_values,
            ptr_size_override,
            arch,
            nvcc_bin,
            seed=seed if has_ref else None,
            artifact_manifest=artifact_manifest,
        )
    except ValueError as exc:
        message = str(exc)
        error_code = "missing_dimension" if "Missing dimension:" in message else "setup_value_error"
        if has_ref:
            result["correctness"]["passed"] = False
        result["error"] = {
            "code": error_code,
            "stage": "setup_cuda" if resolved_backend in {"cuda", "cutlass"} else "setup_backend",
            "message": message,
        }
        _write_json_out(json_out, result)
        raise
    except Exception as exc:
        if has_ref:
            result["correctness"]["passed"] = False
        result["error"] = {
            "code": "setup_failed",
            "stage": "setup_backend",
            "message": str(exc),
        }
        _write_json_out(json_out, result)
        raise
    result["signature"] = state["signature"]
    result["states"]["compile_pass"] = "pass"
    result["ptr_elems"] = state["ptr_elems"]
    result["total_ptr_bytes"] = state["total_ptr_bytes"]
    result["correctness"]["output_tensor_count"] = len(state["output_specs"])

    # Static P0 contract gate. Runtime correctness still remains authoritative,
    # but malformed ABI/dimension contracts must never reach GPU timing.
    if resolved_backend in {"cuda", "cutlass"}:
        contract = check_contract(solution_file, dims=dim_values, compile=False)
        result["contract"] = contract
        result["states"]["contract_pass"] = contract.get("contract_pass", "inconclusive")
        if contract.get("contract_pass") == "fail":
            result["correctness"]["passed"] = False if has_ref else None
            result["error"] = {
                "code": "contract_failed", "stage": "contract_check",
                "message": "; ".join(contract.get("errors", [])),
            }
            _write_json_out(json_out, result)
            raise SystemExit(1)
    else:
        result["contract"] = {"contract_pass": "inconclusive", "note": "Triton runtime handshake required"}

    if not state["output_specs"] and has_ref:
        print("\n[warn] No output tensors detected. Nothing to validate.", file=sys.stderr)

    if has_ref:
        # Multi-seed validation. Every seed starts from a clean pristine state and
        # uses independent reference_inputs — kernel mutations cannot leak across
        # iterations, and reference inputs stay decoupled from kernel outputs.
        seeds_to_check = list(validation_seeds) if validation_seeds else [seed]
        seeds_to_check = [s for s in seeds_to_check if s is not None]
        if not seeds_to_check:
            seeds_to_check = [seed]

        per_seed_results = []
        overall_pass = True
        for seed_idx, cur_seed in enumerate(seeds_to_check):
            if len(seeds_to_check) > 1:
                print(f"\n--- Validation seed {seed_idx + 1}/{len(seeds_to_check)}: seed={cur_seed} ---")

            # Re-seed and regenerate inputs by resetting pristine from a fresh random draw.
            # For seed=original-seed we already have the right pristine; for other seeds
            # we need to regenerate. We do this by re-seeding and redrawing into pristine.
            if seed_idx > 0 or (cur_seed is not None and cur_seed != seed):
                _regenerate_pristine(state, cur_seed)

            # Reset kernel buffers to pristine (critical: outputs must be zero'd again
            # before the kernel call so we're not passing the previous iteration's data).
            _reset_tensor_inputs(state)

            # Independent kwargs for ref_fn — cloned from pristine reference_inputs.
            ref_inputs = {
                name: clone_value(value) for name, value in state["reference_inputs"].items()
            }

            print("\n[kernel]    running ... ", end="", flush=True)
            state["callable"]()
            torch.cuda.synchronize()
            print("done")

            print("[reference] running ... ", end="", flush=True)
            ref_fn(**ref_inputs)
            torch.cuda.synchronize()
            print("done")

            kernel_outputs = {
                name: tensor for name, tensor in state["tensor_inputs"].items()
                if name in {spec[0] for spec in state["output_specs"]}
            }
            ref_outputs = {
                name: tensor for name, tensor in ref_inputs.items()
                if isinstance(tensor, torch.Tensor) and name in kernel_outputs
            }

            seed_passed = _validate_outputs(
                kernel_outputs,
                ref_outputs,
                state["output_specs"],
                _atol,
                _rtol,
            )
            per_seed_results.append({"seed": cur_seed, "passed": seed_passed})
            if not seed_passed:
                overall_pass = False

        validation_passed = overall_pass

        print("=" * 60)
        print(f"  Target     : {os.path.basename(solution_file)}")
        print(f"  Backend    : {resolved_backend}")
        print(f"  Reference  : {os.path.basename(ref_file)}")
        print(f"  GPU        : {gpu_name}")
        print(f"  Arch       : {arch}")
        print(f"  Dims       : {dim_values}")
        print(f"  Buf/ptr    : {state['ptr_elems']} elems")
        print(f"  Tolerance  : atol={_atol}  rtol={_rtol}")
        print(f"  Seeds      : {seeds_to_check}")
        print("-" * 60)
        result_str = "ALL PASS" if validation_passed else "FAILED"
        print(f"  Result     : {_color(result_str, validation_passed)}")
        if len(seeds_to_check) > 1:
            for entry in per_seed_results:
                tag = _color("PASS", entry["passed"]) if entry["passed"] else _color("FAIL", False)
                print(f"    seed={entry['seed']:>6}  {tag}")
        print("=" * 60)

        result["correctness"]["passed"] = validation_passed
        result["states"]["correctness_pass"] = "pass" if validation_passed else "fail"
        result["correctness"]["seeds"] = per_seed_results
        if not validation_passed:
            _write_json_out(json_out, result)
            sys.exit(1)

    times_ref = None
    if has_ref:
        # reference_inputs is pristine (never touched by kernel); clone for benchmark.
        ref_bench_inputs = {
            name: clone_value(value) for name, value in state["reference_inputs"].items()
        }
        print(f"\n[warmup] reference  {warmup} iterations ...")
        times_ref = _time_iterations(lambda: ref_fn(**ref_bench_inputs), warmup, repeat, timing_batches)
        print(f"[bench]  reference  {repeat} iterations ... done")

    if not has_ref:
        preview = 8
        print(f"\n[preview] first {preview} elements before kernel call:")
        for item in state["preview_tensors"]:
            tag = "IN " if item["role"] == "input" else "OUT"
            vals = item["tensor"].reshape(-1)[:preview].float().cpu().tolist()
            print(f"  {tag} {item['name']:>6s} = {_fmt_vals(vals)}")

        state["callable"]()
        torch.cuda.synchronize()

        print(f"\n[preview] first {preview} elements after 1 kernel call:")
        for item in state["preview_tensors"]:
            tag = "IN " if item["role"] == "input" else "OUT"
            vals = item["tensor"].reshape(-1)[:preview].float().cpu().tolist()
            print(f"  {tag} {item['name']:>6s} = {_fmt_vals(vals)}")

    # Reset kernel buffers to pristine state before timing. Without this, the
    # kernel's benchmark measures performance on whatever state the validation
    # step (or a previous iteration) left behind — for RMW kernels this means
    # accumulating garbage into outputs and producing stale timings.
    _reset_tensor_inputs(state)

    print(f"\n[warmup] kernel  {warmup} iterations ...")
    times_kernel = _time_iterations(state["callable"], warmup, repeat, timing_batches)
    print(f"[bench]  kernel  {repeat} iterations ... done")

    avg_k, med_k, mn_k, mx_k = _stats(times_kernel)
    result["kernel"] = _stats_dict(times_kernel)
    result["states"]["timing_valid"] = "pass" if result["kernel"].get("stability") != "unstable" else "fail"
    # The bundled ABI receives preallocated buffers. Preserve explicit timing
    # categories so selectors cannot mistake this for an allocation-inclusive
    # benchmark; adapters may replace the end-to-end record with a measured one.
    result["kernel_only_ms"] = result["kernel"].get("average_ms")
    result["end_to_end_ms"] = result["kernel"].get("average_ms")
    result["end_to_end"] = {
        "measured": False,
        "average_ms": result["end_to_end_ms"],
        "allocation_included": False,
        "workspace_bytes": 0,
        "launch_count": 1,
        "synchronization_count": 1,
        "allocator_cache_hit": None,
        "note": "bundled ABI uses preallocated tensors; use a benchmark adapter for allocation/layout costs",
    }
    result["kernel"]["bandwidth_gbps_rough"] = (
        state["total_ptr_bytes"] / (avg_k / 1000) / 1e9 if avg_k > 0 else None
    )

    target_label = f"{resolved_backend.upper()} Kernel"
    if has_ref:
        avg_r, med_r, mn_r, mx_r = _stats(times_ref)
        result["reference"] = _stats_dict(times_ref)
        result["reference"]["bandwidth_gbps_rough"] = (
            state["total_ptr_bytes"] / (avg_r / 1000) / 1e9 if avg_r > 0 else None
        )
        result["speedup_vs_reference"] = avg_r / avg_k if avg_k > 0 else None
        _print_results(
            target_label,
            avg_k,
            med_k,
            mn_k,
            mx_k,
            state["total_ptr_bytes"],
            state["ptr_elems"],
            solution_file,
            dim_values,
            arch,
            ref_avg=avg_r,
        )
        _print_results(
            f"Reference ({os.path.basename(ref_file)})",
            avg_r,
            med_r,
            mn_r,
            mx_r,
            state["total_ptr_bytes"],
            state["ptr_elems"],
            solution_file,
            dim_values,
            arch,
        )
    else:
        _print_results(
            target_label,
            avg_k,
            med_k,
            mn_k,
            mx_k,
            state["total_ptr_bytes"],
            state["ptr_elems"],
            solution_file,
            dim_values,
            arch,
        )

    _write_json_out(json_out, result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generic operator benchmark (CUDA/CUTLASS/Triton, with optional validation)",
        epilog=(
            "Dimension args: pass --NAME=VALUE for each shape/scalar arg.\n"
            "For CUDA/CUTLASS, the kernel must expose extern \"C\" void solve(...).\n"
            "For Triton, the module must define setup(**kwargs) and run_kernel(**kwargs).\n"
            "ref.py must define `reference(**kwargs)` and may set module-level atol/rtol."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("solution_file", help="Path to solution file (.cu or .py)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cuda", "cutlass", "triton"], help="Backend type")
    parser.add_argument("--ref", type=str, default="", help="Path to reference .py file; enables validation + reference benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=20, help="Benchmark iterations (default: 20)")
    parser.add_argument("--timing-batches", type=int, default=1,
                        help="Independent timing batches (default: 1; large validation can use 5)")
    parser.add_argument("--workload-matrix", default="", help="Optional JSON workload profile")
    parser.add_argument("--workload-scale", choices=["small", "medium", "large"], default="",
                        help="Select a generated workload scale")
    parser.add_argument("--max-working-set-mb", type=int, default=384)
    parser.add_argument("--hard-working-set-mb", type=int, default=512)
    parser.add_argument("--ptr-size", type=int, default=0, help="Override element count for all CUDA/CUTLASS pointer buffers")
    parser.add_argument("--arch", type=str, default="", help="GPU arch, e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for validation (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for validation (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input tensors when validating (default: 42)")
    parser.add_argument("--validation-seeds", type=str, default="",
                        help="Comma-separated list of seeds to validate against (e.g. '1,2,3,42'). "
                             "If set, runs validation once per seed and only reports PASS if ALL seeds pass. "
                             "Overrides --seed for validation (--seed is still used for single-shot timing).")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write structured benchmark results as JSON")
    parser.add_argument("--nvcc-bin", type=str, default="nvcc", help="NVCC executable or full path")
    parser.add_argument("--artifact-manifest", type=str, default="",
                        help="Use a validated prebuilt CUDA artifact manifest")
    parser.add_argument("--numerics-mode", choices=["reference", "strict", "approximate"],
                        default="reference")

    args, unknown = parser.parse_known_args()

    dim_values = {}
    for item in unknown:
        if item.startswith("--") and "=" in item:
            key, val = item[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{item}'", file=sys.stderr)

    workload_meta = None
    if args.workload_scale:
        cases = generate_workload_matrix(
            dim_values, args.ptr_size, profile=args.workload_matrix or None,
            target_mb=args.max_working_set_mb, hard_mb=args.hard_working_set_mb)
        selected = next(c for c in cases if c["scale"] == args.workload_scale)
        dim_values = selected["realized_dims"]
        workload_meta = selected
        if selected.get("realized_ptr_size"):
            args.ptr_size = selected["realized_ptr_size"]

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch(args.gpu)

    validation_seeds = None
    if args.validation_seeds.strip():
        try:
            validation_seeds = [int(s.strip()) for s in args.validation_seeds.split(",") if s.strip()]
        except ValueError as exc:
            print(f"Error: --validation-seeds must be comma-separated integers ({exc})", file=sys.stderr)
            sys.exit(2)

    with gpu_lock(args.gpu):
        run(
            solution_file=args.solution_file,
            ref_file=args.ref,
            dim_values=dim_values,
            warmup=args.warmup,
            repeat=args.repeat,
            ptr_size_override=args.ptr_size,
            arch=arch,
            atol=args.atol,
            rtol=args.rtol,
            seed=args.seed,
            json_out=args.json_out,
            nvcc_bin=args.nvcc_bin,
            backend=args.backend,
            validation_seeds=validation_seeds,
            artifact_manifest=args.artifact_manifest,
            numerics_mode=args.numerics_mode,
            timing_batches=args.timing_batches,
            workload_meta=workload_meta,
        )


if __name__ == "__main__":
    main()
