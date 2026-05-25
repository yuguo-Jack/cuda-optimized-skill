"""Capture TorchInductor Triton autotune data.

Import this module before the first compiled-model execution. It monkey-patches
TorchInductor CachingAutotuner to log configs, shapes, timings, rough bandwidth,
and standalone kernel repro files under TRITON_CAPTURE_DIR.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import textwrap
import time
from pathlib import Path

import torch
import torch._inductor.runtime.triton_heuristics as _th


CAPTURE_DIR = Path(os.environ.get("TRITON_CAPTURE_DIR", "./autotune_kernels")).resolve()
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = CAPTURE_DIR / "autotune.log"
CAPTURE_SINGLE_CONFIG = os.environ.get("TRITON_CAPTURE_SINGLE_CONFIG", "0").lower() in {"1", "true", "yes"}

_logger = logging.getLogger("hygon_triton_capture")
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not _logger.handlers:
    _handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _logger.addHandler(_handler)


def _log(msg: str = "") -> None:
    _logger.info(msg)


def _arg_names(autotuner) -> list[str]:
    return list(getattr(getattr(autotuner, "fn", None), "arg_names", []) or [])


def _kernel_name(autotuner) -> str:
    return str(getattr(getattr(autotuner, "fn", None), "__name__", "unknown_kernel"))


def _calc_total_bytes(arg_names: list[str], args, kwargs, mutated_arg_names) -> int:
    mutated = set(mutated_arg_names or [])
    total = 0
    for name, value in zip(arg_names, args):
        if isinstance(value, torch.Tensor):
            nbytes = value.numel() * value.element_size()
            total += 2 * nbytes if name in mutated else nbytes
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            nbytes = value.numel() * value.element_size()
            total += 2 * nbytes if name in mutated else nbytes
    return int(total)


def _get_hw_bandwidth_gbps() -> float:
    try:
        from triton.testing import get_dram_gbps

        return float(get_dram_gbps())
    except Exception:
        return float("nan")


def _log_inputs(arg_names: list[str], args, kwargs) -> None:
    for name, value in zip(arg_names, args):
        if isinstance(value, torch.Tensor):
            _log(f"  {name:20s}: shape={list(value.shape)}  dtype={value.dtype}  stride={list(value.stride())}")
        else:
            _log(f"  {name:20s}: {value}")
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            _log(f"  {name:20s}: shape={list(value.shape)}  dtype={value.dtype}  stride={list(value.stride())}")
        else:
            _log(f"  {name:20s}: {value}")


def _save_kernel_with_inputs(autotuner, args, kwargs) -> None:
    kernel_name = _kernel_name(autotuner)
    arg_names = _arg_names(autotuner)
    filename = getattr(autotuner, "filename", None)
    py_path = CAPTURE_DIR / f"{kernel_name}.py"
    inputs_path = CAPTURE_DIR / f"{kernel_name}_inputs.pt"

    if not filename or not os.path.isfile(filename):
        _log(f"  [WARN] kernel source file not found: {filename}")
        return

    shutil.copy2(filename, py_path)

    saved_inputs = {}
    call_parts: list[str] = []
    scalar_args = {}

    for name, value in zip(arg_names, args):
        if isinstance(value, torch.Tensor):
            saved_inputs[name] = value.detach().cpu().clone()
            call_parts.append(f"inputs[{name!r}].cuda()")
        else:
            scalar_args[name] = value
            call_parts.append(repr(value))
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            saved_inputs[name] = value.detach().cpu().clone()
            call_parts.append(f"{name}=inputs[{name!r}].cuda()")
        else:
            scalar_args[name] = value
            call_parts.append(f"{name}={value!r}")

    torch.save(saved_inputs, inputs_path)

    mutated = list(getattr(autotuner, "mutated_arg_names", []) or [])
    call_args = ", ".join(call_parts)
    append_code = textwrap.dedent(f"""

    # =============================================================
    # Auto-generated standalone autotune runner
    # =============================================================
    def _capture_load_inputs(path):
        try:
            return torch.load(path, weights_only=True)
        except TypeError:
            return torch.load(path)


    def _capture_calc_bytes(inputs, mutated_arg_names):
        mutated = set(mutated_arg_names)
        total = 0
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                nbytes = tensor.numel() * tensor.element_size()
                total += 2 * nbytes if name in mutated else nbytes
        return int(total)


    if __name__ == "__main__":
        import builtins
        import os
        import math
        import torch

        try:
            from triton.testing import get_dram_gbps
        except Exception:
            get_dram_gbps = None

        inputs = _capture_load_inputs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "{kernel_name}_inputs.pt"))
        mutated_arg_names = {mutated!r}
        arg_names = {arg_names!r}
        scalar_args = {scalar_args!r}
        kernel = {kernel_name}
        kernel.precompile()
        num_configs = len(kernel.launchers)
        print("\\n[AUTOTUNE] kernel={kernel_name}  configs={{}}".format(num_configs))
        for name in arg_names:
            if name in inputs:
                t = inputs[name]
                print("  {{:20s}}: shape={{}}  dtype={{}}  stride={{}}".format(name, list(t.shape), t.dtype, list(t.stride())))
            elif name in scalar_args:
                print("  {{:20s}}: {{}}".format(name, scalar_args[name]))

        total_bytes = _capture_calc_bytes(inputs, mutated_arg_names)
        hw_bw = float(get_dram_gbps()) if get_dram_gbps else float("nan")
        timings = kernel.benchmark_all_configs({call_args})
        best_launcher = builtins.min(timings, key=timings.get)
        best_ms = float(timings[best_launcher])
        eff_bw = (total_bytes / 1e9) / (best_ms / 1e3) if best_ms > 0 else float("nan")

        print("\\n  {{:<60}} {{:>10}}".format("Config", "Time(ms)"))
        print("  " + "-" * 70)
        for launcher, ms in sorted(timings.items(), key=lambda item: item[1]):
            marker = "  BEST" if launcher is best_launcher else ""
            print("  {{:<60}} {{:>10.4f}}{{}}".format(str(launcher.config), float(ms), marker))

        print("\\n  Total tensor bytes      : {{:.2f}} MB".format(total_bytes / 1e6))
        print("  Best config time        : {{:.4f}} ms".format(best_ms))
        print("  Effective bandwidth     : {{:.2f}} GB/s".format(eff_bw))
        print("  HW peak bandwidth       : {{:.2f}} GB/s".format(hw_bw))
        util = eff_bw / hw_bw * 100 if hw_bw and not math.isnan(hw_bw) else float("nan")
        print("  Bandwidth utilization   : {{:.1f}}%".format(util))
    """)

    with open(py_path, "a", encoding="utf-8") as f:
        f.write(append_code)

    _log(f"  Kernel source saved    : {py_path}")
    _log(f"  Inputs saved           : {inputs_path}")
    _log(f"  Run standalone         : python {py_path}")


def _install_patch() -> None:
    if getattr(_th.CachingAutotuner, "_hygon_triton_capture_patch", False):
        return

    orig_benchmark_all_configs = _th.CachingAutotuner.benchmark_all_configs
    orig_autotune = _th.CachingAutotuner.autotune_to_one_config
    orig_run = getattr(_th.CachingAutotuner, "run", None)
    autotune_logged: set[str] = set()
    single_logged: set[str] = set()

    def patched_benchmark_all_configs(self, *args, **kwargs):
        timings = orig_benchmark_all_configs(self, *args, **kwargs)
        self._hygon_capture_last_timings = timings
        return timings

    def patched_autotune(self, *args, **kwargs):
        kernel_name = _kernel_name(self)
        autotune_logged.add(kernel_name)
        launchers = list(getattr(self, "launchers", []) or [])
        arg_names = _arg_names(self)
        _log("")
        _log("=" * 64)
        _log(f"[AUTOTUNE] kernel={kernel_name}  configs={len(launchers)}")
        _log_inputs(arg_names, args, kwargs)
        t0 = time.perf_counter()
        result = orig_autotune(self, *args, **kwargs)
        elapsed = time.perf_counter() - t0
        _log(f"  [TIMING] kernel={kernel_name}  autotune elapsed: {elapsed:.3f}s ({elapsed * 1000:.1f}ms)")

        timings = getattr(self, "_hygon_capture_last_timings", None)
        if timings:
            best_launcher = min(timings, key=timings.get)
            best_ms = float(timings[best_launcher])
            total_bytes = _calc_total_bytes(arg_names, args, kwargs, getattr(self, "mutated_arg_names", []))
            eff_bw = (total_bytes / 1e9) / (best_ms / 1e3) if best_ms > 0 else float("nan")
            hw_bw = _get_hw_bandwidth_gbps()
            util = eff_bw / hw_bw * 100 if hw_bw and not math.isnan(hw_bw) else float("nan")
            _log("")
            _log(f"  {'Config':<60} {'Time(ms)':>10}")
            _log(f"  {'-' * 70}")
            for launcher, ms in sorted(timings.items(), key=lambda item: item[1]):
                marker = "  BEST" if launcher is best_launcher else ""
                _log(f"  {str(launcher.config):<60} {float(ms):>10.4f}{marker}")
            _log("")
            _log(f"  Total tensor bytes      : {total_bytes / 1e6:.2f} MB")
            _log(f"  Best config time        : {best_ms:.4f} ms")
            _log(f"  Effective bandwidth     : {eff_bw:.2f} GB/s")
            _log(f"  HW peak bandwidth       : {hw_bw:.2f} GB/s")
            _log(f"  Bandwidth utilization   : {util:.1f}%")

        _save_kernel_with_inputs(self, args, kwargs)
        _log("=" * 64)
        return result

    def patched_run(self, *args, **kwargs):
        kernel_name = _kernel_name(self)
        should_log = (
            CAPTURE_SINGLE_CONFIG
            and kernel_name not in single_logged
            and kernel_name not in autotune_logged
        )
        result = orig_run(self, *args, **kwargs)
        if should_log and kernel_name not in autotune_logged:
            single_logged.add(kernel_name)
            arg_names = _arg_names(self)
            _log("")
            _log("=" * 64)
            _log(f"[SINGLE-CONFIG] kernel={kernel_name}  configs={len(getattr(self, 'launchers', []) or [])}")
            _log_inputs(arg_names, args, kwargs)
            total_bytes = _calc_total_bytes(arg_names, args, kwargs, getattr(self, "mutated_arg_names", []))
            _log(f"  Total tensor bytes      : {total_bytes / 1e6:.2f} MB")
            _log("  [NOTE] single config path; no autotune timing was collected")
            _save_kernel_with_inputs(self, args, kwargs)
            _log("=" * 64)
        return result

    _th.CachingAutotuner.benchmark_all_configs = patched_benchmark_all_configs
    _th.CachingAutotuner.autotune_to_one_config = patched_autotune
    if CAPTURE_SINGLE_CONFIG and orig_run is not None:
        _th.CachingAutotuner.run = patched_run
    _th.CachingAutotuner._hygon_triton_capture_patch = True
    _log(f"[capture] enabled dir={CAPTURE_DIR}")


_install_patch()

