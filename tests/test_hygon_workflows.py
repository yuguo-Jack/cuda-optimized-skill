"""CPU regression coverage for orchestration/evidence contracts; no HCU run claimed."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import types
import uuid
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HIP = ROOT / "skills/hygon-hip-kernel-optimizer/scripts"
BASE = ROOT / "skills/hygon-hip-baseline-generator/scripts"
TRITON = ROOT / "skills/hygon-triton-kernel-optimizer/scripts"
sys.path.insert(0, str(HIP))


def load(name, directory=HIP):
    spec = importlib.util.spec_from_file_location("tested_" + name, directory / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def good_bench(source, reference=None):
    exp = load("experiment")
    return {"correctness": {"checked": True, "passed": True},
            "kernel": exp.timing_stats([1., 1.01, .99, 1., 1.]),
            "source_sha256": exp.file_sha256(source),
            "reference_sha256": exp.file_sha256(reference) if reference else None}


@pytest.mark.parametrize("change", [{"correctness": {}}, {"error": "failed"},
                                      {"kernel": {"average_ms": float("nan")}},
                                      {"kernel": {"average_ms": 1., "samples_ms": [1., 5., 1., 2., 1.]}}])
def test_evidence_gate_rejects_invalid(change, tmp_path):
    source = tmp_path / "kernel.hip"
    source.write_text("source")
    bench = good_bench(source)
    bench.update(change)
    assert load("experiment").benchmark_gate(bench, source)[0] is False


def test_gate_source_change_and_real_samples(tmp_path):
    source = tmp_path / "kernel.hip"
    source.write_text("a")
    bench = good_bench(source)
    assert load("experiment").benchmark_gate(bench, source)[0]
    source.write_text("b")
    assert not load("experiment").benchmark_gate(bench, source)[0]


def test_crashed_process_cannot_reuse_stale_json(tmp_path):
    out = tmp_path / "bench.json"
    write(out, {"correctness": {"checked": True, "passed": True}})
    res = load("experiment").run_json([sys.executable, "-c", "raise SystemExit(2)"], out)
    assert res["process_returncode"] == 2
    assert res["correctness"]["passed"] is False


def test_raw_counts_are_not_utilization():
    report = load("roofline").compute_deltas({"compute": [{"name": "SQ_BUSY_CYCLES", "value": 300000}],
                 "memory": [{"name": "TCP_TOTAL_CACHE_ACCESSES", "value": 90000}],
                 "latency": [{"name": "SQ_WAVE_CYCLES", "value": 999}],
                 "sqtt_analysis": {"waitcnt_count": 900}}, {})
    assert all(report["delta_" + a] is None for a in ["compute", "memory", "latency"])


def test_normalized_metrics_need_definition_scope_and_source():
    rf = load("roofline")
    record = {"value": 25, "unit": "percent", "definition": "rate/peak", "scope": "dispatch=1", "source": "perf"}
    assert rf.compute_deltas({"normalized": {"compute": record}}, {})["delta_compute"] == .75
    record.pop("scope")
    assert rf.compute_deltas({"normalized": {"compute": record}}, {})["delta_compute"] is None


@pytest.mark.parametrize("dtype,values", [(torch.int64, [2**40, 2**40 + 1]), (torch.float64, [1., 1. + 1e-10])])
def test_output_validation_preserves_precision(dtype, values):
    assert not load("benchmark")._validate_outputs({"out": torch.tensor([values[0]], dtype=dtype)},
                {"out": torch.tensor([values[1]], dtype=dtype)}, [("out", str(dtype))], 0., 0.)


def test_no_outputs_nan_and_empty_output_are_distinct():
    bench = load("benchmark")
    assert not bench._validate_outputs({}, {}, [], 0, 0)
    assert not bench._validate_outputs({"o": torch.tensor([float("nan")])}, {"o": torch.tensor([float("nan")])}, [("o", "f32")], 0, 0)
    assert bench._validate_outputs({"o": torch.empty(0)}, {"o": torch.empty(0)}, [("o", "f32")], 0, 0)


def test_no_allocation_truncation():
    with pytest.raises(ValueError):
        load("benchmark")._determine_ptr_elems([], 300 * 1024 * 1024)


def test_signature_spacing_and_rejection_match(tmp_path):
    src = tmp_path / "kernel.hip"
    src.write_text('extern "C" void solve(const float *x, double*y, int N) {}')
    assert load("benchmark").parse_solve_signature(str(src)) == load("preflight")._parse_solve(str(src))
    src.write_text('extern "C" void solve(float* x[3]) {}')
    with pytest.raises(ValueError):
        load("benchmark").parse_solve_signature(str(src))


def test_arch_is_not_guessed():
    with patch.object(torch.cuda, "is_available", return_value=False), patch("subprocess.run", side_effect=OSError()):
        with pytest.raises(RuntimeError):
            load("benchmark").detect_arch()


def test_timing_resets_every_call_and_keeps_real_samples():
    bench = load("benchmark")
    class Event:
        value = 0
        def __init__(self, **kwargs): pass
        def record(self): pass
        def elapsed_time(self, other):
            Event.value += 1
            return Event.value
    events = []
    with patch.object(torch.cuda, "Event", Event), patch.object(torch.cuda, "synchronize"):
        times = bench._time_iterations(lambda: events.append("run"), 1, 3, lambda: events.append("reset"))
    assert times == [1, 2, 3]
    assert events == ["reset", "run"] * 4


def test_missing_ablation_is_unknown(tmp_path):
    it = tmp_path / "iterv1"
    it.mkdir()
    src, ref = it / "kernel.hip", tmp_path / "ref.py"
    src.write_text("kernel")
    ref.write_text("oracle")
    write(it / "bench.json", good_bench(src, ref))
    write(it / "methods.json", {"methods": [{"id": "memory.lds_tiling"}]})
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "ref_file": str(ref)})
    item = load("ablate").run(state, 1)["attributions"][0]
    assert item["contributed"] is None and item["attribution_ms"] is None


def test_invalid_ablation_does_not_prove_essential(tmp_path):
    it = tmp_path / "iterv1"
    it.mkdir()
    src, ref = it / "kernel.hip", tmp_path / "ref.py"
    src.write_text("kernel")
    ref.write_text("oracle")
    write(it / "bench.json", good_bench(src, ref))
    write(it / "methods.json", {"methods": [{"id": "memory.lds_tiling"}]})
    ab = it / "ablations/memory_lds_tiling/kernel.hip"
    ab.parent.mkdir(parents=True)
    ab.write_text("bad")
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "ref_file": str(ref)})
    mod = load("ablate")
    with patch.object(mod, "compare", return_value={"assessment": {"valid": False, "reason": "bad"}}):
        assert mod.run(state, 1)["attributions"][0]["contributed"] is None


def test_regex_presence_requires_review():
    m = load("sass_check")
    res = m.check_method("m", "v_mmac_f32\n", {"methods": {"m": {"isa_patterns": ["v_mmac"]}}})
    assert res["pattern_presence"] is True
    assert res["verified"] is False and res["inconclusive"] is True
    assert m.check_method("x", "", {})["verified"] is False


def test_capture_preserves_alias_stride_offset():
    cap = load("capture_inputs", TRITON)
    base = torch.arange(30, dtype=torch.float64)
    inputs = {"a": base[2:22:2], "b": base[4:24:2]}
    restored = cap.restore(cap.snapshot(inputs), "cpu")
    assert restored["a"].stride() == inputs["a"].stride()
    assert restored["a"].storage_offset() == 2
    assert torch.equal(restored["a"], inputs["a"])
    restored["a"][1] = 100
    assert restored["b"][0].item() == 100
    assert base[4].item() == 4


def test_capture_rejects_mixed_dtype_alias():
    cap = load("capture_inputs", TRITON)
    a = torch.arange(4, dtype=torch.int32)
    with pytest.raises(ValueError):
        cap.snapshot({"a": a, "b": a.view(torch.float32)})


def test_capture_generated_runner_is_valid_python(tmp_path):
    """Execute capture builder only, without installing the Inductor monkey patch."""
    tree = ast.parse((TRITON / "autotune_capture_patch.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_save_kernel_with_inputs", "_kernel_name", "_arg_names"}]
    namespace = dict(Path=Path, ast=ast, hashlib=hashlib, json=json, uuid=uuid, os=os, shutil=shutil,
                     textwrap=textwrap, torch=torch, snapshot=load("capture_inputs", TRITON).snapshot,
                     CAPTURE_DIR=tmp_path, _log=lambda *args: None, __file__=str(TRITON / "autotune_capture_patch.py"))
    exec(compile(ast.Module(body=functions, type_ignores=[]), "capture_functions", "exec"), namespace)
    src = tmp_path / "source.py"
    src.write_text('kernel = None\nif __name__ == "__main__":\n    raise RuntimeError("old entry")\n')
    tuner = types.SimpleNamespace(fn=types.SimpleNamespace(__name__="kernel", arg_names=["x"]), filename=str(src), mutated_arg_names=["x"])
    namespace["_save_kernel_with_inputs"](tuner, (torch.arange(10)[::2],), {"stream": 999})
    target = next(tmp_path.glob("kernel_*/kernel.py"))
    compile(target.read_text(), str(target), "exec")
    assert "torch.cuda.current_stream().cuda_stream" in target.read_text()
    assert 'raise RuntimeError("old entry")' not in target.read_text()
    payload = torch.load(target.with_name("kernel_inputs.pt"), weights_only=True)
    assert payload["views"]["x"]["stride"] == [2]
    namespace["_save_kernel_with_inputs"](tuner, (torch.arange(10), 123), {})
    error = next(tmp_path.glob("kernel_*/capture-error.txt"))
    assert "exceed known kernel arg_names" in error.read_text()
    assert not (error.parent / "capture.json").exists()


def test_baseline_adapter_does_not_pass_unaccepted_dims(tmp_path):
    src = tmp_path / "orig.py"
    src.write_text("def reference(a, b):\n    return a + b\n")
    inspector = load("inspect_ref", BASE)
    analysis = inspector.inspect_ref(str(src), {"N": 8})
    path = write(tmp_path / "analysis.json", analysis)
    manifest = load("generate_baseline", BASE).generate(path, str(tmp_path / "case"), "auto", "out")
    spec = importlib.util.spec_from_file_location("adapter", manifest["ref"])
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    a, b, out = torch.ones(8), torch.ones(8), torch.empty(8)
    adapter.reference(a, b, out, 8)
    assert torch.equal(out, a+b)


def test_unsupported_baseline_is_blocked_and_output_not_overwritten(tmp_path):
    src = tmp_path / "orig.py"
    src.write_text("def reference(a):\n    return a.sum()\n")
    analysis = load("inspect_ref", BASE).inspect_ref(str(src), {"N": 8})
    path = write(tmp_path / "analysis.json", analysis)
    gen = load("generate_baseline", BASE)
    out = tmp_path / "case"
    manifest = gen.generate(path, str(out), "auto", "out")
    assert manifest["status"] == "needs_manual_implementation"
    assert Path(manifest["kernel"]).read_text().startswith("#error")
    with pytest.raises(SystemExit):
        gen.generate(path, str(out), "auto", "out")


def test_suite_regression_is_rejected(tmp_path):
    mod = load("workload_suite")
    src = tmp_path / "k"
    src.write_text("source")
    state = {"baseline_file": str(src), "ref_file": str(src), "workloads": [{"id": "big", "dims": {"N": 1}, "seeds": [1], "max_regression_pct": 5}]}
    n = 0
    def fake(cmd, output, log):
        nonlocal n
        n += 1
        result = good_bench(src, src)
        if n % 2 == 0:
            result["kernel"] = load("experiment").timing_stats([1.2]*5)
        return result
    with patch.object(mod, "run_json", side_effect=fake):
        result = mod.evaluate(state, str(src), str(src), tmp_path / "suite", 1, 5)
    assert result["passed"] is False
    assert result["cases"][0]["dims"] == {"N": 1}


def test_no_recursive_capture_copy(tmp_path):
    m = load("collect_inductor_artifacts", TRITON)
    with pytest.raises(ValueError):
        m._copy_tree(tmp_path, tmp_path / "artifacts")


def test_llvm_ir_is_not_final_isa(tmp_path):
    (tmp_path / "a.ll").write_text("v_mmac_f32")
    (tmp_path / "amdgcn.ll").write_text("v_mmac_f32")
    (tmp_path / "a.amdgcn").write_text("v_mmac_f32")
    assert load("scan_amdgcn", TRITON).scan(str(tmp_path))["totals"]["v_mmac"] == 1
    assert load("scan_amdgcn", TRITON).scan(str(tmp_path / "amdgcn.ll"))["totals"]["v_mmac"] == 0


def test_method_selection_is_not_gfx_numeric_inheritance(tmp_path):
    root = tmp_path
    reg = HIP.parent / "references/method_registry.json"
    state = write(root / "state.json", {"run_dir": str(root), "env": {"gpus": [{"gfx_arch": "gfx938"}]}})
    data = {"iter": 1, "methods": [{"id": "memory.shaobo_tls", "axis": "memory", "priority": 12, "target_evidence": "probe",
                 "skipped_higher": [{"id": mid, "reason": "no_trigger"} for mid, m in json.loads(reg.read_text(encoding="utf-8"))["methods"].items() if m["axis"] == "memory" and m["priority"] < 12]}]}
    methods = write(root / "methods.json", data)
    ok, errors = load("validate_methods").validate(methods, state)
    assert not ok and any("gfx946" in error for error in errors)


def test_state_rejects_historical_timing_and_unpaired_attribution(tmp_path):
    mod = load("state")
    ref, kernel = tmp_path / "ref.py", tmp_path / "kernel.hip"
    ref.write_text("oracle")
    kernel.write_text("kernel")
    state = {"run_dir": str(tmp_path), "ref_file": str(ref), "reference_sha256": load("experiment").file_sha256(ref),
             "best_metric_ms": 2., "selected_methods": [], "effective_methods": [], "ineffective_methods": [],
             "implementation_failed_methods": [], "history": [], "roofline_history": [], "frontier": []}
    state_path = write(tmp_path / "state.json", state)
    methods = write(tmp_path / "methods.json", {"methods": [{"id": "m", "axis": "memory"}]})
    bench = write(tmp_path / "bench.json", good_bench(kernel, ref))
    args = argparse.Namespace(state=state_path, iter=1, kernel=str(kernel), bench=bench, methods_json=methods,
                              skip_validation=True, allow_ineffective=False, attribution=None, sass_check=None, retries=0)
    # A legacy regex success plus an ablation must not become semantic proof.
    args.attribution = write(tmp_path / "attribution.json", {"champion_source_sha256": load("experiment").file_sha256(kernel),
        "attributions": [{"method_id": "m", "validation_passed": True, "attribution_ms": .5, "contributed": True}]})
    args.sass_check = write(tmp_path / "isa_check.json", {"checks": [{"method_id": "m", "verified": True}]})
    mod.cmd_update(args)
    saved = json.loads(Path(state_path).read_text())
    assert saved["best_metric_ms"] == 2.
    assert saved["history"][0]["status"] == "unconfirmed"
    assert len(saved["unverified_methods"]) == 1
    assert saved["effective_methods"] == []
    with pytest.raises(SystemExit):
        mod.cmd_update(args)
    evidence = tmp_path / "iterv2/isa.txt"
    evidence.parent.mkdir()
    evidence.write_text("reviewed target-specific mechanism")
    write(evidence.with_name("mechanism-review.json"), {"source_sha256": load("experiment").file_sha256(kernel),
          "methods": [{"id": "m", "status": "verified", "explanation": "Fixture checks provenance only",
                       "artifact": "isa.txt", "artifact_sha256": load("experiment").file_sha256(evidence)}]})
    args.iter = 2
    mod.cmd_update(args)
    assert len(json.loads(Path(state_path).read_text())["effective_methods"]) == 0


def test_branch_explore_runs_processes_and_selects_only_valid_candidate(tmp_path):
    """Exercise actual subprocess JSON contract and champion path, using a CPU fake benchmark."""
    bench = tmp_path / "fake_bench.py"
    bench.write_text('''import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--json-out');p.add_argument('--ref');a,_=p.parse_known_args()
value=Path(a.source).read_text().strip();valid=value!='invalid';ms=float(value) if valid else .01
Path(a.json_out).write_text(json.dumps({'source_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
'reference_sha256':hashlib.sha256(Path(a.ref).read_bytes()).hexdigest(),
'correctness':{'checked':True,'passed':valid},'kernel':{'average_ms':ms,'samples_ms':[ms]*5}}))
''', encoding="utf-8")
    it = tmp_path / "iterv1"
    for idx, val in [(1, "2"), (2, "1"), (3, "invalid")]:
        source = it / f"branches/b{idx}/kernel.hip"
        source.parent.mkdir(parents=True)
        source.write_text(val)
    ref = tmp_path / "ref.py"
    ref.write_text("oracle")
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "ref_file": str(ref), "branches": 3,
                                           "benchmark_file": str(bench), "benchmark_sha256": load("experiment").file_sha256(bench)})
    report = load("branch_explore").run(state, 1)
    assert report["champion"]["branch_index"] == 2
    assert Path(report["champion"]["kernel"]).read_text() == "1"
    assert report["valid_branches"] == 2


def test_installer_backs_up_and_does_not_touch_other_skills(tmp_path):
    mod = load("install_hygon_skills", ROOT / "tools")
    dest = tmp_path / "installed"
    old = dest / mod.NAMES[0] / "SKILL.md"
    old.parent.mkdir(parents=True)
    old.write_text("old")
    unrelated = dest / "unrelated/SKILL.md"
    unrelated.parent.mkdir()
    unrelated.write_text("keep")
    report = mod.install(dest, tmp_path / "backups")
    assert (Path(report["backup"]) / mod.NAMES[0] / "SKILL.md").read_text() == "old"
    assert unrelated.read_text() == "keep"
    for name in mod.NAMES:
        assert mod.manifest(dest / name) == mod.manifest(ROOT / "skills" / name)


def test_gemm_not_inferred_from_dimension_names(tmp_path):
    src = tmp_path / "ref.py"
    src.write_text("def reference(a, b):\n    return a + b\n")
    assert load("inspect_ref", BASE).inspect_ref(str(src), {"M": 10, "N": 20, "K": 30})["classification"]["op_kind"] != "matmul"


def test_capture_log_new_metric_names(tmp_path):
    log = tmp_path / "log"
    log.write_text("[AUTOTUNE] kernel=k configs=1\n  Footprint rate estimate   : 1.3e2 GB/s\n")
    result = load("summarize_autotune_log", TRITON).parse_log(str(log))
    assert result["kernels"][0]["metrics"]["footprint_rate_estimate_value"] == 130.


def test_custom_benchmark_is_reused_and_frozen(tmp_path):
    mod = load("orchestrate")
    path = tmp_path / "bf16_adapter.py"
    path.write_text("original")
    state = {"benchmark_file": str(path), "benchmark_sha256": load("experiment").file_sha256(path)}
    args = argparse.Namespace(benchmark=None)
    mod._resolve_benchmark(args, state)
    assert args.benchmark == str(path.resolve())
    path.write_text("changed")
    with pytest.raises(SystemExit):
        mod._resolve_benchmark(args, state)


def test_hipprof_uses_selected_champion_not_stale_extension(tmp_path):
    mod = load("profile_hipprof")
    folder = tmp_path / "iterv1"
    folder.mkdir()
    (folder / "kernel.hip").write_text("stale")
    winner = folder / "kernel.py"
    winner.write_text("selected")
    write(folder / "branch_results.json", {"champion": {"kernel": str(winner)}})
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path)})
    with patch.object(sys, "argv", ["profile", "--state", state, "--iter", "1", "--which", "kernel"]), patch.object(mod.shutil, "which", return_value=None):
        mod.main()
    assert json.loads((folder / "dcu_top.json").read_text())["profiled_file"] == str(winner)
    assert json.loads((folder / "dcu_top.json").read_text())["tool"] == "hipprof"


def test_hipprof_named_group_and_legacy_are_distinct():
    mod = load("profile_hipprof")
    assert mod._pmc_plan("pmc", "out", "wave")[0][1] == "--pmc"
    assert mod._pmc_plan("read", "out")[0][1] == "--pmc-read"
    with pytest.raises(ValueError):
        mod._pmc_plan("all", "out", "wave")
    with patch.object(mod.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
        mod._run_hipprof(hipprof_bin="hipprof", out_prefix="out", benchmark_py="bench.py", solution="kernel.hip",
                        dims={}, ptr_size=0, warmup=1, repeat=3, pmc_group="wave", benchmark_json="captured.json")
    argv = run.call_args.args[0]
    assert argv[argv.index("--pmc") + 1] == "wave"
    assert "--pmc-read" not in argv
    assert argv[argv.index("--json-out") + 1] == "captured.json"


def test_hipprof_binary_receipt_rejects_stale_and_malformed(tmp_path):
    mod = load("profile_hipprof")
    exp = load("experiment")
    source, binary = tmp_path / "kernel.hip", tmp_path / "cache.so"
    source.write_text("current source")
    binary.write_bytes(b"current binary")
    sha = exp.file_sha256(source)
    receipt = write(tmp_path / "captured.json", {"source_sha256": sha, "build": {
        "source_sha256": sha, "binary": str(binary), "binary_sha256": exp.file_sha256(binary)}})
    assert mod._measured_binary([receipt], str(source)) == str(binary)
    source.write_text("new source")
    assert mod._measured_binary([receipt], str(source)) is None
    source.write_text("current source")
    binary.write_bytes(b"replaced binary")
    assert mod._measured_binary([receipt], str(source)) is None
    for value in [[], {"build": ["wrong shape"]}, {"build": None}]:
        write(Path(receipt), value)
        assert mod._measured_binary([receipt], str(source)) is None


@pytest.mark.parametrize("sqtt_rc,has_trace,degraded", [(1, True, True), (0, False, True), (0, True, False)])
def test_hipprof_sqtt_failure_is_not_hidden_by_pmc(tmp_path, sqtt_rc, has_trace, degraded):
    mod = load("profile_hipprof")
    kernel = tmp_path / "kernel.hip"
    kernel.write_text("test")
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "best_file": str(kernel)})
    trace = {"file_count": int(has_trace), "csv_files": [], "parse_errors": []}
    def capture(**kwargs):
        return (sqtt_rc if kwargs['collect_flag'] == '--sqtt' else 0), "test capture"
    args = ["profile", "--state", state, "--iter", "1", "--which", "best_input", "--sqtt-type", "1", "--no-codeobj-analyze"]
    with patch.object(sys, "argv", args), patch.object(mod.shutil, "which", return_value="hipprof"), \
         patch.object(mod.subprocess, "run", return_value=types.SimpleNamespace(stdout="--pmc --pmc-type --sqtt --sqtt-type", stderr="")), \
         patch.object(mod, "_run_hipprof", side_effect=capture), patch.object(mod, "_sqtt_env_with_llvm_objdump", return_value=({}, None)), \
         patch.object(mod, "analyze_sqtt_json", return_value=trace), patch.object(mod, "_parse_csv_metrics", return_value={"SQ_BUSY": {
             "value": 50, "axis": "compute", "samples": 1, "source_files": []}}):
        mod.main()
    assert json.loads((tmp_path / "iterv1/dcu_top.json").read_text())["degraded"] is degraded


def test_sqtt_excludes_benchmark_receipt(tmp_path):
    mod = load("analyze_sqtt")
    write(tmp_path / "capture.sqtt.benchmark.json", {"kernel": "s_waitcnt"})
    result = mod.analyze([str(tmp_path / "capture.sqtt")])
    assert result["file_count"] == 0 and result["instruction_count"] == 0
    write(tmp_path / "capture.sqtt.trace.json", {"traceEvents": [{"name": "s_waitcnt", "dur": 1}]})
    result = mod.analyze([str(tmp_path / "capture.sqtt")])
    assert result["file_count"] == 1 and result["waitcnt_count"] == 1


@pytest.mark.parametrize("arch,name", [("gfx92a:xnack-", "月英"), ("gfx948", "塞班b1"), ("GFX928", "孔明e")])
def test_target_names_preserve_hex_ids_without_claiming_capabilities(arch, name):
    target = load("hcu_targets").describe(arch)
    assert target["architecture_name"] == name
    assert target["gfx"] == arch.split(":")[0].lower()
    assert target["capabilities_verified"] is False


def test_unknown_target_does_not_inherit_hcu_capabilities():
    mod = load("hcu_targets")
    assert mod.describe("gfx949")["architecture_name"] is None
    assert mod.describe("gfx92a-invalid")["gfx"] is None


def test_hipprof_help_does_not_claim_working_hardware_collection():
    mod = load("check_env")
    with patch.object(mod, "_detect_tool", return_value={"available": True, "path": "hipprof"}), \
         patch.object(mod, "_run", return_value=(0, "--pmc default/read/wave", "")) as run:
        info = mod._detect_hipprof()
    assert run.call_args.args[0] == ["hipprof", "-h"]
    assert info["pmc_flag_advertised"] is True and info["pmc_available"] is None


def test_capture_key_separates_same_name_different_layouts():
    tree = ast.parse((TRITON / "autotune_capture_patch.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_capture_key"]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "capture_key", "exec"), namespace)
    key = namespace["_capture_key"]
    tuner = object()
    assert key(tuner, (torch.arange(4),), {}) != key(tuner, (torch.arange(8)[::2],), {})
    assert key(tuner, (torch.arange(4),), {"stream": 1}) == key(tuner, (torch.arange(4),), {"stream": 2})


def test_captured_timeout_replaces_previous_success_json(tmp_path):
    mod = load("run_captured_kernel", TRITON)
    output = tmp_path / "result.json"
    write(output, {"returncode": 0})
    argv = ["capture", "fake.py", "--json-out", str(output)]
    with patch.object(sys, "argv", argv), patch.object(mod.subprocess, "run", side_effect=mod.subprocess.TimeoutExpired("fake", 300, output=b"partial")):
        with pytest.raises(SystemExit) as exc:
            mod.main()
    assert exc.value.code == 124
    assert json.loads(output.read_text())["returncode"] == 124


def test_xprof_command_and_artifact_identity(tmp_path):
    mod = load("profile_hcu")
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "best_file": "kernel.hip", "profiler": "xprof"})
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if "--help" in command:
            return mod.subprocess.CompletedProcess(command, 0, "--sections --output-dir", "")
        Path(command[command.index("--output-dir")+1], "capture.perf").write_text("raw")
        return mod.subprocess.CompletedProcess(command, 0, "", "")
    with patch.object(sys, "argv", ["profile", "--state", state, "--iter", "1", "--which", "best_input"]), patch.object(mod.shutil, "which", return_value="xprof"), patch.object(mod.subprocess, "run", side_effect=run):
        mod.main()
    result = json.loads((tmp_path / "iterv1/dcu_top.json").read_text())
    assert result["tool"] == "xprof"
    assert any(p.endswith("capture.perf") for p in result["artifacts"])
    assert result["degraded"] is True  # Captured raw data still needs XCompute review.
    assert "--sections" in commands[-1] and "--pmc" not in commands[-1]


def test_no_profiling_is_not_reported_as_xprof(tmp_path):
    mod = load("profile_hcu")
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "best_file": "kernel.hip", "profiler": "none"})
    with patch.object(sys, "argv", ["profile", "--state", state, "--iter", "1", "--which", "best_input"]), patch.object(mod.subprocess, "run") as run:
        mod.main()
    run.assert_not_called()
    result = json.loads((tmp_path / "iterv1/dcu_top.json").read_text())
    assert result["tool"] == "none" and result["degraded"] is True


def test_auto_without_xprof_routes_only_to_hipprof(tmp_path):
    mod = load("profile_hcu")
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "profiler": "auto"})
    with patch.object(sys, "argv", ["profile", "--state", state, "--iter", "1", "--which", "best_input"]), patch.object(mod.shutil, "which", return_value=None), patch.object(mod.subprocess, "call", return_value=7) as call:
        with pytest.raises(SystemExit) as exc:
            mod.main()
    assert exc.value.code == 7
    assert Path(call.call_args.args[0][1]).name == "profile_hipprof.py"
    assert "--sections" not in call.call_args.args[0]


def test_hipprof_command_does_not_use_xprof_switches():
    mod = load("profile_hipprof")
    proc = mod.subprocess.CompletedProcess([], 0, "", "")
    with patch.object(mod.subprocess, "run", return_value=proc) as run:
        mod._run_hipprof(hipprof_bin="hipprof", out_prefix="out", benchmark_py="bench.py", solution="kernel.hip",
                        dims={}, ptr_size=0, warmup=1, repeat=1, kernel_name="target", collect_flag="--pmc", pmc_type="3")
    command = run.call_args.args[0]
    assert "--pmc" in command and "--pmc-type" in command and "--kernel-name" in command
    assert not {"--sections", "--output-dir", "--kernels", "--enable-sqtt"}.intersection(command)


def test_summary_preserves_per_iteration_tool_identity(tmp_path):
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "profiler": "auto"})
    for i, tool in [(1, "xprof"), (2, "hipprof"), (3, "none")]:
        write(tmp_path / f"iterv{i}/dcu_top.json", {"tool": tool, "raw_directory": f"raw_{tool}", "degraded": True})
    write(tmp_path / "iterv4/dcu_top.json", {"raw_directory": "legacy"})
    output = tmp_path / "summary.md"
    load("summarize").render(state, str(output))
    report = output.read_text(encoding="utf-8")
    for i, tool in [(1, "xprof"), (2, "hipprof"), (3, "none"), (4, "not_recorded")]:
        assert f"| iterv{i} | {tool} |" in report


def cli(script, *args):
    return subprocess.run([sys.executable, "-X", "utf8", str(HIP / script), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


@pytest.mark.parametrize("command", [
    ("orchestrate.py", "open-iter"), ("orchestrate.py", "close-iter"),
    ("run_iteration.py", "benchmark"), ("branch_explore.py",),
    ("profile_hcu.py", "--which", "kernel"), ("profile_hipprof.py", "--which", "kernel"),
    ("ablate.py",), ("sass_check.py",), ("roofline.py",),
])
def test_closed_iteration_entrypoints_preserve_evidence(tmp_path, command):
    state_path = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "history": [{"iter": 1}]})
    artifact = tmp_path / "iterv1/bench.json"
    write(artifact, {"evidence": "keep"})
    before = artifact.read_bytes()
    flag, value = ("--run-dir", tmp_path) if command[0] == "orchestrate.py" else ("--state", state_path)
    result = cli(*command, flag, value, "--iter", 1)
    assert result.returncode != 0 and "already closed" in result.stderr
    assert artifact.read_bytes() == before
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == ["iterv1/bench.json", "state.json"]


def test_reference_hash_is_required_for_validation(tmp_path):
    source, ref = tmp_path / "source", tmp_path / "ref"
    source.write_text("source")
    ref.write_text("oracle")
    mod = load("experiment")
    assert not mod.benchmark_gate(good_bench(source), source, ref)[0]
    bench = good_bench(source, ref)
    assert mod.benchmark_gate(bench, source, ref)[0]
    ref.write_text("changed")
    assert not mod.benchmark_gate(bench, source, ref)[0]


def test_workload_zero_regression_and_invalid_shapes(tmp_path):
    mod = load("workload_suite")
    case = {"id": "zero", "dims": {"N": 0}, "seeds": [1], "max_regression_pct": 0}
    path = write(tmp_path / "cases.json", {"cases": [case]})
    assert mod.load_cases(path) == [case]
    for invalid in [{"seeds": 42}, {"max_regression_pct": -1}, {"ptr_size": -1}]:
        write(Path(path), {"cases": [{**case, **invalid}]})
        with pytest.raises(ValueError):
            mod.load_cases(path)


def test_summary_supports_historical_missing_method_names(tmp_path):
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "history": [
        {"iter": 1, "status": "improved", "methods": ["memory.coalesced_access"], "method_names": [None]}]})
    output = tmp_path / "summary.md"
    load("summarize").render(state, str(output))
    assert "memory.coalesced_access" in output.read_text(encoding="utf-8")


def test_frozen_inputs_checked_before_seed_and_suite(tmp_path):
    source, ref, bench = (tmp_path / n for n in ("baseline.py", "ref.py", "bench.py"))
    for p in (source, ref, bench):
        p.write_text("original")
    exp = load("experiment")
    original = {"baseline_file": str(source), "baseline_source_sha256": exp.file_sha256(source),
                "ref_file": str(ref), "reference_sha256": exp.file_sha256(ref),
                "benchmark_file": str(bench), "benchmark_sha256": exp.file_sha256(bench)}
    for p in (source, ref, bench):
        p.write_text("changed")
        with pytest.raises(SystemExit, match="changed since setup"):
            exp.resolve_benchmark(original)
        p.write_text("original")


def test_stale_suite_cannot_promote_candidate(tmp_path):
    source, ref = tmp_path / "kernel.py", tmp_path / "ref.py"
    source.write_text("source")
    ref.write_text("oracle")
    exp, mod = load("experiment"), load("state")
    cases = [{"id": "case", "dims": {"N": 1}, "seeds": [1]}]
    state = {"run_dir": str(tmp_path), "ref_file": str(ref), "reference_sha256": exp.file_sha256(ref),
             "baseline_file": str(source), "benchmark_file": str(source), "workloads": cases,
             "history": [], "best_metric_ms": 2.}
    bench = good_bench(source, ref)
    identity = load("workload_suite").suite_identity(state, source, source)
    bench["workload_suite"] = {"passed": True, "weighted_speedup": 2., "identity": identity}
    # A summary measured for N=1 cannot be reused for N=2.
    state["workloads"][0]["dims"]["N"] = 2
    state_path = write(tmp_path / "state.json", state)
    args = argparse.Namespace(state=state_path, iter=1, kernel=str(source),
        bench=write(tmp_path / "bench.json", bench), methods_json=write(tmp_path / "methods.json", {"methods": []}),
        skip_validation=True, allow_ineffective=False, attribution=None, sass_check=None, retries=0)
    with pytest.raises(SystemExit, match="suite does not match"):
        mod.cmd_update(args)
    assert json.loads(Path(state_path).read_text()) == state


def test_cpu_run_lifecycle_with_custom_benchmark_and_matrix(tmp_path):
    """Actual subprocess wiring only; synthetic times make no GPU performance claim."""
    baseline, ref, benchmark = (tmp_path / n for n in ("baseline.py", "ref.py", "bench.py"))
    baseline.write_text("2")
    ref.write_text("oracle")
    benchmark.write_text('''import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--ref');p.add_argument('--json-out')
for flag,default in [('--seed',42),('--warmup',10),('--repeat',20),('--ptr-size',0)]: p.add_argument(flag,type=int,default=default)
a,rest=p.parse_known_args();ms=float(Path(a.source).read_text());digest=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
dims={item[2:].split('=')[0]:int(item.split('=')[1]) for item in rest}
Path(a.json_out).write_text(json.dumps({'source_sha256':digest(a.source),'reference_sha256':digest(a.ref),
'inputs_sha256':'synthetic fixed inputs','gpu_index':0,'gpu_name':'CPU MOCK','arch':'gfx938',
'signature':[{'name':'x','type':'fixture'}],'dims':dims,'seed':a.seed,'warmup':a.warmup,'repeat':a.repeat,'ptr_size_override':a.ptr_size,
'correctness':{'checked':True,'passed':True},'kernel':{'average_ms':ms,'samples_ms':[ms]*5}}))
''', encoding="utf-8")
    workloads = write(tmp_path / "workloads.json", {"cases": [
        {"id": "small", "dims": {"N": 8}, "seeds": [1, 2], "max_regression_pct": 0},
        {"id": "tail", "dims": {"N": 65}, "seeds": [1], "max_regression_pct": 0}]})
    result = cli("state.py", "init", "--baseline", baseline, "--ref", ref, "--benchmark", benchmark,
                 "--iterations", 1, "--branches", 2, "--workloads", workloads, "--profiler", "none")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    run_dir, state_path = Path(data["run_dir"]), Path(data["state"])
    for command in [("run_iteration.py", "seed-baseline", "--state", state_path),
                    ("orchestrate.py", "open-iter", "--run-dir", run_dir, "--iter", 1)]:
        result = cli(*command)
        assert result.returncode == 0, result.stderr
    it = run_dir / "iterv1"
    (it / "branches/b1/kernel.py").write_text("1")
    (it / "branches/b2/kernel.py").write_text("3")  # correct but regresses every case
    (it / "kernel.hip").write_text("stale extension")
    write(it / "methods.json", {"iter": 1, "methods": [{"id": "memory.coalesced_access", "axis": "memory",
          "priority": 1, "name": "coalescing", "description": "CPU wiring fixture; no hardware claim"}]})
    result = cli("orchestrate.py", "close-iter", "--run-dir", run_dir, "--iter", 1)
    assert result.returncode == 0, result.stderr
    state = json.loads(state_path.read_text())
    assert state["best_file"] == str(it / "kernel.py")
    assert state["best_suite_score"] == pytest.approx(2)
    assert state["effective_methods"] == [] and len(state["unverified_methods"]) == 1
    assert len(state["history"][0]["workload_suite"]["cases"]) == 3
    confirmation = json.loads((it / "confirmation/comparison.json").read_text())
    assert len(confirmation["pairs"]) == 12
    assert state["history"][0]["confirmation"]["round_speedups"] == [2.] * 4
    report = cli("orchestrate.py", "finalize", "--run-dir", run_dir)
    assert report.returncode == 0, report.stderr
    assert "Frozen workload weighted speedup" in (run_dir / "summary.md").read_text(encoding="utf-8")
    before = (run_dir / "baseline/bench.json").read_bytes()
    reseed = cli("run_iteration.py", "seed-baseline", "--state", state_path)
    assert reseed.returncode != 0 and "already seeded" in reseed.stderr
    assert (run_dir / "baseline/bench.json").read_bytes() == before


def test_standalone_benchmark_uses_selected_champion(tmp_path):
    mod = load("run_iteration")
    it = tmp_path / "iterv1"
    it.mkdir()
    (it / "kernel.hip").write_text("stale")
    winner, ref = it / "kernel.py", tmp_path / "ref.py"
    winner.write_text("winner")
    ref.write_text("oracle")
    write(it / "branch_results.json", {"champion": {"kernel": str(winner)}})
    state = write(tmp_path / "state.json", {"run_dir": str(tmp_path), "ref_file": str(ref)})
    def fake(**kwargs):
        assert kwargs["solution"] == str(winner)
        write(Path(kwargs["json_out"]), good_bench(winner, ref))
    with patch.object(mod, "_run_bench", side_effect=fake):
        mod.cmd_benchmark(argparse.Namespace(state=state, iter=1, benchmark=None, warmup=1, repeat=5))


def test_triton_collection_preserves_colliding_names_and_avoids_cache_recursion(tmp_path):
    mod = load("collect_inductor_artifacts", TRITON)
    for i in range(10):
        src = tmp_path / str(i) / "kernel.amdgcn"
        src.parent.mkdir()
        src.write_text(str(i))
        mod._copy_file(src, tmp_path / "collected")
    assert {p.read_text() for p in (tmp_path / "collected").iterdir()} == {str(i) for i in range(10)}
    args = argparse.Namespace(log="", capture_dir="", cache_root=str(tmp_path), kernel="kernel", out=str(tmp_path / "artifacts"))
    with pytest.raises(ValueError, match="source cache"):
        mod.collect(args)
    assert not Path(args.out).exists()


def comparison_fixture(tmp_path, ratios=(2., 2., 2., 2.), context_change=None):
    mod = load("paired_measurement")
    candidate, control, ref, benchmark = [tmp_path / n for n in ("candidate.hip", "control.hip", "ref.py", "bench.py")]
    for p in (candidate, control, ref, benchmark):
        p.write_text(p.name)
    state = {"run_dir": str(tmp_path), "best_file": str(control), "ref_file": str(ref),
             "benchmark_file": str(benchmark), "dims": {"N": 10}, "ptr_size": 10}
    orders = []
    def measure(cmd, output, log):
        r = int(Path(output).relative_to(tmp_path / "paired").parts[0])
        role = Path(output).stem
        orders.append(role)
        source = control if role == "control" else candidate
        ms = ratios[r] if role == "control" else 1.
        b = good_bench(source, ref)
        b.update(kernel=load("experiment").timing_stats([ms] * 5), inputs_sha256="fixture inputs",
                 gpu_index=0, gpu_name="CPU MOCK", arch="gfx938", signature=[{"name": "x", "type": "f32"}],
                 dims={"N": 10}, seed=42, warmup=10, repeat=20, ptr_size_override=10)
        if role == "candidate" and context_change:
            b.update(context_change)
        write(Path(output), b)
        return b
    with patch.object(mod, "run_json", side_effect=measure):
        report = mod.compare(state, str(candidate), str(control), str(benchmark), tmp_path / "paired")
    return mod, report, state, candidate, control, benchmark, orders


def test_fresh_comparison_alternates_and_rechecks_raw_evidence(tmp_path):
    mod, report, state, candidate, control, benchmark, orders = comparison_fixture(tmp_path)
    assert orders == ["control", "candidate", "candidate", "control"] * 2
    assert report["assessment"]["improved"] is True
    artifact = Path(report["pairs"][0]["candidate"]["artifact"])
    artifact.write_text("{}")
    assert not mod.assess(report, state, candidate, control, benchmark)["valid"]


@pytest.mark.parametrize("ratios,direction", [((1.01,)*4, "inconclusive"), ((1.1, 1.1, .9, 1.1), "inconclusive"),
                                               ((.8,)*4, "harmful")])
def test_noisy_or_opposite_comparisons_do_not_promote(tmp_path, ratios, direction):
    report = comparison_fixture(tmp_path, ratios)[1]
    assert not report["assessment"]["improved"]
    assert report["assessment"]["direction"] == direction


@pytest.mark.parametrize("change", [{"inputs_sha256": "different"}, {"arch": "gfx946"},
                                      {"warmup": 5}, {"dims": {"N": 5}}, {"gpu_index": 1}])
def test_comparison_rejects_changed_inputs_device_or_protocol(tmp_path, change):
    assert not comparison_fixture(tmp_path, context_change=change)[1]["assessment"]["valid"]


def test_comparison_requires_metadata_even_if_both_legacy_results_match(tmp_path):
    mod, report, state, candidate, control, benchmark, _ = comparison_fixture(tmp_path)
    for row in report["pairs"]:
        for role in ("candidate", "control"):
            path = Path(row[role]["artifact"])
            b = json.loads(path.read_text()); b.pop("inputs_sha256")
            write(path, b); row[role]["sha256"] = load("experiment").file_sha256(path)
    report["assessment"]["improved"] = True
    assert not mod.assess(report, state, candidate, control, benchmark)["valid"]


def test_state_promotes_only_confirmed_comparison(tmp_path):
    _, report, state, candidate, control, benchmark, _ = comparison_fixture(tmp_path)
    state.update(best_metric_ms=2., selected_methods=[], effective_methods=[], ineffective_methods=[],
                 implementation_failed_methods=[], history=[], roofline_history=[], frontier=[])
    state_path = write(tmp_path / "state.json", state)
    b = good_bench(candidate, state["ref_file"]); b["confirmation"] = report
    bench = write(tmp_path / "bench.json", b)
    methods = write(tmp_path / "methods.json", {"methods": [{"id": "memory.lds_tiling", "axis": "memory"}]})
    args = argparse.Namespace(state=state_path, iter=1, kernel=str(candidate), bench=bench, methods_json=methods,
                              skip_validation=True, allow_ineffective=False, attribution=None, sass_check=None, retries=0)
    load("state").cmd_update(args)
    saved = json.loads(Path(state_path).read_text())
    assert saved["best_file"] == str(candidate) and saved["history"][0]["status"] == "improved"
    assert saved["best_source_sha256"] == load("experiment").file_sha256(candidate)
    assert len(saved["unverified_methods"]) == 1
    candidate.write_text("changed after closing")
    with pytest.raises(SystemExit, match="best_file changed"):
        load("experiment").check_frozen_inputs(saved)


def test_output_poison_detects_unwritten_zero_and_padding_damage():
    mod = load("benchmark")
    state = {"tensor_inputs": {"x": torch.ones(5), "out": torch.zeros(5)},
             "pristine_tensors": {"x": torch.ones(5), "out": torch.zeros(5)}, "output_specs": [("out", "float*")]}
    reference = {"x": torch.ones(5), "out": torch.zeros(5)}
    extents = mod._prepare_validation(state, reference, {"out": 3})
    reference["out"][:3] = 0.
    assert not mod._validate_contract(state, reference, extents, 0., 0.)
    state["tensor_inputs"]["out"][:3] = 0.
    assert mod._validate_contract(state, reference, extents, 0., 0.)
    state["tensor_inputs"]["out"][4] = 1.
    assert not mod._validate_contract(state, reference, extents, 0., 0.)
    state["tensor_inputs"]["out"][4] = 0.
    state["tensor_inputs"]["x"][0] = 2.
    assert not mod._validate_contract(state, reference, extents, 0., 0.)


def test_output_extent_and_input_fingerprint():
    mod = load("benchmark")
    state = {"tensor_inputs": {"out": torch.zeros(5, dtype=torch.int32)},
             "output_specs": [("out", "int*")], "reference_inputs": {"x": torch.arange(4), "N": 4}}
    ref = {"out": torch.zeros(5, dtype=torch.int32)}
    for extent in (-1, 6, True):
        with pytest.raises(ValueError):
            mod._prepare_validation(state, ref, {"out": extent})
    mod._prepare_validation(state, ref, {"out": 5})
    assert not torch.equal(state["tensor_inputs"]["out"], ref["out"])
    before = mod._input_fingerprint(state)
    state["reference_inputs"]["x"][0] = 10
    assert mod._input_fingerprint(state) != before


def test_isa_parser_excludes_comments_macros_and_other_symbols(tmp_path):
    source = '''.type first,@function
.type second,@function
.macro helper
 v_mmac_f32_16x16x16_bf16 v0, v1, v2, v3
.endm
first:
 // v_mmac_f32_16x16x16_bf16 ignored
 .LBB0:
 000000: D0000000 buffer_load_dword v0, v1, s[0:3]
 s_waitcnt_vbcnt 0 ; matrix_load_b16 is a comment
.size first, .-first
second:
 v_mmac_scale_f32_16x16x64_fp4 v0, v1, v2, v3
'''
    parser = load("isa")
    assert len(parser.parse(source)["instructions"]) == 3
    assert [r["mnemonic"] for r in parser.parse(source, "first")["instructions"]] == ["buffer_load_dword", "s_waitcnt_vbcnt"]
    assert parser.parse(source, "absent")["scope"] == "unresolved_symbol"
    path = tmp_path / "first.amdgcn"; path.write_text(source)
    result = load("scan_amdgcn", TRITON).scan(str(path), "first")
    assert result["totals"]["v_mmac"] == 0
    assert result["totals"]["s_waitcnt_vbcnt"] == 1
    assert load("scan_amdgcn", TRITON).scan(str(path), "absent")["unresolved"]


def test_isa_scan_never_treats_builtin_or_notes_as_machine_instructions():
    text = '// buffer_load_dword v0, v1 lds\n__builtin_amdgcn_raw_buffer_load_lds(p);\nmetadata: v_mmac_f32\n'
    sig = json.loads((HIP.parent / "references/dcu_isa_signatures.json").read_text(encoding="utf-8"))
    assert not load("sass_check").check_method("memory.global_to_lds_async", text, sig)["pattern_presence"]
    assert not load("isa").parse(text)["instructions"]


def test_hex_gfx_identifier_and_coupled_independent_deltas(tmp_path):
    mod = load("validate_methods")
    assert mod._parse_gfx_arch("gfx92a:sramecc+") == "gfx92a"
    assert mod._parse_gfx_arch("sm_90") == ""
    registry = {"methods": {"a": {"axis": "memory", "priority": 1}, "b": {"axis": "memory", "priority": 2}},
                "coupled_methods": [{"ids": ["a", "b"]}]}
    state = write(tmp_path / "state.json", {})
    reg = write(tmp_path / "reg.json", registry)
    data = {"methods": [{"id": "a", "axis": "memory", "priority": 1}, {"id": "b", "axis": "memory", "priority": 2}]}
    path = write(tmp_path / "methods.json", data)
    assert not mod.validate(path, state, reg)[0]
    data["coupling_reviews"] = [{"ids": ["a", "b"], "separate_deltas": "separate loader and layout edits",
                                 "validation_plan": "semantic checks followed by isolated ablations"}]
    write(Path(path), data)
    assert mod.validate(path, state, reg)[0]


def test_compile_records_exact_binary_not_neighbor(tmp_path):
    mod = load("benchmark")
    source, output = tmp_path / "kernel.hip", tmp_path / "measured.so"
    source.write_text("source")
    def compiler(cmd, **kwargs):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"measured binary")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    with patch.object(mod.subprocess, "run", side_effect=compiler):
        build = mod.compile_hip(str(source), str(output), "gfx938", "hipcc", "hip")
    assert build["binary_sha256"] == load("experiment").file_sha256(output)
    assert build["source_sha256"] == load("experiment").file_sha256(source)
    assert "--offload-arch=gfx938" in build["command"]


def test_paired_matrix_cannot_hide_case_regression(tmp_path):
    mod, _, state, candidate, control, benchmark, _ = comparison_fixture(tmp_path)
    state['workloads'] = [{'id':'fast','dims':{'N':10},'seeds':[42],'weight':9},
                          {'id':'regression','dims':{'N':10},'seeds':[42],'weight':1}]
    def measure(cmd, output, log):
        role = Path(output).stem; case = Path(output).parent.parent.name
        source = control if role == 'control' else candidate
        ms = (2. if case == 'fast' else .9) if role == 'control' else 1.
        b=good_bench(source,state['ref_file']); b.update(kernel=load('experiment').timing_stats([ms]*5),
            inputs_sha256='fixed',gpu_index=0,gpu_name='CPU MOCK',arch='gfx938',signature=[{'type':'f32'}],
            dims={'N':10},seed=42,warmup=10,repeat=20,ptr_size_override=10)
        write(Path(output),b); return b
    with patch.object(mod,'run_json',side_effect=measure):
        report=mod.compare(state,str(candidate),str(control),str(benchmark),tmp_path/'matrix')
    a=report['assessment']
    assert a['valid'] and all(r>1.5 for r in a['round_speedups'])
    assert not a['case_regression_guards_passed'] and not a['improved']


def test_valid_paired_ablation_and_mechanism_are_both_required(tmp_path):
    _, report, state, candidate, control, _, _ = comparison_fixture(tmp_path)
    state.update(best_metric_ms=2.,selected_methods=[],effective_methods=[],ineffective_methods=[],
                 implementation_failed_methods=[],history=[],roofline_history=[],frontier=[])
    state_path=write(tmp_path/'state.json',state)
    b=good_bench(candidate,state['ref_file']); b['confirmation']=report
    bench=write(tmp_path/'bench.json',b)
    mid='memory.lds_tiling'
    methods=write(tmp_path/'methods.json',{'methods':[{'id':mid,'axis':'memory'}]})
    attribution=write(tmp_path/'attr.json',{'champion_source_sha256':b['source_sha256'],
        'attributions':[{'method_id':mid,'ablated_kernel':str(control),'comparison':report}]})
    evidence=tmp_path/'iterv1/isa.txt';evidence.parent.mkdir();evidence.write_text('synthetic mechanism fixture')
    write(evidence.with_name('mechanism-review.json'),{'source_sha256':b['source_sha256'],
        'methods':[{'id':mid,'status':'verified','explanation':'Fixture for provenance and attribution gates',
                    'artifact':'isa.txt','artifact_sha256':load('experiment').file_sha256(evidence)}]})
    args=argparse.Namespace(state=state_path,iter=1,kernel=str(candidate),bench=bench,methods_json=methods,
                           skip_validation=True,allow_ineffective=False,attribution=attribution,sass_check=None,retries=0)
    load('state').cmd_update(args)
    assert len(json.loads(Path(state_path).read_text())['effective_methods'])==1


def test_boolean_output_sentinels_remain_distinct():
    mod=load('benchmark')
    state={'tensor_inputs':{'out':torch.zeros(4,dtype=torch.bool)},'output_specs':[('out','bool')]}
    ref={'out':torch.zeros(4,dtype=torch.bool)}
    mod._prepare_validation(state,ref,None)
    assert not torch.equal(state['tensor_inputs']['out'],ref['out'])


def test_duplicate_case_seeds_rejected(tmp_path):
    path=write(tmp_path/'cases.json',{'cases':[{'id':'same','dims':{},'seeds':[42,42]}]})
    with pytest.raises(ValueError,match='unique'):
        load('workload_suite').load_cases(path)


def test_generated_matmul_adapter_declares_logical_extent():
    analysis={'chosen_reference':{'name':'reference','args':['A','B']},
              'signature_hint':{'tensor_args':['A','B'],'output_args':[],'return_style':'return'},
              'dims':{'M':3,'N':5,'K':7}}
    text=load('generate_baseline',BASE)._adapter_py(analysis,'C','matmul','original.py')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='output_extents')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<fixture>','exec'),namespace)
    assert namespace['output_extents'](M=3,N=5,K=7)=={'C':15}


def test_sass_check_rejects_neighbor_binary_without_build_record(tmp_path):
    it=tmp_path/'iterv1';it.mkdir()
    source,ref=it/'kernel.hip',tmp_path/'ref.py'
    source.write_text('source');ref.write_text('oracle');source.with_suffix('.so').write_bytes(b'unmeasured')
    write(it/'methods.json',{'methods':[]});write(it/'bench.json',good_bench(source,ref))
    state=write(tmp_path/'state.json',{'run_dir':str(tmp_path)})
    mod=load('sass_check')
    with patch.object(mod,'_dump_isa') as dump:
        result=mod.run(state,1)
    dump.assert_not_called()
    assert result['error']=='measured_binary_missing_or_changed'
