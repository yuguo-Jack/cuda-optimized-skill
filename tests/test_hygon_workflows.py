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
    with patch.object(mod, "_bench_kernel", return_value={"error": "bad"}):
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


def test_state_promotes_kernel_without_fabricating_attribution(tmp_path):
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
    assert saved["best_metric_ms"] == 1.
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
    assert len(json.loads(Path(state_path).read_text())["effective_methods"]) == 1


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
a,_=p.parse_known_args();ms=float(Path(a.source).read_text());digest=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
Path(a.json_out).write_text(json.dumps({'source_sha256':digest(a.source),'reference_sha256':digest(a.ref),
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
