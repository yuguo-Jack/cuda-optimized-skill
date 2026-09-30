import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_lints():
    lint = load("lint_registry")
    assert lint.lint(str(ROOT / "references/method_registry.json"),
                     str(ROOT / "references/metric_registry.json"),
                     str(ROOT / "references/optimization_catalog.md")) == []


def test_missing_metrics_are_unknown():
    roofline = load("roofline")
    result = roofline.compute_deltas({"compute": [], "memory": [], "latency": []}, {"gpus": [{"sm_arch": "sm_90"}]})
    assert result["delta_compute"] is None
    assert result["delta_memory"] is None
    assert result["delta_latency"] is None
    assert roofline.allocate_budget(None, None, None) == {"compute": 0, "memory": 0, "latency": 0}


def test_sass_target_symbol_scope():
    sass = load("sass_check")
    signatures = {"methods": {"compute.tensor_core": {"sass_patterns": ["HMMA"], "require_any": True}}}
    text = "Function : unrelated\n HMMA\nFunction : target\n FFMA\n"
    result = sass.check_method_sass("compute.tensor_core", text, signatures, "target")
    assert result["status"] == "fail"
    assert result["verified"] is False


def test_new_strategy_metadata():
    registry = json.loads((ROOT / "references/method_registry.json").read_text(encoding="utf-8"))
    for method in ["compute.structured_sparsity_2to4", "memory.kv_cache_layout", "latency.grouped_gemm_scheduler"]:
        meta = registry["methods"][method]
        assert meta["workload_archetypes"]
        assert meta["verification"]
        assert meta["evidence_requirements"]["missing_policy"] in {"skip", "inconclusive"}


def test_workload_matrix_scales_and_respects_hard_limit():
    wm = load("workload_matrix")
    cases = wm.generate_workload_matrix(
        {"B": 8, "H": 32, "S": 4096, "D": 128}, 8 * 32 * 4096 * 128,
        archetype="attention", target_mb=2, hard_mb=4, alignment=8)
    assert [c["scale"] for c in cases] == ["small", "medium", "large"]
    assert all(c["realized_working_set_bytes"] <= 4 * 1024 * 1024 for c in cases)
    assert all(c["realized_dims"]["S"] % 8 == 0 for c in cases)
    assert cases[-1]["downscaled"] is True


def test_workload_shrink_retry():
    wm = load("workload_matrix")
    entry = {"realized_dims": {"M": 100, "N": 100}, "scale_axes": ["M", "N"],
             "alignment": 8, "realized_ptr_size": 10000,
             "realized_working_set_bytes": 40000}
    shrunk = wm.shrink_workload(entry)
    assert shrunk["downscale_reason"] == "runtime_oom_retry"
    assert shrunk["realized_dims"]["M"] % 8 == 0
    assert shrunk["realized_working_set_bytes"] < entry["realized_working_set_bytes"]


def test_contract_check_static_signature_and_gate_separation(tmp_path):
    contract = load("contract_check")
    kernel = tmp_path / "kernel.cu"
    kernel.write_text(
        '__global__ void copy_kernel(const float* q, float* out, int M) {\n'
        '  int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
        '  if (i >= M) return;\n'
        '  out[i] = q[i];\n'
        '}\n'
        'extern "C" void solve(const float* q, float* out, int M) { copy_kernel<<<1, 1>>>(q, out, M); }\n', encoding="utf-8")
    bench = {"correctness": {"passed": True}, "kernel": {"average_ms": 1.25}}
    result = contract.check_contract(str(kernel), {"M": 16}, bench)
    assert result["compile_pass"] == "inconclusive"
    assert result["contract_pass"] == "pass"
    assert result["correctness_pass"] == "pass"
    assert result["timing_valid"] == "pass"
    assert result["race_safe"] == "inconclusive"
    assert result["index_layout"]["bounds_guard"] is True


def test_contract_check_missing_dims_and_bad_signature(tmp_path):
    contract = load("contract_check")
    kernel = tmp_path / "kernel.cu"
    kernel.write_text('extern "C" void solve(float* out, int M) { out[0] = 0; }\n', encoding="utf-8")
    result = contract.check_contract(str(kernel), {})
    assert result["contract_pass"] == "fail"
    assert result["dims"]["missing"] == ["M"]
    assert result["correctness_pass"] == "inconclusive"

    bad = tmp_path / "bad.cu"
    bad.write_text('__global__ void k() {}\n', encoding="utf-8")
    bad_result = contract.check_contract(str(bad), {})
    assert bad_result["contract_pass"] == "fail"
    assert bad_result["errors"]


def test_timing_stats_include_robust_cv_and_bootstrap():
    benchmark = load("benchmark")
    result = benchmark._stats_dict([1.0, 1.1, 0.9, 1.0, 1.05])
    assert result["median_ms"] == 1.0
    assert result["robust_cv"] is not None
    assert result["confidence_interval_95"]["low_ms"] <= result["median_ms"] <= result["confidence_interval_95"]["high_ms"]


def test_adaptive_ncu_policy_thresholds():
    profile = load("profile_ncu")
    assert profile._choose_profile_set(9.99)[0] == "full"
    assert profile._choose_profile_set(10.0)[0] == "light"
    assert profile._choose_profile_set(50.0)[0] == "light"
    assert profile._choose_profile_set(50.01)[0] == "light"
