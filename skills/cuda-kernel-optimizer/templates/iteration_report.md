# Iteration {{iter}} — Analysis

**Kernel profiled (input)**: `{{best_file_before}}`
**Time before**: {{best_ms_before}} ms
**GPU / arch**: {{gpu_name}} / {{sm_arch}}

---

## Roofline Analysis (from `roofline.json`)

| Axis | Δ (gap) | Utilization | Budget |
|------|---------|-------------|--------|
| Compute | {{delta_compute}} | {{compute_util_pct}}% | {{budget_compute}} |
| Memory | {{delta_memory}} | {{memory_util_pct}}% | {{budget_memory}} |
| Latency | {{delta_latency}} | max stall {{max_stall_pct}}% | {{budget_latency}} |

**Primary bound**: {{bound}}

---

## Top ncu metrics (from `ncu_top.json`)

### Compute
{{compute_metrics_table}}

### Memory
{{memory_metrics_table}}

### Latency / stalls
{{latency_metrics_table}}

---

## Diagnosis

_Which axis is the dominant bottleneck right now? Cite specific metric values. Explain how the roofline budget allocation reflects this diagnosis._

> Example:
> `sm__pipe_tensor_op_hmma_cycles_active...pct_of_peak = 8.4%` on a GEMM signals tensor cores barely used — compute Δ = 0.92. Simultaneously `smsp__warp_issue_stalled_long_scoreboard...pct = 61%` → latency Δ = 0.61. Memory bandwidth at 43% → Δ_m = 0.57. Budget: compute=2, memory=0, latency=1. Attack compute and latency; memory already near limit.

## Chosen methods

For each, state: (a) **priority level** from catalog, (b) metric evidence, (c) the method, (d) the implementation delta vs. the current best, (e) expected ncu metric shift.

### {{axis_1}} — `{{method_1_id}}` (Priority: {{P_level_1}})
- **Budget for this axis**: {{budget_axis_1}}
- **Priority scan**: _List all higher-priority methods that were scanned and why each was skipped_
- **Trigger evidence**: _(cite specific ncu metric name = value)_
- **Trigger strength**: _(continuous value 0-1 if b_axis ≥ 2)_
- **Method**:
- **Delta vs current best**:
- **Expected ncu shift**:
- **Risks / coupling**:

### {{axis_2}} — `{{method_2_id}}` (Priority: {{P_level_2}})
- **Budget for this axis**: {{budget_axis_2}}
- **Priority scan**: _List higher-priority methods scanned + skip reasons_
- **Trigger evidence**:
- **Method**:
- **Delta vs current best**:
- **Expected ncu shift**:
- **Risks / coupling**:

### {{axis_3}} — `{{method_3_id}}` (Priority: {{P_level_3}})
_(only if B=3 methods; omit if axis budget is 0)_
- **Budget for this axis**: {{budget_axis_3}}
- **Priority scan**:
- **Trigger evidence**:
- **Method**:
- **Delta vs current best**:
- **Expected ncu shift**:
- **Risks / coupling**:

## Orthogonality check

_Verify: (1) no pair is the same optimization under two names, (2) coupled pairs (memory.P5 + latency.P3) not both selected, (3) all arch-compatible, (4) axis distribution matches roofline budget._

## Excluded candidates (higher-priority methods that were skipped)

_List every higher-priority method on each axis that was NOT selected, with the exact reason:_

- `compute.tensor_core` (P1) — skipped: kernel is pure elementwise, no matmul semantics
- `memory.kernel_fusion` (P1) — skipped: already in `selected_methods` from iter 1
- `latency.warp_shuffle_sync` (P1) — skipped: Triton compiler already handles via `tl.reduce`

## Branch variants

_Describe the K hyperparameter variants generated for branch-and-select:_

| Branch | Tile (M×N×K) | Stages | Warps | Other diff |
|--------|-------------|--------|-------|------------|
| b1 | 128×128×32 | 3 | 4 | — |
| b2 | 128×256×32 | 3 | 8 | — |
| b3 | 256×128×32 | 4 | 4 | — |
| b4 | 128×128×64 | 5 | 4 | — |

---

## Result (filled after benchmarking)

- **Champion branch**: b{{champion_idx}}
- **New ms**: {{new_ms}}
- **Speedup vs previous best**: {{speedup_vs_best_before}}
- **Speedup vs reference**: {{speedup_vs_ref}}
- **Validation**: {{validation_status}}
- **Retries needed**: {{retries}}
- **Outcome**: {{outcome}} (improved / regressed / failed_validation)

### Attribution (from ablation)

| Method | Ablated ms | Attribution ms | Contributed? | SASS verified? |
|--------|-----------|---------------|-------------|----------------|
| {{m1_id}} | {{m1_ablated_ms}} | {{m1_attr_ms}} | {{m1_contributed}} | {{m1_sass}} |
| {{m2_id}} | {{m2_ablated_ms}} | {{m2_attr_ms}} | {{m2_contributed}} | {{m2_sass}} |
| {{m3_id}} | {{m3_ablated_ms}} | {{m3_attr_ms}} | {{m3_contributed}} | {{m3_sass}} |

### Post-hoc vs expected

_Did the metric shift match the prediction? If not, why?_

_Did attribution confirm the methods we expected to be effective?_

_This note becomes part of the final retrospective._
