# Iteration {{iter}} — Analysis

**Kernel profiled (input)**: `{{best_file_before}}`
**Time before**: {{best_ms_before}} ms
**GPU / arch**: {{gpu_name}} / {{sm_arch}}

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

_Which axis is the dominant bottleneck right now? Cite specific metric values._

> Example:
> `sm__pipe_tensor_op_hmma_cycles_active...pct_of_peak = 8.4%` on a GEMM signals we're not using tensor cores at all — compute is the primary miss. Simultaneously `smsp__warp_issue_stalled_long_scoreboard...pct = 61%` shows a large fraction of warps are waiting on global memory, suggesting staging + pipelining is also missing. Memory-bandwidth utilization sits at 43%, high but not saturated. → Attack compute and latency first; memory layout is secondary.

## Chosen methods

For each, state: (a) **priority level** from catalog, (b) metric evidence, (c) the method, (d) the implementation delta vs. the current best, (e) expected ncu metric shift.

### Compute — `{{method_compute_id}}` (Priority: {{P_level}})
- **Priority scan**: _List all higher-priority methods that were scanned and why each was skipped (already selected / arch incompatible / skip condition met / trigger not matched)_
- **Trigger evidence**: _(cite specific ncu metric name = value)_
- **Method**:
- **Delta vs current best**:
- **Expected ncu shift**:
- **Risks / coupling**:

### Memory — `{{method_memory_id}}` (Priority: {{P_level}})
- **Priority scan**: _List all higher-priority methods that were scanned and why each was skipped_
- **Trigger evidence**:
- **Method**:
- **Delta vs current best**:
- **Expected ncu shift**:
- **Risks / coupling**:

### Latency — `{{method_latency_id}}` (Priority: {{P_level}})
- **Priority scan**: _List all higher-priority methods that were scanned and why each was skipped_
- **Trigger evidence**:
- **Method**:
- **Delta vs current best**:
- **Expected ncu shift**:
- **Risks / coupling**:

## Orthogonality check

_Verify: (1) no pair is the same optimization under two names, (2) if memory.P5 and latency.P3 are both selected they must be replaced, (3) all three are arch-compatible._

## Excluded candidates (higher-priority methods that were skipped)

_List every higher-priority method on each axis that was NOT selected, with the exact reason:_

- `compute.tensor_core` (P1) — skipped: kernel is pure elementwise, no matmul semantics
- `memory.kernel_fusion` (P1) — skipped: already in `selected_methods` from iter 1
- `latency.warp_shuffle_sync` (P1) — skipped: Triton compiler already handles via `tl.reduce`

---

## Result (filled after benchmarking)

- **New ms**: {{new_ms}}
- **Speedup vs previous best**: {{speedup_vs_best_before}}
- **Speedup vs reference**: {{speedup_vs_ref}}
- **Validation**: {{validation_status}}
- **Retries needed**: {{retries}}
- **Outcome**: {{outcome}} (improved / regressed / failed_validation)

### Post-hoc vs expected

_Did the metric shift match the prediction? If not, why?_

_This note becomes part of the final retrospective._
