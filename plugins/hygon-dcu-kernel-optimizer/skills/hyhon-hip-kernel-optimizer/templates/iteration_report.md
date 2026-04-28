# Iteration {{iter}} - Analysis

**Kernel profiled (input)**: `{{best_file_before}}`
**Time before**: {{best_ms_before}} ms
**GPU / arch**: {{gpu_name}} / {{gfx_arch}}

---

## Roofline Analysis (from `roofline.json`)

| Axis | delta (gap) | Utilization | Budget |
|------|-------------|-------------|--------|
| Compute | {{delta_compute}} | {{compute_util_pct}}% | {{budget_compute}} |
| Memory | {{delta_memory}} | {{memory_util_pct}}% | {{budget_memory}} |
| Latency | {{delta_latency}} | max stall {{max_stall_pct}}% | {{budget_latency}} |

**Primary bound**: {{bound}}

---

## Top DCU Metrics (from `dcu_top.json`)

### Compute
{{compute_metrics_table}}

### Memory
{{memory_metrics_table}}

### Latency / Stalls
{{latency_metrics_table}}

---

## Diagnosis

_Which axis is the dominant bottleneck right now? Cite specific hipprof/PMC/SQTT values. Explain how the roofline budget allocation reflects this diagnosis._

> Example:
> `SQ_INSTS_MMOP = 0` on a GEMM-like kernel means MMAC is missing. High `TCC_EA_RDREQ_sum` points at memory traffic. Barrier or wait-heavy SQTT stats push latency budget.

## Chosen Methods

For each method, state: (a) priority level from the catalog, (b) DCU metric or dccobjdump evidence, (c) the exact implementation delta vs. current best, and (d) expected metric shift.

### {{axis_1}} - `{{method_1_id}}` (Priority: {{P_level_1}})
- **Budget for this axis**: {{budget_axis_1}}
- **Priority scan**: _List all higher-priority methods that were scanned and why each was skipped._
- **Trigger evidence**: _(cite specific DCU metric / ISA evidence)_
- **Trigger strength**: _(continuous value 0-1 if axis budget >= 2)_
- **Method**:
- **Delta vs current best**:
- **Expected metric shift**:
- **Risks / coupling**:

### {{axis_2}} - `{{method_2_id}}` (Priority: {{P_level_2}})
- **Budget for this axis**: {{budget_axis_2}}
- **Priority scan**: _List higher-priority methods scanned and skip reasons._
- **Trigger evidence**:
- **Method**:
- **Delta vs current best**:
- **Expected metric shift**:
- **Risks / coupling**:

### {{axis_3}} - `{{method_3_id}}` (Priority: {{P_level_3}})
_(Only if a third method is selected; omit if the relevant axis budget is 0.)_
- **Budget for this axis**: {{budget_axis_3}}
- **Priority scan**:
- **Trigger evidence**:
- **Method**:
- **Delta vs current best**:
- **Expected metric shift**:
- **Risks / coupling**:

## Orthogonality Check

_Verify: (1) no pair is the same optimization under two names, (2) coupled pairs are counted once when they are the same code delta, (3) all methods are arch-compatible, and (4) axis distribution matches roofline budget._

## Excluded Candidates

_List every higher-priority method on each axis that was not selected, with the exact reason._

- `compute.mmac_tensor_core` (P1) - skipped: kernel is pure elementwise, no matmul semantics
- `memory.kernel_fusion` (P1) - skipped: already in `selected_methods` from iter 1
- `latency.wavefront_shuffle_ds_bpermute` (P3) - skipped: exchange crosses waves, not wave-local

## Branch Variants

_Describe the K hyperparameter variants generated for branch-and-select._

| Branch | Tile (MxNxK) | Stages | Wave/block shape | Other diff |
|--------|--------------|--------|------------------|------------|
| b1 | 128x128x32 | 3 | 256 threads | n/a |
| b2 | 128x256x32 | 3 | 256 threads | n/a |
| b3 | 256x128x32 | 4 | 512 threads | n/a |
| b4 | 128x128x64 | 5 | 256 threads | n/a |

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

| Method | Ablated ms | Attribution ms | Contributed? | DCU ISA verified? |
|--------|------------|----------------|--------------|-------------------|
| {{m1_id}} | {{m1_ablated_ms}} | {{m1_attr_ms}} | {{m1_contributed}} | {{m1_isa}} |
| {{m2_id}} | {{m2_ablated_ms}} | {{m2_attr_ms}} | {{m2_contributed}} | {{m2_isa}} |
| {{m3_id}} | {{m3_ablated_ms}} | {{m3_attr_ms}} | {{m3_contributed}} | {{m3_isa}} |

### Post-Hoc vs Expected

_Did the metric shift match the prediction? If not, why?_

_Did attribution confirm the methods expected to be effective?_

_This note becomes part of the final retrospective._
