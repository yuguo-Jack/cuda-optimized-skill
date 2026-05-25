# Investigation Report Template

## 1. Profile

| Question | Result |
| --- | --- |
| Kernel name and profile rank/time share |  |
| Graph IR or log snippet connecting graph to kernel |  |
| Current timing |  |
| Input shapes/dtypes/strides |  |
| Autotune configs and best config |  |
| Captured artifact path |  |

## 2. Hints and Instructions

| Question | Result |
| --- | --- |
| Pointer arguments |  |
| `tt.divisibility` status |  |
| `tt.pointer_range` status |  |
| Dominant AMDGCN load/store families |  |
| `buffer_load/store_dwordx4` observed |  |
| `global_*` or `flat_*` still dominant |  |

## 3. Kernel Tuning

| Question | Result |
| --- | --- |
| `tl.assume` added or already present |  |
| `tl.multiple_of` added or already present |  |
| Config or block-size variants tested |  |
| Timing and bandwidth delta |  |
| Instruction delta |  |
| Continue Triton tuning? |  |

## 4. Avoid Generated Kernel

| Question | Result |
| --- | --- |
| Model code that triggers the kernel |  |
| Eager fallback or compile boundary |  |
| End-to-end performance delta |  |

## 5. Model Rewrite

| Question | Result |
| --- | --- |
| Rewrite opportunity |  |
| Reason it should help |  |
| Correctness impact |  |
| Final performance |  |

