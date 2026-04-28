# Ref-To-Baseline Patterns

## Contract Target

The downstream Hygon optimizer expects:

```cpp
extern "C" void solve(const float* x, float* out, int N);
```

or a similar signature. `const` pointer parameters are inputs. Non-const pointer parameters are outputs. Scalar dimensions must be provided through `--dims` and command-line `--<name>=<value>`.

The generated `ref.py` must expose:

```python
def reference(...):
    ...
```

and must write output tensors in-place.

## Torch References

Prefer a function named `reference`, `torch_ref`, `ref`, `golden`, `forward`, or `model_forward`.

Return-style Torch ref:

```python
def reference(x, z, N):
    return x + 2.0 * z
```

Adapter pattern:

```python
def reference(x, z, out, N):
    out.copy_(_orig.reference(x, z, N))
```

In-place Torch ref:

```python
def reference(x, z, out, N):
    out.copy_(x + 2.0 * z)
```

Adapter can delegate directly. Preserve output names.

Matmul return-style ref:

```python
def reference(a, b, M, N, K):
    return torch.matmul(a.view(M, K), b.view(K, N))
```

Adapter should copy to `out.view(M, N)`.

## Triton References

Treat `@triton.jit` functions as implementation clues. They usually cannot be used directly by the generic benchmark unless the file also provides a plain Python wrapper.

Search for:

- a Torch oracle in the same file;
- a `call(...)`, `run(...)`, or `forward(...)` wrapper that launches Triton;
- block sizes and masks that reveal layout and boundary behavior.

When only a Triton kernel exists, write a Torch oracle from the Triton math first, then generate the HIP baseline against that oracle.

Common mapping:

| Triton idiom | Baseline meaning |
| --- | --- |
| `tl.arange(0, BLOCK)` | one-dimensional flat tile |
| `tl.load(ptr + offsets, mask=...)` | guarded global load |
| `tl.store(out + offsets, value, mask=...)` | guarded output write |
| `tl.dot(a, b)` | matmul-like contraction |
| `tl.sum`, `tl.max` | reduction axis |

## TileLang References

Treat TileLang kernels as implementation clues unless they expose a simple callable wrapper. Extract:

- tensor shapes from `T.Buffer` annotations or launch parameters;
- loop nests and accumulation axes;
- block/thread binding;
- output write indices.

For GEMM-like TileLang code, generate a naive HIP matmul first:

```cpp
C[m * N + n] = sum_k A[m * K + k] * B[k * N + n];
```

For reductions, generate correctness-first serial or block-local reductions and benchmark only after correctness is stable.

## Baseline Templates

### Elementwise

Use one thread per output element:

```cpp
__global__ void kernel(const float* x, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = x[i];
}
```

Then replace the right-hand side with the reference expression.

### Matmul

Use one thread per C element:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

Do not optimize this template before the iterative optimizer starts.

## Correctness Debug Checklist

- Run the smallest shape that exercises all boundary conditions.
- Make sure every input tensor is `const` in `solve(...)`.
- Make sure every output tensor is non-const and is zeroed by the benchmark before launch.
- Align wrapper views with flat allocation size.
- For broadcasting, explicitly expand in `ref.py` and write the equivalent indexing in HIP.
- For tolerances, set `atol` and `rtol` in generated `ref.py`.
- For dtype-sensitive refs, record unsupported dtype assumptions in `baseline_manifest.json`.
- If the generated baseline is a placeholder, say so in the manifest and final answer.
