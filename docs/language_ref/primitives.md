Tile Language: Primitives
=========================

## Ascend compilation options and C/V scopes

Ascend kernels contain work for Cube and Vector execution units. By default, the
compiler identifies which unit each operation needs and places it in the matching
scope. This also applies to kernels that use only Cube or only Vector operations.

The following `tilelang.PassConfigKey` options can be passed through `pass_configs`
to `@tilelang.jit` or `tilelang.compile`:

| Option | Default | Effect |
| --- | --- | --- |
| `TL_ASCEND_AUTO_CV_COMBINE` | `True` | Identify and separate Cube and Vector work |
| `TL_ASCEND_AUTO_SYNC` | `False` | Insert synchronization within each execution unit |
| `TL_ASCEND_MEMORY_PLANNING` | `False` | Reuse on-chip memory according to buffer lifetimes |
| `TL_ASCEND_AUTO_CV_SYNC` | `False` | Insert synchronization for recognized Cube/Vector GM workspace handoffs; requires CombineCV |

Omit `TL_ASCEND_AUTO_CV_COMBINE` to use automatic scope placement. You may also write
`with T.Scope("C")` or `with T.Scope("V")`: valid explicit scopes are preserved, so
handwritten scopes do not require disabling this option. The operations inside a
scope must belong to that unit; nesting C inside V, or V inside C, is rejected.

Set `TL_ASCEND_AUTO_CV_COMBINE=False` when you want to supply all scopes yourself.
Every operation tied to Cube or Vector must then be inside its matching scope.
Operations whose unit cannot be inferred, including opaque external calls and raw
source code, require an explicit scope even with automatic placement enabled.

Automatic scope placement does not enable the other three options. In particular,
placing a producer and consumer on the correct units does not synchronize them;
their data dependencies still need automatic or explicit synchronization.
