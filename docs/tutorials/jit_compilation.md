# JIT compilation on Ascend

TileLang offers two public entry points:

- `tilelang.compile(program, ...)` compiles an existing `T.prim_func`.
- `@tilelang.jit(...)` compiles the program returned by a Python kernel factory.

Both entry points accept `pass_configs` for TileLang compiler passes and
`compile_flags` for extra Bisheng command-line options. They solve different
problems: a Bisheng flag does not enable or disable a TileLang pass.

## Kernel-scoped Bisheng flags

`compile_flags` accepts `list[str]`, `str`, or `None`:

```python
@tilelang.jit(
    out_idx=[1],
    target="ascendc",
    compile_flags=["-O3"],
)
def build_kernel(M, N):
    ...
```

The same option is available without the decorator:

```python
kernel = tilelang.compile(program, target="ascendc", compile_flags=["-O3"])
```

Flags are resolved separately for each kernel and are included in the kernel
cache key. TileLang emits its derived defaults first, then appends the explicit
flags. Bisheng uses the last repeated option: when the derived AscendC flags
contain `-O2`, the explicit `-O3` above appears later and wins.

The legacy `TL_CCE_AUTO_SYNC`, `TL_CCE_OPT_LEVEL`, and `TL_PTO_DEBUG`
environment variables remain target-specific, read-only fallbacks on the
backends that support them; `TL_PTO_DEBUG` affects only `target="pto"`.
Compiling one kernel never rewrites the process environment or changes the
defaults of a later kernel. Prefer `compile_flags` when an option belongs to
one kernel.

## Synchronization and debugging

Disabling Bisheng automatic synchronization is safe only when the TileLang
program or compiler passes already express every required dependency. Keep the
TileLang synchronization pass in `pass_configs`; use `--cce-auto-sync=off` only
to control the later Bisheng stage.

Ascend device printing can be enabled for one kernel without changing the process
environment:

```python
@tilelang.jit(
    target="ascendc",
    compile_flags=["-D_DEBUG", "--cce-enable-print"],
)
def debug_kernel(...):
    ...
```

Each `compile_flags` entry is whitespace-split and appended after the derived
flags; an exact duplicate already present in the base command is omitted.
TileLang does not validate options against the selected target, so prefer one
option per list element and use only flags supported by that backend. See
[`examples/compile_flags/compile_flags_example.py`](../../examples/compile_flags/compile_flags_example.py)
for an executable kernel-scoping example.

## Ahead-of-time compilation

For a deployable shared library, use the framework's `LibraryGenerator`
workflow instead of maintaining a copied Bisheng command. The canonical
end-to-end example is
[`examples/gemm_aot`](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples/gemm_aot).
