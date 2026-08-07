"""Performance gate for compiler-managed Vector masks.

The ``counter_chain`` case reproduces the focused workload from issue #1411:
each of two AIVs adds two 64-KiB fp32 UB buffers through one ``T.tile.add``
call site in a 256-iteration runtime loop.  The chunks are disjoint, and the
hot loop contains no synchronization.  Identical boundary barriers isolate
the surrounding GM-to-UB and UB-to-GM copies.

The script measures device kernel duration with the lightweight Level0
``torch_npu.profiler`` configuration.  TileLang auto-sync and BiSheng CCE
auto-sync are both disabled.  Run the same file from the old and new checkout
roots, and keep the kernel cache disabled, so that only the imported compiler
changes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
import torch_npu

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}
ITERATIONS = 128
COUNTER_NUM_AIVS = 2
COUNTER_BUFFER_BYTES = 64 * 1024
COUNTER_DTYPE = "float32"
COUNTER_DTYPE_BYTES = 4
COUNTER_ELEMENTS = COUNTER_BUFFER_BYTES // COUNTER_DTYPE_BYTES
COUNTER_CHUNK_ELEMENTS = 64
COUNTER_CHUNKS = COUNTER_ELEMENTS // COUNTER_CHUNK_ELEMENTS
MODE_SWITCH_NUM_AIVS = 48
MODE_SWITCH_CONSUMERS = 4
KernelCall = Callable[[], torch.Tensor]


def profile_kernel_us(
    call: KernelCall,
    kernel_name: str,
    warmups: int,
    launches: int,
) -> list[float]:
    """Collect one device-duration sample per launch with the Level0 profiler."""
    for _ in range(warmups):
        call()
    torch.npu.synchronize()

    with tempfile.TemporaryDirectory() as profile_dir:
        profiler_level = torch_npu.profiler.ProfilerLevel.Level0
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=profiler_level,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_dir),
            experimental_config=experimental_config,
        ):
            for _ in range(launches):
                call()
            torch.npu.synchronize()

        detail_paths = glob.glob(f"{profile_dir}/**/kernel_details.csv", recursive=True)
        if len(detail_paths) != 1:
            raise RuntimeError(f"expected one kernel_details.csv, found {len(detail_paths)}")

        durations = []
        with open(detail_paths[0], newline="", encoding="utf-8") as detail_file:
            for row in csv.DictReader(detail_file):
                if row.get("Name") == kernel_name:
                    durations.append(float(row["Duration(us)"]))

    if len(durations) != launches:
        raise RuntimeError(f"expected {launches} {kernel_name} samples, found {len(durations)}")
    return durations


@T.prim_func
def counter_chain(
    acc: T.Tensor((COUNTER_NUM_AIVS, COUNTER_ELEMENTS), COUNTER_DTYPE),
    addend: T.Tensor((COUNTER_NUM_AIVS, COUNTER_ELEMENTS), COUNTER_DTYPE),
    output: T.Tensor((COUNTER_NUM_AIVS, COUNTER_ELEMENTS), COUNTER_DTYPE),
):
    """Add 256 disjoint full-repeat chunks through one runtime call site."""
    with T.Kernel(1, is_npu=True) as (cid, vid):  # noqa: F841
        acc_ub = T.alloc_ub((COUNTER_ELEMENTS,), COUNTER_DTYPE)
        addend_ub = T.alloc_ub((COUNTER_ELEMENTS,), COUNTER_DTYPE)

        with T.Scope("V"):
            T.copy(acc[vid, :], acc_ub)
            T.copy(addend[vid, :], addend_ub)
            T.barrier_all()

            for chunk in T.serial(COUNTER_CHUNKS):
                offset = chunk * COUNTER_CHUNK_ELEMENTS
                T.tile.add(
                    acc_ub[offset : offset + COUNTER_CHUNK_ELEMENTS],
                    acc_ub[offset : offset + COUNTER_CHUNK_ELEMENTS],
                    addend_ub[offset : offset + COUNTER_CHUNK_ELEMENTS],
                )

            T.barrier_all()
            T.copy(acc_ub, output[vid, :])


@T.prim_func
def mode_switch(
    a: T.Tensor((MODE_SWITCH_NUM_AIVS, 128), "float16"),
    b: T.Tensor((MODE_SWITCH_NUM_AIVS, 128), "float16"),
    c: T.Tensor((MODE_SWITCH_NUM_AIVS, 128), "float16"),
):
    """Alternate raw NORMAL reductions and raw COUNTER arithmetic without sync."""
    with T.Kernel(MODE_SWITCH_NUM_AIVS, threads=1, is_npu=True) as cid:  # noqa: F841
        a_ub = T.alloc_ub((128,), "float16")
        b_ub = T.alloc_ub((128,), "float16")
        c_ub = T.alloc_ub((128,), "float16")
        reduced_ub = T.alloc_ub((8,), "float16")
        T.tile.fill(a_ub, 1.0)
        T.tile.fill(b_ub, 0.0)
        T.tile.fill(c_ub, 0.0)
        T.tile.fill(reduced_ub, 0.0)
        for _ in T.serial(ITERATIONS):
            T.tile.block_reduce_max(reduced_ub, a_ub, 1, 128, 1, 1, 8)
            T.tile.add(c_ub, a_ub, b_ub)
            T.tile.block_reduce_max(reduced_ub, c_ub, 1, 128, 1, 1, 8)
            T.tile.add(a_ub, c_ub, b_ub)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["counter_chain", "mode_switch"], required=True)
    parser.add_argument("--launches", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--check-runs", type=int, default=20)
    parser.add_argument("--expected-tilelang-root", type=Path)
    args = parser.parse_args()

    # Every benchmark process must lower its current checkout afresh.
    tilelang.disable_cache()
    tilelang_root = Path(tilelang.__file__).resolve().parent.parent
    if args.expected_tilelang_root is not None:
        expected_root = args.expected_tilelang_root.resolve()
        if tilelang_root != expected_root:
            raise RuntimeError(f"expected TileLang root {expected_root}, imported {tilelang_root}")

    program = counter_chain if args.case == "counter_chain" else mode_switch
    consumers = 1 if args.case == "counter_chain" else MODE_SWITCH_CONSUMERS
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs=PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off", "-O3"],
        target="ascendc",
        platform="A3",
    )
    source = kernel.get_kernel_source()
    count_form_add_count = source.count("AscendC::Add(")
    raw_add_count = source.count("AscendC::Add<")
    if args.case == "counter_chain" and count_form_add_count + raw_add_count != 1:
        raise RuntimeError("counter_chain source does not contain exactly one Add call site")

    if args.case == "counter_chain":
        generator = torch.Generator().manual_seed(0)
        counter_shape = (COUNTER_NUM_AIVS, COUNTER_ELEMENTS)
        a_cpu = torch.randn(counter_shape, dtype=torch.float32, generator=generator)
        b_cpu = torch.randn(counter_shape, dtype=torch.float32, generator=generator)
        expected = a_cpu + b_cpu
        a = a_cpu.npu()
        b = b_cpu.npu()
    else:
        expected = None
        a = torch.empty((MODE_SWITCH_NUM_AIVS, 128), dtype=torch.float16, device="npu")
        b = torch.empty((MODE_SWITCH_NUM_AIVS, 128), dtype=torch.float16, device="npu")

    check_failures = 0
    if args.case == "counter_chain":
        for _ in range(args.check_runs):
            actual = kernel(a, b).cpu()
            if not torch.equal(actual, expected):
                check_failures += 1
        if check_failures:
            raise RuntimeError(f"counter_chain failed {check_failures}/{args.check_runs} checks")

    kernel_name = f"{args.case}_kernel"
    durations = profile_kernel_us(
        lambda: kernel(a, b),
        kernel_name,
        args.warmups,
        args.launches,
    )
    set_mode_count = source.count("AscendC::SetMaskCount();")
    set_mode_count += source.count("AscendC::SetMaskNorm();")

    result = {
        "case": args.case,
        "tilelang_root": str(tilelang_root),
        "launches": args.launches,
        "warmups": args.warmups,
        "iterations": COUNTER_CHUNKS if args.case == "counter_chain" else ITERATIONS,
        "aiv_count": COUNTER_NUM_AIVS if args.case == "counter_chain" else MODE_SWITCH_NUM_AIVS,
        "consumers_per_iteration": consumers,
        "check_runs": args.check_runs if args.case == "counter_chain" else 0,
        "check_failures": check_failures,
        "count_form_add_count_in_source": count_form_add_count,
        "raw_add_count_in_source": raw_add_count,
        "set_mode_count_in_source": set_mode_count,
        "set_payload_count_in_source": source.count("AscendC::SetVectorMask"),
        "duration_us_min": min(durations),
        "duration_us_median": statistics.median(durations),
        "duration_us_mean": statistics.fmean(durations),
        "duration_us_max": max(durations),
    }
    if args.case == "counter_chain":
        result.update(
            {
                "buffer_bytes_per_aiv": COUNTER_BUFFER_BYTES,
                "chunk_elements": COUNTER_CHUNK_ELEMENTS,
                "chunks_per_launch": COUNTER_CHUNKS,
            }
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
