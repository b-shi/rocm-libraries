#!/usr/bin/env python3
################################################################################
# GPU functional test for graTileAssignment with parameterized tile configs
#
# Uses the actual graTileAssignment function to generate the offset computation
# assembly, then wraps it with a minimal kernel prologue/epilogue for GPU
# execution and validation.
#
# Tests that sharedVgprGROffset[0] for both A and B contain the correct
# global-read byte offsets for each thread.
#
# Usage:
#   python3 test_graTileAssignment_gpu.py
#   # or via pytest:
#   pytest test_graTileAssignment_gpu.py -v -s
################################################################################

import ctypes
import os
import re
import sys
import struct
import subprocess
import tempfile

# Add tensilelite to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

try:
    from hip import hip, hiprtc  # type: ignore
    HAS_HIP = True
except ImportError:
    HAS_HIP = False

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace
from dataclasses import dataclass

from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from Tensile.Components.SubtileBasedKernel import TileInfo, graTileAssignment

# ---- Constants ----
GFX_TARGET = "gfx950"
WAVESIZE   = 64
NUM_WAVES  = 4
NUM_THREADS = WAVESIZE * NUM_WAVES  # 256
BPE        = 2      # fp16
LOAD_WIDTH = 16     # dwordx4


@dataclass
class TileConfig:
    """Parameterized tile configuration for testing."""
    mt_a: int       # MacroTileA
    mt_b: int       # MacroTileB
    depth_u: int    # DepthU
    stride_a: int   # StrideA0I (in elements)
    stride_b: int   # StrideB1J (in elements)

    @property
    def label(self):
        return f"{self.mt_a}x{self.mt_b}x{self.depth_u}"


# Tile configs to test
TILE_CONFIGS = [
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=64, stride_b=64),
]


def hip_check(result):
    """Check HIP call result."""
    if isinstance(result, tuple):
        err = result[0]
        if err != 0:
            raise RuntimeError(f"HIP error {err}")
        return result[1] if len(result) == 2 else result[1:]
    if result != 0:
        raise RuntimeError(f"HIP error {result}")


def _mock_dtype(num_bytes=2):
    """Create a mock DataType that returns numBytes()."""
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def _create_kernel(cfg):
    """Create a minimal kernel dict matching the given tile config."""
    dtype = _mock_dtype(BPE)
    return {
        "DepthU": cfg.depth_u,
        "MacroTileA": cfg.mt_a,
        "MacroTileB": cfg.mt_b,
        "MacroTile0": cfg.mt_a,
        "MacroTile1": cfg.mt_b,
        "MatrixInstM": 16,
        "MatrixInstK": 32,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": WAVESIZE,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
        },
    }


def _create_writer_for_gpu(cfg):
    """Create a mock writer with register pools laid out for GPU execution.

    Register layout (must match the kernel prologue/epilogue):
      v0           = Serial 
      v1+          = allocated by allocOffsetRegisters and graTileAssignment

      s0:s1        = kernarg_segment_ptr 
      s2           = workgroup_id_x 
      s3           = padding 
      s[4:5]       = output_ptr_A (loaded from kernargs in prologue)
      s[6:7]       = output_ptr_B (loaded from kernargs in prologue)
      s8           = StrideA0I (loaded from kernargs, mapped via .set)
      s9           = StrideB1J (loaded from kernargs, mapped via .set)
      s10+         = sgprPool for temps (sHalfOffset, subtile offsets, etc.)
    """
    writer = SimpleNamespace()

    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                    defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                    defaultPreventOverflow=False, printRP=False)

    # Reserve v0 for Serial (hardware workitem_id)
    writer.vgprPool.checkOut(1)

    # Reserve s0-s9 for hardware regs + kernarg loads
    writer.sgprPool.checkOut(10)

    # Build kernel and TileInfo
    kernel = _create_kernel(cfg)
    tileInfoA = TileInfo('A', kernel)
    tileInfoB = TileInfo('B', kernel)

    writer.states = SimpleNamespace(
        a=SimpleNamespace(tileInfo=tileInfoA),
        b=SimpleNamespace(tileInfo=tileInfoB),
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256},
    )

    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    print("tileInfoA", tileInfoA)
    print("tileInfoB", tileInfoB)

    return writer, kernel, tileInfoA, tileInfoB


def generate_test_kernel(cfg):
    """Generate a test kernel using graTileAssignment's actual output for a given tile config."""
    writer, kernel, tileInfoA, tileInfoB = _create_writer_for_gpu(cfg)

    offsetA_vgpr = tileInfoA.sharedVgprGROffset[0]
    offsetB_vgpr = tileInfoB.sharedVgprGROffset[0]

    # Initialize rocIsa for correct instruction encoding
    from rocisa import rocIsa
    ri = rocIsa.getInstance()
    if not ri.isInit():
        import shutil
        asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
        ri.init((9, 5, 0), asmpath)
    ri.setKernel((9, 5, 0), WAVESIZE)

    module = graTileAssignment(writer, kernel)
    gra_asm = str(module)

    # Scan generated asm for highest register indices used
    vgpr_indices = set(int(m) for m in re.findall(r'\bv(\d+)\b', gra_asm))
    sgpr_indices = set(int(m) for m in re.findall(r'\bs(\d+)\b', gra_asm))

    epilogue_vgpr = max(vgpr_indices | {0}) + 1
    max_vgpr = epilogue_vgpr + 1
    max_vgpr = max(((max_vgpr + 3) // 4) * 4, 4) # align-4 for amdhsa_accum_offset
    max_sgpr = max(sgpr_indices | {9}) + 1

    asm = f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{GFX_TARGET}"

// Register name mappings for graTileAssignment symbolic references
.set vgprSerial, 0
.set sgprStrideA0I, 8
.set sgprStrideB1J, 9

.text
.protected test_gra_offset
.globl test_gra_offset
.p2align 8
.type test_gra_offset,@function

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_gra_offset
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset {max_vgpr}
  .amdhsa_next_free_vgpr {max_vgpr}
  .amdhsa_next_free_sgpr {max_sgpr}
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.text
test_gra_offset:
  // ---- Prologue: Load kernel arguments ----
  // s[0:1] = kernarg ptr (hardware)
  // Layout: output_ptr_A(8B), output_ptr_B(8B), strideA(4B), strideB(4B)
  s_load_dwordx2 s[4:5], s[0:1], 0x00     // output_ptr_A
  s_load_dwordx2 s[6:7], s[0:1], 0x08     // output_ptr_B
  s_load_dword s[sgprStrideA0I], s[0:1], 0x10   // strideA -> s8
  s_load_dword s[sgprStrideB1J], s[0:1], 0x14   // strideB -> s9
  s_waitcnt lgkmcnt(0)

  // v0 = Serial (hardware workitem_id) - already set by hardware

  // ---- Generated graTileAssignment code ----
{gra_asm}
  // ---- Epilogue: Write results to global memory ----
  // byte offset = threadIdx * 4
  v_lshlrev_b32 v{epilogue_vgpr}, 2, v0

  // global_store offsetA -> output_ptr_A[threadIdx]
  global_store_dword v{epilogue_vgpr}, v{offsetA_vgpr}, s[4:5]
  // global_store offsetB -> output_ptr_B[threadIdx]
  global_store_dword v{epilogue_vgpr}, v{offsetB_vgpr}, s[6:7]

  s_waitcnt vmcnt(0)
  s_endpgm

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: test_gra_offset
    .symbol: 'test_gra_offset.kd'
    .language: OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            output_ptr_A
        .size:            8
        .offset:          0
        .value_kind:      global_buffer
        .value_type:      u32
        .address_space:   global
      - .name:            output_ptr_B
        .size:            8
        .offset:          8
        .value_kind:      global_buffer
        .value_type:      u32
        .address_space:   global
      - .name:            strideA
        .size:            4
        .offset:          16
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: {WAVESIZE}
    .sgpr_count: {max_sgpr}
    .vgpr_count: {max_vgpr}
    .max_flat_workgroup_size: {NUM_THREADS}
...
.end_amdgpu_metadata
"""
    return asm, offsetA_vgpr, offsetB_vgpr


def compute_expected_offset(thread_id, stride, mt0, depth_u, bpe, load_width, wavesize):
    """Python reference implementation matching _grComputeOffset logic.

    Traces through the exact instruction sequence in _grComputeOffset:
      1. tmp  = stride * row_id           (VMulLOU32 -> tmpVgpr)
      2. tmp  = col_id << (bpe.bit_len-1) (VLShiftLeftB32 -> tmpVgpr, OVERWRITES #1)
      3. tmp  = col_id + tmp              (VAddU32 -> tmpVgpr)
      4. half = (MT0 * bpe) // 2
      5. tmp2 = split_id * half           (VMulLOU32 -> tmpVgpr+1)
      6. tmp2 = stride * tmp2             (VMulLOU32 -> tmpVgpr+1)
      7. offset = tmp + tmp2              (VAddU32 -> addrVgpr)
    """
    block_size = (depth_u * bpe) // load_width

    # graTileAssignment: new_serial computation
    wave_id = thread_id >> (wavesize.bit_length() - 1)
    new_serial = thread_id & 31
    wave_id = wave_id << 5
    new_serial = (wave_id + new_serial) & 0xFFFFFFFF

    # col_id and row_id from new_serial
    col_id = (new_serial & (block_size - 1)) << (load_width.bit_length() - 1)
    row_id = new_serial >> (block_size.bit_length() - 1)

    # split_id from original Serial
    split_id = (thread_id >> ((wavesize // 2).bit_length() - 1)) & 1

    # _grComputeOffset: exact instruction sequence
    # Step 1: tmp = stride * row_id  (immediately overwritten)
    # Step 2: tmp = col_id << (bpe.bit_length()-1)
    tmp = (col_id << (bpe.bit_length() - 1)) & 0xFFFFFFFF
    # Step 3: tmp = col_id + tmp
    tmp = (col_id + tmp) & 0xFFFFFFFF

    # Step 4-6: split_wave_offset
    half_offset = (mt0 * bpe) // 2
    tmp2 = (split_id * half_offset) & 0xFFFFFFFF
    tmp2 = (stride * tmp2) & 0xFFFFFFFF

    # Step 7: final offset
    offset = (tmp + tmp2) & 0xFFFFFFFF
    return offset


def assemble_kernel(asm_source, output_path):
    """Assemble .s source to .co code object."""
    with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
        f.write(asm_source)
        asm_path = f.name

    obj_path = asm_path.replace(".s", ".o")

    try:
        subprocess.check_call([
            "amdclang++", "-x", "assembler",
            "--target=amdgcn-amd-amdhsa",
            f"-mcpu={GFX_TARGET}",
            "-mwavefrontsize64",
            "-mcode-object-version=5",
            "-o", obj_path,
            asm_path
        ])
        os.rename(obj_path, output_path)
    finally:
        if os.path.exists(asm_path):
            os.unlink(asm_path)
        if os.path.exists(obj_path) and obj_path != output_path:
            os.unlink(obj_path)


def run_on_gpu(co_path, stride_a, stride_b, num_threads):
    """Load code object, launch kernel, read results."""
    hip_check(hip.hipInit(0))
    device = hip_check(hip.hipGetDevice())

    module = hip_check(hip.hipModuleLoad(co_path.encode() if isinstance(co_path, str) else co_path))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"test_gra_offset"))

    buf_size = num_threads * 4  # 4 bytes per u32
    d_out_a = hip_check(hip.hipMalloc(buf_size))
    d_out_b = hip_check(hip.hipMalloc(buf_size))

    hip_check(hip.hipMemset(d_out_a, 0, buf_size))
    hip_check(hip.hipMemset(d_out_b, 0, buf_size))

    ptr_a_int = int(d_out_a)
    ptr_b_int = int(d_out_b)

    class KernelArgs(ctypes.Structure):
        _fields_ = [
            ("ptr_a", ctypes.c_uint64),
            ("ptr_b", ctypes.c_uint64),
            ("stride_a", ctypes.c_uint32),
            ("stride_b", ctypes.c_uint32),
        ]

    kargs = KernelArgs(ptr_a_int, ptr_b_int, stride_a, stride_b)
    kargs_size = ctypes.sizeof(kargs)
    kargs_ptr = ctypes.addressof(kargs)

    HIP_LAUNCH_PARAM_BUFFER_POINTER = 0x01
    HIP_LAUNCH_PARAM_BUFFER_SIZE    = 0x02
    HIP_LAUNCH_PARAM_END            = 0x03

    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(HIP_LAUNCH_PARAM_BUFFER_POINTER),
        ctypes.c_void_p(kargs_ptr),
        ctypes.c_void_p(HIP_LAUNCH_PARAM_BUFFER_SIZE),
        ctypes.c_void_p(ctypes.addressof(ctypes.c_size_t(kargs_size))),
        ctypes.c_void_p(HIP_LAUNCH_PARAM_END),
    )

    hip_check(hip.hipModuleLaunchKernel(
        kernel,
        1, 1, 1,                 # grid
        num_threads, 1, 1,       # block
        0,                       # shared mem
        None,                    # stream
        None,                    # kernel params (unused with extra)
        extra                    # extra params
    ))
    hip_check(hip.hipDeviceSynchronize())

    h_out_a = bytearray(buf_size)
    h_out_b = bytearray(buf_size)
    hip_check(hip.hipMemcpyDtoH(h_out_a, d_out_a, buf_size))
    hip_check(hip.hipMemcpyDtoH(h_out_b, d_out_b, buf_size))

    hip_check(hip.hipFree(d_out_a))
    hip_check(hip.hipFree(d_out_b))
    hip_check(hip.hipModuleUnload(module))

    results_a = struct.unpack(f"{num_threads}I", h_out_a)
    results_b = struct.unpack(f"{num_threads}I", h_out_b)
    return results_a, results_b


@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGraTileAssignmentGPU:

    @pytest.fixture(params=TILE_CONFIGS, ids=lambda c: c.label)
    def tile_env(self, request, tmp_path):
        """Generate and compile the test kernel for a given tile config."""
        cfg = request.param
        co_path = str(tmp_path / f"test_gra_offset_{cfg.label}.co")
        asm, offsetA_vgpr, offsetB_vgpr = generate_test_kernel(cfg)

        asm_path = str(tmp_path / f"test_gra_offset_{cfg.label}.s")
        with open(asm_path, "w") as f:
            f.write(asm)
        print(f"\n[{cfg.label}] Assembly written to: {asm_path}")
        print(f"[{cfg.label}] Offset A in v{offsetA_vgpr}, Offset B in v{offsetB_vgpr}")

        assemble_kernel(asm, co_path)
        print(f"[{cfg.label}] Code object: {co_path}")

        return SimpleNamespace(
            cfg=cfg,
            co_path=co_path,
            offsetA_vgpr=offsetA_vgpr,
            offsetB_vgpr=offsetB_vgpr,
        )

    def test_offset_a(self, tile_env):
        """Validate sharedVgprGROffset[0] for matrix A across all threads."""
        cfg = tile_env.cfg
        results_a, _ = run_on_gpu(tile_env.co_path, cfg.stride_a, cfg.stride_b, NUM_THREADS)

        for tid in range(NUM_THREADS):
            expected = compute_expected_offset(tid, cfg.stride_a, cfg.mt_a, cfg.depth_u, BPE, LOAD_WIDTH, WAVESIZE)
            actual = results_a[tid]
            assert actual == expected, \
                f"[{cfg.label}] A offset mismatch at tid={tid}: got {actual}, expected {expected}"

    def test_offset_b(self, tile_env):
        """Validate sharedVgprGROffset[0] for matrix B across all threads."""
        cfg = tile_env.cfg
        _, results_b = run_on_gpu(tile_env.co_path, cfg.stride_a, cfg.stride_b, NUM_THREADS)

        for tid in range(NUM_THREADS):
            expected = compute_expected_offset(tid, cfg.stride_b, cfg.mt_b, cfg.depth_u, BPE, LOAD_WIDTH, WAVESIZE)
            actual = results_b[tid]
            assert actual == expected, \
                f"[{cfg.label}] B offset mismatch at tid={tid}: got {actual}, expected {expected}"

    def test_print_first_wave(self, tile_env):
        """Print offsets for the first wave for visual inspection."""
        cfg = tile_env.cfg
        results_a, results_b = run_on_gpu(tile_env.co_path, cfg.stride_a, cfg.stride_b, NUM_THREADS)

        print(f"\n[{cfg.label}] {'tid':>4} | {'offsetA':>10} | {'offsetB':>10}")
        print("-" * 32)
        for tid in range(WAVESIZE):
            print(f"{tid:4d} | {results_a[tid]:10d} | {results_b[tid]:10d}")


def print_offset_grid(label, results, wavesize, num_waves):
    """Print offsets as a 2D grid: rows = waves, columns = lanes."""
    print(f"\n--- {label} offsets (rows=waves, cols=lanes) ---")
    # Header: lane indices
    print(f"{'wave':>6}", end="")
    for lane in range(wavesize):
        print(f" {lane:>6}", end="")
    print()
    print("-" * (7 + 7 * wavesize))
    for w in range(num_waves):
        print(f"{w:>6}", end="")
        for lane in range(wavesize):
            tid = w * wavesize + lane
            print(f" {results[tid]:>6}", end="")
        print()


if __name__ == "__main__":
    """Run standalone without pytest."""
    import argparse
    parser = argparse.ArgumentParser(description="GPU test for graTileAssignment")
    parser.add_argument("--grid", action="store_true",
                        help="Display offsets as 2D grid (waves x lanes) for A and B")
    args = parser.parse_args()

    for cfg in TILE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Tile Config: {cfg.label}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            co_path = os.path.join(tmp_dir, f"test_gra_offset_{cfg.label}.co")
            asm, offsetA_vgpr, offsetB_vgpr = generate_test_kernel(cfg)

            asm_path = os.path.join(tmp_dir, f"test_gra_offset_{cfg.label}.s")
            with open(asm_path, "w") as f:
                f.write(asm)
            print(f"Assembly: {asm_path}")
            print(f"Offset A in v{offsetA_vgpr}, Offset B in v{offsetB_vgpr}")

            # Print the generated assembly for inspection
            print("\n--- Generated Assembly (graTileAssignment section) ---")
            in_gra = False
            for line in asm.split('\n'):
                if 'Generated graTileAssignment' in line:
                    in_gra = True
                if in_gra:
                    print(line)
                if 'Epilogue' in line and in_gra:
                    break
            print("--- End ---\n")

            assemble_kernel(asm, co_path)
            print(f"Code object: {co_path}")

            if HAS_HIP:
                results_a, results_b = run_on_gpu(co_path, cfg.stride_a, cfg.stride_b, NUM_THREADS)

                if args.grid:
                    print_offset_grid(f"Matrix A ({cfg.label})", results_a, WAVESIZE, NUM_WAVES)
                    print_offset_grid(f"Matrix B ({cfg.label})", results_b, WAVESIZE, NUM_WAVES)
                else:
                    print(f"\n{'tid':>4} | {'offsetA':>10} | {'offsetB':>10} | {'expA':>10} | {'expB':>10} | {'ok':>3}")
                    print("-" * 60)

                errors = 0
                for tid in range(NUM_THREADS):
                    exp_a = compute_expected_offset(tid, cfg.stride_a, cfg.mt_a, cfg.depth_u, BPE, LOAD_WIDTH, WAVESIZE)
                    exp_b = compute_expected_offset(tid, cfg.stride_b, cfg.mt_b, cfg.depth_u, BPE, LOAD_WIDTH, WAVESIZE)
                    ok = "OK" if (results_a[tid] == exp_a and results_b[tid] == exp_b) else "FAIL"
                    if ok == "FAIL":
                        errors += 1
                    if not args.grid and (tid < 64 or ok == "FAIL"):
                        print(f"{tid:4d} | {results_a[tid]:10d} | {results_b[tid]:10d} | {exp_a:10d} | {exp_b:10d} | {ok}")

                print(f"\nTotal: {NUM_THREADS} threads, {errors} errors")
            else:
                print("HIP not available - assembly generated but not executed")
                print(f"Manual test: compile and run {asm_path}")
