#!/usr/bin/env python3
################################################################################
# GPU functional test for graTileAssignment with parameterized tile configs
#
# Generic test framework: each test kernel exports a single register (vgpr or
# sgpr) to one output buffer.  Different kernel variants are generated for each
# register under test.
#
# Usage:
#   pytest test_graTileAssignment.py -v -s
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
    use_swizzling: bool = False  # Whether to enable swizzling

    @property
    def label(self):
        swz = "_swz" if self.use_swizzling else ""
        return f"{self.mt_a}x{self.mt_b}x{self.depth_u}{swz}"


# Tile configs to test
TILE_CONFIGS = [
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=64, stride_b=64, use_swizzling=False),
    # TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=64, stride_b=64, use_swizzling=True),
    # TileConfig(mt_a=16, mt_b=64, depth_u=64, stride_a=64, stride_b=64, use_swizzling=True),
    # No change in offset calculation (will use OOB to mask 2nd 16x128 sub-tile)
    # TileConfig(mt_a=16, mt_b=64, depth_u=64, stride_a=64, stride_b=64, use_swizzling=True),
    # TileConfig(mt_a=80, mt_b=64, depth_u=64, stride_a=64, stride_b=64, use_swizzling=True),
]


# ---- HIP helpers ----

def hip_check(result):
    """Check HIP call result."""
    if isinstance(result, tuple):
        err = result[0]
        if err != 0:
            raise RuntimeError(f"HIP error {err}")
        return result[1] if len(result) == 2 else result[1:]
    if result != 0:
        raise RuntimeError(f"HIP error {result}")


# ---- Mock / setup helpers ----

def _mock_dtype(num_bytes=2):
    """Create a mock DataType that returns numBytes()."""
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def _create_kernel(cfg):
    """Create a minimal kernel dict matching the given tile config."""
    dtype = _mock_dtype(BPE)
    if ((cfg.mt_a//16) % 2 == 0) and ((cfg.mt_b//16) % 2 == 0):
        MIWaveGroup = [2,2]
    elif ((cfg.mt_a//16) % 2 != 0) and ((cfg.mt_b//16) % 4 == 0):
        MIWaveGroup = [1,4]
    elif ((cfg.mt_a//16) % 4 == 0) and ((cfg.mt_b//16) % 2 != 0):
        MIWaveGroup = [4,1]
    else:
        raise ValueError(f"Unsupported tile config for wave grouping: mt_a={cfg.mt_a}, mt_b={cfg.mt_b}")

    return {
        "DepthU": cfg.depth_u,
        "MacroTileA": cfg.mt_a,
        "MacroTileB": cfg.mt_b,
        "MacroTile0": cfg.mt_a,
        "MacroTile1": cfg.mt_b,
        "MatrixInstM": 16,
        "MatrixInstK": 32,
        "MIWaveGroup": MIWaveGroup,
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
      s[4:5]       = output_ptr (loaded from kernargs in prologue)
      s[6:7]       = free
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
    print("Kernel config:", kernel)
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


def _init_rocisa():
    """Initialize rocIsa singleton if needed."""
    from rocisa import rocIsa
    ri = rocIsa.getInstance()
    if not ri.isInit():
        import shutil
        asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
        ri.init((9, 5, 0), asmpath)
    ri.setKernel((9, 5, 0), WAVESIZE)


def generate_gra_asm(cfg):
    """Run graTileAssignment and return (gra_asm, tileInfoA, tileInfoB, kernel)."""
    writer, kernel, tileInfoA, tileInfoB = _create_writer_for_gpu(cfg)
    _init_rocisa()

    module = graTileAssignment(writer, kernel, useSwizzling=cfg.use_swizzling)
    gra_asm = str(module)
    return gra_asm, tileInfoA, tileInfoB, kernel


# ---- Generic kernel generator ----

def generate_export_kernel(gra_asm, export_reg, is_sgpr=False):
    """Generate a kernel that runs gra_asm and exports a single register.

    Args:
        gra_asm:    Assembly string from graTileAssignment.
        export_reg: Register index to export (e.g. 3 for v3 or s3).
        is_sgpr:    True to export an sgpr (uniform value, broadcast to all
                    threads), False to export a vgpr (per-thread value).

    Kernarg layout (20 bytes, padded to 24):
        offset  0: output_ptr  (8B, global_buffer)
        offset  8: strideA     (4B, by_value)
        offset 12: strideB     (4B, by_value)

    Returns:
        Assembly source string.
    """
    # Find highest register indices used by gra_asm
    vgpr_indices = set(int(m) for m in re.findall(r'\bv(\d+)\b', gra_asm))
    sgpr_indices = set(int(m) for m in re.findall(r'\bs(\d+)\b', gra_asm))

    tmp_vgpr = max(vgpr_indices | {0}) + 1      # byte-offset register
    data_vgpr = tmp_vgpr + 1 if is_sgpr else export_reg
    max_vgpr = max(tmp_vgpr, data_vgpr) + 1
    max_vgpr = max(((max_vgpr + 3) // 4) * 4, 4)  # align-4 for accum_offset
    max_sgpr = max(sgpr_indices | {9}) + 1

    # Build epilogue
    epilogue = f"  v_lshlrev_b32 v{tmp_vgpr}, 2, v0\n"
    if is_sgpr:
        epilogue += f"  v_mov_b32 v{data_vgpr}, s{export_reg}\n"
    epilogue += f"  global_store_dword v{tmp_vgpr}, v{data_vgpr}, s[4:5]\n"

    return f"""\
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
  s_load_dwordx2 s[4:5], s[0:1], 0x00     // output_ptr
  s_load_dword s[sgprStrideA0I], s[0:1], 0x08   // strideA -> s8
  s_load_dword s[sgprStrideB1J], s[0:1], 0x0c   // strideB -> s9
  s_waitcnt lgkmcnt(0)

  // ---- Generated graTileAssignment code ----
{gra_asm}
  // ---- Epilogue: Export register ----
{epilogue}
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
      - .name:            output_ptr
        .size:            8
        .offset:          0
        .value_kind:      global_buffer
        .value_type:      u32
        .address_space:   global
      - .name:            strideA
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
    .kernarg_segment_size: 16
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


# ---- Assemble / run ----

def assemble_kernel(asm_source, output_path):
    """Assemble .s source to .co code object."""
    with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
        f.write(asm_source)
        asm_path = f.name

    obj_path = asm_path.replace(".s", ".o")

    try:
        #  print(asm_source)
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
    """Load code object, launch kernel, read single output buffer."""
    hip_check(hip.hipInit(0))
    device = hip_check(hip.hipGetDevice())

    module = hip_check(hip.hipModuleLoad(co_path.encode() if isinstance(co_path, str) else co_path))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"test_gra_offset"))

    buf_size = num_threads * 4  # 4 bytes per u32
    d_out = hip_check(hip.hipMalloc(buf_size))
    hip_check(hip.hipMemset(d_out, 0, buf_size))

    class KernelArgs(ctypes.Structure):
        _fields_ = [
            ("ptr_out", ctypes.c_uint64),
            ("stride_a", ctypes.c_uint32),
            ("stride_b", ctypes.c_uint32),
        ]

    kargs = KernelArgs(int(d_out), stride_a, stride_b)
    kargs_size = ctypes.c_size_t(ctypes.sizeof(kargs))

    HIP_LAUNCH_PARAM_BUFFER_POINTER = 0x01
    HIP_LAUNCH_PARAM_BUFFER_SIZE    = 0x02
    HIP_LAUNCH_PARAM_END            = 0x03

    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(HIP_LAUNCH_PARAM_BUFFER_POINTER),
        ctypes.c_void_p(ctypes.addressof(kargs)),
        ctypes.c_void_p(HIP_LAUNCH_PARAM_BUFFER_SIZE),
        ctypes.c_void_p(ctypes.addressof(kargs_size)),
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

    h_out = bytearray(buf_size)
    hip_check(hip.hipMemcpyDtoH(h_out, d_out, buf_size))

    hip_check(hip.hipFree(d_out))
    hip_check(hip.hipModuleUnload(module))
    

    return struct.unpack(f"{num_threads}I", h_out)


def build_and_run(gra_asm, export_reg, is_sgpr, cfg, tmp_path, label):
    """Generate, assemble, run a single-register export kernel. Returns results tuple."""
    sys.stdout.flush()  # flush before GPU calls to avoid buffering issues with HIP runtime
    asm = generate_export_kernel(gra_asm, export_reg, is_sgpr=is_sgpr)
    co_path = str(tmp_path / f"test_{label}.co")
    asm_path = str(tmp_path / f"test_{label}.s")
    with open(asm_path, "w") as f:
        f.write(asm)
    assemble_kernel(asm, co_path)
    return run_on_gpu(co_path, cfg.stride_a, cfg.stride_b, NUM_THREADS)


# ---- Reference implementations ----

def compute_expected_offset(thread_id, stride, mt0, depth_u, bpe, load_width, wavesize,
                            use_swizzling=False):
    """Python reference implementation matching _grComputeOffset logic.

    When use_swizzling=True, applies the LDS bank-conflict avoidance
    swizzle (quad_perm + rotation) before computing the final byte offset.
    """
    block_size = (depth_u * bpe) // load_width
    new_serial = (thread_id & (wavesize//2 - 1)) | ((thread_id // wavesize) * (wavesize//2))
    wave_split_id = (thread_id // (wavesize//2)) % 2

    # local col/row in wave
    col = new_serial % block_size
    row = new_serial // block_size

    if use_swizzling:
        col = col + 1  if col % 2 ==0 else col - 1  # swap even/odd cols for initial swizzle
        rowLds = row // 2
        col = (col + (block_size - (rowLds // 2) * 2))%block_size  # rotation to avoid bank conflicts: block_size - (lds_row_id//4)*2

    # number of rows per wave (half-wave because of wave_split_id)
    numRows = (wavesize // 2) * load_width // (depth_u * bpe)

    row_g = row + wave_split_id * (mt0 // 2)
    col_g = col * load_width
    return row_g * stride * bpe + col_g


def compute_expected_subtile(subtile_id0, stride, depth_u, bpe, load_width, wavesize):
    """Compute expected subtile register value: rowsPerWave * bpe * subtileId0 * stride."""
    block_size = (depth_u * bpe) // load_width
    rows_per_wave = wavesize // block_size // 2
    return rows_per_wave * bpe * subtile_id0 * stride


# ---- Pytest tests ----

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGraTileAssignmentGPU:

    @pytest.fixture(params=TILE_CONFIGS, ids=lambda c: c.label)
    def gra_env(self, request, tmp_path):
        """Generate graTileAssignment asm once per tile config."""
        cfg = request.param
        gra_asm, tileInfoA, tileInfoB, kernel = generate_gra_asm(cfg)
        return SimpleNamespace(
            cfg=cfg,
            gra_asm=gra_asm,
            tileInfoA=tileInfoA,
            tileInfoB=tileInfoB,
            kernel=kernel,
            tmp_path=tmp_path,
        )

    def test_offset_a(self, gra_env):
        """Validate all sharedVgprGROffset vgprs for matrix A across all threads."""
        cfg = gra_env.cfg
        for idx, reg in enumerate(gra_env.tileInfoA.sharedVgprGROffset):
            results = build_and_run(gra_env.gra_asm, reg, False, cfg, gra_env.tmp_path,
                                    f"offsetA_v{reg}_{cfg.label}")

            for tid in range(NUM_THREADS):
                expected = compute_expected_offset(tid, cfg.stride_a, cfg.mt_a, cfg.depth_u,
                                                   BPE, LOAD_WIDTH, WAVESIZE, cfg.use_swizzling)
                assert results[tid] == expected, \
                    f"[{cfg.label}] A offset[{idx}] v{reg} mismatch at tid={tid}: got {results[tid]}, expected {expected}"

    def test_offset_b(self, gra_env):
        """Validate all sharedVgprGROffset vgprs for matrix B across all threads."""
        cfg = gra_env.cfg
        for idx, reg in enumerate(gra_env.tileInfoB.sharedVgprGROffset):
            results = build_and_run(gra_env.gra_asm, reg, False, cfg, gra_env.tmp_path,
                                    f"offsetB_v{reg}_{cfg.label}")

            for tid in range(NUM_THREADS):
                expected = compute_expected_offset(tid, cfg.stride_b, cfg.mt_b, cfg.depth_u,
                                                   BPE, LOAD_WIDTH, WAVESIZE, cfg.use_swizzling)
                assert results[tid] == expected, \
                    f"[{cfg.label}] B offset[{idx}] v{reg} mismatch at tid={tid}: got {results[tid]}, expected {expected}"

    def test_subtile_registers_a(self, gra_env):
        """Validate localSubtilesRegister values for matrix A."""
        cfg = gra_env.cfg
        tileInfo = gra_env.tileInfoA
        for st in tileInfo.localSubtiles:
            for reg in tileInfo.localSubtilesRegister[st.regListId]:
                results = build_and_run(gra_env.gra_asm, reg, st.useSgpr, cfg,
                                        gra_env.tmp_path,
                                        f"subtileA_s{reg}_{cfg.label}")
                expected = compute_expected_subtile(st.subtileId[0], cfg.stride_a,
                                                    cfg.depth_u, BPE, LOAD_WIDTH, WAVESIZE)
                # sgpr is uniform: check thread 0
                actual = results[0]
                assert actual == expected, \
                    f"[{cfg.label}] A subtile s{reg} (subtileId0={st.subtileId[0]}): " \
                    f"got {actual}, expected {expected}"

    def test_subtile_registers_b(self, gra_env):
        """Validate localSubtilesRegister values for matrix B."""
        cfg = gra_env.cfg
        tileInfo = gra_env.tileInfoB
        for st in tileInfo.localSubtiles:
            for reg in tileInfo.localSubtilesRegister[st.regListId]:
                results = build_and_run(gra_env.gra_asm, reg, st.useSgpr, cfg,
                                        gra_env.tmp_path,
                                        f"subtileB_s{reg}_{cfg.label}")
                expected = compute_expected_subtile(st.subtileId[0], cfg.stride_b,
                                                    cfg.depth_u, BPE, LOAD_WIDTH, WAVESIZE)
                actual = results[0]
                assert actual == expected, \
                    f"[{cfg.label}] B subtile s{reg} (subtileId0={st.subtileId[0]}): " \
                    f"got {actual}, expected {expected}"



# ---- Utilities ----

def print_offset_grid(label, results, wavesize, num_waves):
    """Print offsets as a 2D grid: rows = waves, columns = lanes."""
    print(f"\n--- {label} offsets (rows=waves, cols=lanes) ---")
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
    parser.add_argument("--debug", action="store_true",
                        help="Display expected matrix in grid mode (implies --grid)")
    args = parser.parse_args()
    if args.debug:
        args.grid = True

    for cfg in TILE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Tile Config: {cfg.label}")
        print(f"{'='*60}")

        gra_asm, tileInfoA, tileInfoB, kernel = generate_gra_asm(cfg)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

            # Print the generated assembly for inspection
            print("\n--- Generated Assembly (graTileAssignment section) ---")
            in_gra = False
            for line in gra_asm.split('\n'):
                if 'GR Offset' in line or in_gra:
                    in_gra = True
                    print(line)
            print("--- End ---\n")

            if HAS_HIP:
                # Test all sharedVgprGROffset vgprs for both matrices
                for tc, tileInfo, stride, mt in [("A", tileInfoA, cfg.stride_a, cfg.mt_a),
                                                  ("B", tileInfoB, cfg.stride_b, cfg.mt_b)]:
                    for idx, reg in enumerate(tileInfo.sharedVgprGROffset):
                        print("Regl",reg)
                        results = build_and_run(gra_asm, reg, False, cfg, tmp_path,
                                                f"offset{tc}_v{reg}_{cfg.label}")

                        if args.grid:
                            print_offset_grid(f"Matrix {tc} GPU offset[{idx}] v{reg} ({cfg.label})",
                                              results, WAVESIZE, NUM_WAVES)

                            if args.debug:
                                expected = [compute_expected_offset(tid, stride, mt, cfg.depth_u,
                                                                     BPE, LOAD_WIDTH, WAVESIZE, cfg.use_swizzling)
                                            for tid in range(NUM_THREADS)]
                                print_offset_grid(f"Matrix {tc} EXPECTED offset[{idx}] ({cfg.label})",
                                                  expected, WAVESIZE, NUM_WAVES)

                                mismatches = sum(1 for t in range(NUM_THREADS) if results[t] != expected[t])
                                if mismatches:
                                    print(f"\n--- Matrix {tc} offset[{idx}] DIFF ({mismatches} mismatches) ---")
                                    for w in range(NUM_WAVES):
                                        print(f"  w{w}: ", end="")
                                        for lane in range(WAVESIZE):
                                            tid = w * WAVESIZE + lane
                                            if results[tid] != expected[tid]:
                                                print(f" t{tid}:{results[tid]}!={expected[tid]}", end="")
                                        print()
                                else:
                                    print(f"\n  Matrix {tc} offset[{idx}]: all match.")

                        errors = 0
                        for tid in range(NUM_THREADS):
                            exp = compute_expected_offset(tid, stride, mt, cfg.depth_u,
                                                          BPE, LOAD_WIDTH, WAVESIZE, cfg.use_swizzling)
                            if results[tid] != exp:
                                errors += 1
                                if not args.grid:
                                    print(f"  FAIL {tc} offset[{idx}] v{reg} tid={tid}: got {results[tid]}, expected {exp}")
                            elif not args.grid and tid < 64:
                                print(f"  OK   {tc} offset[{idx}] v{reg} tid={tid}: {results[tid]}")

                        print(f"  Matrix {tc} offset[{idx}] v{reg}: {NUM_THREADS} threads, {errors} errors")

                # Subtile registers
                for tc, tileInfo, stride in [("A", tileInfoA, cfg.stride_a),
                                              ("B", tileInfoB, cfg.stride_b)]:
                    for st in tileInfo.localSubtiles:
                        for reg in tileInfo.localSubtilesRegister[st.regListId]:
                            print("Regl",reg)
                            results = build_and_run(gra_asm, reg, st.useSgpr, cfg, tmp_path,
                                                    f"subtile{tc}_s{reg}_{cfg.label}")
                            expected = compute_expected_subtile(st.subtileId[0], stride,
                                                                cfg.depth_u, BPE, LOAD_WIDTH, WAVESIZE)
                            actual = results[0]
                            status = "OK" if actual == expected else "FAIL"
                            print(f"  Subtile {tc} s{reg} (id0={st.subtileId[0]}): {actual} (expected {expected}) {status}")
            else:
                print("HIP not available - assembly generated but not executed")
