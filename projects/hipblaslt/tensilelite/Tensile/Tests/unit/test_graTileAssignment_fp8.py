#!/usr/bin/env python3
################################################################################
# GPU functional test for graTileAssignment with FP8 (bpe=1, MatrixInstK=128)
#
# FP8 uses a block-swap GR swizzle instead of the FP4 pair-swap + rotation:
#   - GR load 1: colId ^= (ldsRowId & 1) * 4   (XOR 4 for odd ldsRowId)
#   - GR load 2: rotatedColId = ((colId & 3) + 2) % 4  +  (colId & 4)
#
# Usage:
#   pytest test_graTileAssignment_fp8.py -v -s
################################################################################

import math
import os
import struct
import sys
import tempfile

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    LOAD_WIDTH, WAVESIZE, NUM_THREADS, NUM_WAVES,
    create_writer,
    init_rocisa,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
    generate_export_epilogue,
    print_offset_grid,
)
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from Tensile.Components.SubtileBasedKernel import TileInfo, graTileAssignment

# ---- FP8-specific constants ----
BPE_FP8        = 1     # 1 byte per FP8 element
MATRIX_INST_K  = 128   # v_mfma_f32_16x16x128_fp8_fp8

EXPORT_LOAD_PARAMS = (
    (4, 2, 0x00, "output_ptr"),
    ("StrideA0I", 1, 0x08, "strideA"),
    ("StrideB1J", 1, 0x0c, "strideB"),
)

EXPORT_ARGS = (
    ("output_ptr", 8, "global_buffer", "u32"),
    ("strideA",    4, "by_value",      "u32"),
    ("strideB",    4, "by_value",      "u32"),
)


# ---- FP8 kernel / writer helpers ----

def _mock_dtype_fp8():
    """Mock FP8 DataType: bpe=1, is8bitFloat()=True."""
    mock = MagicMock()
    mock.numBytes.return_value = BPE_FP8
    mock.is8bitFloat.return_value = True
    return mock


def _create_kernel_fp8(cfg, mi_wave_group=None):
    """Create a minimal kernel dict for FP8 tile configs (MatrixInstK=128)."""
    dtype = _mock_dtype_fp8()

    if mi_wave_group is not None:
        MIWaveGroup = mi_wave_group
    elif ((cfg.mt_a // 16) % 2 == 0) and ((cfg.mt_b // 16) % 2 == 0):
        MIWaveGroup = [2, 2]
    elif ((cfg.mt_a // 16) % 2 != 0) and ((cfg.mt_b // 16) % 4 == 0):
        MIWaveGroup = [1, 4]
    elif ((cfg.mt_a // 16) % 4 == 0) and ((cfg.mt_b // 16) % 2 != 0):
        MIWaveGroup = [4, 1]
    else:
        raise ValueError(f"Unsupported FP8 tile config: mt_a={cfg.mt_a}, mt_b={cfg.mt_b}")

    return {
        "DepthU":      cfg.depth_u,
        "_DepthU":     cfg.depth_u,
        "_DepthUA":    cfg.depth_u,
        "_DepthUB":    cfg.depth_u,
        "MacroTileA":  cfg.mt_a,
        "MacroTileB":  cfg.mt_b,
        "MacroTile0":  cfg.mt_a,
        "MacroTile1":  cfg.mt_b,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": MATRIX_INST_K,
        "MIWaveGroup": MIWaveGroup,
        "WavefrontSize": WAVESIZE,
        "UseSubtileImpl": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "ProblemType": {
            "DataTypeA":      dtype,
            "DataTypeB":      dtype,
            "ComputeDataType": MagicMock(**{"numBytes.return_value": 4}),
            # Non-zero MXBlock triggers subtileShape=[1,1] for FP8
            "MXBlockA": 32,
            "MXBlockB": 32,
        },
    }


def create_writer_fp8(cfg, mi_wave_group=None):
    """Create a mock writer with FP8 kernel (MatrixInstK=128, bpe=1).

    Mirrors create_writer() from gpu_test_helpers but uses _create_kernel_fp8.
    """
    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprs = {}
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr,
                                   defaultPreventOverflow=False, printRP=False)

    # v0 reserved for Serial (hardware workitem_id)
    writer.vgprPool.checkOut(1)

    kernel    = _create_kernel_fp8(cfg, mi_wave_group=mi_wave_group)
    tileInfoA = TileInfo('A', kernel)
    tileInfoB = TileInfo('B', kernel)

    writer.states = SimpleNamespace(
        a=SimpleNamespace(tileInfo=tileInfoA),
        b=SimpleNamespace(tileInfo=tileInfoB),
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
    )

    readSize      = 2 * tileInfoA.subtileSize
    numASubtiles  = tileInfoA.globalSubtileGrid[0] * tileInfoA.globalSubtileGrid[1]
    writer.ldsStartOffsetA = 0
    writer.ldsStartOffsetB = (
        (numASubtiles * tileInfoA.subtileSize + readSize - 1) // readSize
    ) * readSize

    return writer, kernel, tileInfoA, tileInfoB


# ---- GRA assembly generator ----

def generate_gra_asm_fp8(cfg):
    """Run graTileAssignment for FP8 and return (gra_asm, writer, tileInfoA, tileInfoB, kernel)."""
    writer, kernel, tileInfoA, tileInfoB = create_writer_fp8(cfg)
    init_rocisa()

    # Reserve s0-s11 for hardware regs + kernarg loads
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    prologue = generate_load_params(EXPORT_LOAD_PARAMS)
    module   = graTileAssignment(writer, kernel, useSwizzling=cfg.use_swizzling)
    gra_asm  = f"{prologue}\n{module}"
    return gra_asm, writer, tileInfoA, tileInfoB, kernel


def export_register(writer, test_asm, export_reg, is_sgpr, cfg, tmp_path, label):
    """Assemble, run on GPU, return per-thread u32 results."""
    epilogue, allocated = generate_export_epilogue(writer, export_reg, is_sgpr)
    kernel_asm = generate_kernel_asm(f"{test_asm}\n{epilogue}", writer, EXPORT_ARGS)
    for v in allocated:
        writer.vgprPool.checkIn(v)

    raw = assemble_and_run(kernel_asm, tmp_path, label, NUM_THREADS * 4,
                           scalars=(cfg.stride_a, cfg.stride_b))
    return struct.unpack(f"{NUM_THREADS}I", raw)


# ---- Reference implementations ----

def compute_expected_offset_fp8(thread_id, cfg, tileInfo):
    """Python reference for FP8 graTileAssignment / _grComputeOffset.

    GR load 1 swizzle (block swap):
      swap_bit = ldsRowId & 1          (ldsRowId = laneId >> log2(blockSize) >> log2(numRowsPerLDSBanks))
      colId   ^= swap_bit * 4          (XOR 4 flips the K_lo/K_hi block bit for odd ldsRowId)

    GR load 2 swizzle (intra-block K_group rotation +2):
      colId2   = ((colId & 3) + 2) % 4  +  (colId & 4)
    """
    stride       = cfg.stride_a if tileInfo.tc == 'A' else cfg.stride_b
    bpe          = BPE_FP8
    depthUBytes  = cfg.depth_u * bpe
    blockSize    = depthUBytes // LOAD_WIDTH
    numRowsPerLDSBanks = (64 * 4) // depthUBytes  # ldsRowBankSize / depthUBytes

    waveId = thread_id // WAVESIZE
    laneId = thread_id % WAVESIZE

    # --- colId swizzle: step 1 block swap, step 2 K_group rotation ---
    colId    = thread_id & (blockSize - 1)
    rowInWave = laneId >> (blockSize.bit_length() - 1)
    ldsRowId  = rowInWave >> (numRowsPerLDSBanks.bit_length() - 1)
    swap_bit  = ldsRowId & 1
    colId    ^= swap_bit * 4           # step 1: XOR 4 for odd ldsRowId (block swap)

    # step 2: K_group rotation for M_rows 8..15
    # For loadRatioGR != 0.5: odd waveId means the wave covers rows 8..15 → rotation=+2
    # For loadRatioGR == 0.5: rotation is applied per-load in _grComputeAllOffsets instead
    if tileInfo.loadRatioGR != 0.5:
        rotation = (waveId & 1) * 2
        colId = ((colId & 3) + rotation) % 4 + (colId & 4)

    colId_bytes = colId * LOAD_WIDTH

    # --- _grComputeRowOffset ---
    numRowsPerWave   = WAVESIZE // blockSize
    partitionOffset  = tileInfo.mmaTileShape[0] * tileInfo.localSubtileGrid[0]

    if tileInfo.loadRatioGR == 1.0:
        localRow     = waveId & 1
        partitionRow = waveId >> 1
    elif tileInfo.loadRatioGR == 0.5:
        localRow     = 0
        partitionRow = waveId
    elif tileInfo.loadRatioGR == 2.0:
        localRow     = waveId
        partitionRow = 0
    else:
        raise NotImplementedError(f"Unsupported loadRatioGR: {tileInfo.loadRatioGR}")

    localRow     = localRow << (numRowsPerWave.bit_length() - 1)
    partitionRow = partitionOffset * partitionRow
    waveRowOffset = localRow + partitionRow

    # --- _grComputeOffset ---
    rowId    = laneId >> (blockSize.bit_length() - 1)
    totalRow = rowId + waveRowOffset
    base     = totalRow * stride * bpe + colId_bytes

    if tileInfo.numGRPerSubtile == 1:
        return [base]

    # GR load 2: advance row + intra-block K_group rotation +2
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    rowAdvance  = math.ceil(subtileSize * tileInfo.loadRatioGR)
    totalRow2   = totalRow + rowAdvance

    # ((colId & 3) + 2) % 4  +  (colId & 4)  — rotate K_group within each block
    colId2       = ((colId & 3) + 2) % 4 + (colId & 4)
    colId2_bytes = colId2 * LOAD_WIDTH
    offset2      = totalRow2 * stride * bpe + colId2_bytes
    return [base, offset2]


def compute_expected_subtile_fp8(regId, stride, tileInfo):
    """Expected localSubtilesRegister value for FP8: rowOffset * bpe * regId * stride."""
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    rowOffset   = 2 * subtileSize if tileInfo.loadRatioGR == 2.0 else subtileSize
    return rowOffset * BPE_FP8 * regId * stride


# ---- FP8 tile configs ----
#
# All configs use depth_u=128 and MatrixInstK=128 (one subtile covers full K).
#
#  2x2 wave group  → loadRatioGR=1.0 for both A and B (numGRPerSubtile=1)
#                    Tests block-swap swizzle only (no second GR).
#  4x1 wave group  → A: loadRatioGR=0.5 (numGRPerSubtile=2) — tests second GR rotation
#                    B: loadRatioGR=2.0 (numGRPerSubtile=1)
#  1x4 wave group  → A: loadRatioGR=2.0 (numGRPerSubtile=1)
#                    B: loadRatioGR=0.5 (numGRPerSubtile=2) — tests second GR rotation
#
FP8_TILE_CONFIGS = [
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=512, stride_b=512, use_swizzling=True),
]


# ---- Pytest tests ----

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGraTileAssignmentFP8GPU:

    @pytest.fixture(params=FP8_TILE_CONFIGS, ids=lambda c: c.label)
    def gra_env(self, request, tmp_path):
        """Generate FP8 graTileAssignment asm once per tile config."""
        cfg = request.param
        gra_asm, writer, tileInfoA, tileInfoB, kernel = generate_gra_asm_fp8(cfg)
        return SimpleNamespace(
            cfg=cfg,
            gra_asm=gra_asm,
            writer=writer,
            tileInfoA=tileInfoA,
            tileInfoB=tileInfoB,
            kernel=kernel,
            tmp_path=tmp_path,
        )

    def test_offset_a(self, gra_env):
        """Validate all sharedVgprGROffset vgprs for matrix A across all threads."""
        cfg = gra_env.cfg
        for idx, reg in enumerate(gra_env.tileInfoA.sharedVgprGROffset):
            results = export_register(gra_env.writer, gra_env.gra_asm, reg, False,
                                      cfg, gra_env.tmp_path, f"fp8_offsetA_v{reg}_{cfg.label}")
            for tid in range(NUM_THREADS):
                expected = compute_expected_offset_fp8(tid, cfg, gra_env.tileInfoA)
                assert results[tid] == expected[idx], \
                    f"[{cfg.label}] A offset[{idx}] v{reg} mismatch at tid={tid}: " \
                    f"got {results[tid]}, expected {expected[idx]}"

    def test_offset_b(self, gra_env):
        """Validate all sharedVgprGROffset vgprs for matrix B across all threads."""
        cfg = gra_env.cfg
        for idx, reg in enumerate(gra_env.tileInfoB.sharedVgprGROffset):
            results = export_register(gra_env.writer, gra_env.gra_asm, reg, False,
                                      cfg, gra_env.tmp_path, f"fp8_offsetB_v{reg}_{cfg.label}")
            for tid in range(NUM_THREADS):
                expected = compute_expected_offset_fp8(tid, cfg, gra_env.tileInfoB)
                assert results[tid] == expected[idx], \
                    f"[{cfg.label}] B offset[{idx}] v{reg} mismatch at tid={tid}: " \
                    f"got {results[tid]}, expected {expected[idx]}"

    def _test_subtile_registers(self, gra_env, tc):
        """Validate localSubtilesRegister values for matrix tc."""
        cfg      = gra_env.cfg
        tileInfo = gra_env.tileInfoA if tc == 'A' else gra_env.tileInfoB
        stride   = cfg.stride_a      if tc == 'A' else cfg.stride_b
        seen = set()
        for st in tileInfo.localSubtiles:
            regId = st.regListId
            if regId in seen:
                continue
            seen.add(regId)
            for reg in tileInfo.localSubtilesRegister[regId]:
                results  = export_register(gra_env.writer, gra_env.gra_asm, reg, st.useSgpr,
                                           cfg, gra_env.tmp_path,
                                           f"fp8_subtile{tc}_s{reg}_{cfg.label}")
                expected = compute_expected_subtile_fp8(regId, stride, tileInfo)
                assert results[0] == expected, \
                    f"[{cfg.label}] {tc} subtile s{reg} (regId={regId}): " \
                    f"got {results[0]}, expected {expected}"

    def test_subtile_registers_a(self, gra_env):
        self._test_subtile_registers(gra_env, 'A')

    def test_subtile_registers_b(self, gra_env):
        self._test_subtile_registers(gra_env, 'B')


if __name__ == "__main__":
    """Run standalone without pytest."""
    import argparse
    parser = argparse.ArgumentParser(description="GPU test for FP8 graTileAssignment")
    parser.add_argument("--grid",  action="store_true", help="Display offsets as 2D grid")
    parser.add_argument("--debug", action="store_true", help="Show expected grid and diffs (implies --grid)")
    args = parser.parse_args()
    if args.debug:
        args.grid = True

    for cfg in FP8_TILE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  FP8 Tile Config: {cfg.label}")
        print(f"{'='*60}")

        gra_asm, writer, tileInfoA, tileInfoB, kernel = generate_gra_asm_fp8(cfg)
        print(f"  A: loadRatioGR={tileInfoA.loadRatioGR}, numGRPerSubtile={tileInfoA.numGRPerSubtile}")
        print(f"  B: loadRatioGR={tileInfoB.loadRatioGR}, numGRPerSubtile={tileInfoB.numGRPerSubtile}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

            if HAS_HIP:
                for tc, tileInfo, stride in [("A", tileInfoA, cfg.stride_a),
                                              ("B", tileInfoB, cfg.stride_b)]:
                    for idx, reg in enumerate(tileInfo.sharedVgprGROffset):
                        results = export_register(writer, gra_asm, reg, False, cfg, tmp_path,
                                                  f"fp8_offset{tc}_v{reg}_{cfg.label}")

                        if args.grid:
                            print_offset_grid(f"FP8 Matrix {tc} GPU offset[{idx}] v{reg} ({cfg.label})",
                                              results, WAVESIZE, NUM_WAVES)
                            if args.debug:
                                expected_all = [compute_expected_offset_fp8(tid, cfg, tileInfo)[idx]
                                                for tid in range(NUM_THREADS)]
                                print_offset_grid(f"FP8 Matrix {tc} EXPECTED offset[{idx}] ({cfg.label})",
                                                  expected_all, WAVESIZE, NUM_WAVES)
                                mismatches = sum(1 for t in range(NUM_THREADS)
                                                 if results[t] != expected_all[t])
                                print(f"\n  Matrix {tc} offset[{idx}]: {mismatches} mismatches")

                        errors = sum(1 for tid in range(NUM_THREADS)
                                     if results[tid] != compute_expected_offset_fp8(tid, cfg, tileInfo)[idx])
                        print(f"  Matrix {tc} offset[{idx}] v{reg}: {NUM_THREADS} threads, {errors} errors")
            else:
                print("  HIP not available — assembly generated but not executed")
