#!/usr/bin/env python3
################################################################################
# GPU functional test for lraTileAssignment with FP8 (bpe=1, MatrixInstK=128)
#
# FP8 uses a block-swap LR swizzle to achieve zero LDS bank conflicts:
#   M_row   = lane % 16
#   K_group = lane // 16   (0-3)
#   swap_bit  = (M_row >> 1) & 1
#   finalColId = (K_group + 2 * (M_row >= 8)) % 4
#
#   Load 1: addr = M_row * depthUBytes + finalColId * 16 + swap_bit * 64
#   Load 2: addr = Load1_addr XOR 64  (flips the 64-byte block bit)
#
# Usage:
#   pytest test_lraTileAssignment_fp8.py -v -s
#   python test_lraTileAssignment_fp8.py --debug --grid
################################################################################

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
    init_rocisa,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
    generate_export_epilogue,
    print_offset_grid,
)
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from Tensile.Components.SubtileBasedKernel import TileInfo, lraTileAssignment

# ---- FP8-specific constants ----
BPE_FP8       = 1    # 1 byte per FP8 element
MATRIX_INST_K = 128  # v_mfma_f32_16x16x128_fp8_fp8

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
    mock = MagicMock()
    mock.numBytes.return_value = BPE_FP8
    mock.is8bitFloat.return_value = True
    return mock


def _create_kernel_fp8(cfg, mi_wave_group=None):
    """Create a minimal kernel dict for FP8 (MatrixInstK=128, bpe=1)."""
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
            "DataTypeA":       dtype,
            "DataTypeB":       dtype,
            "ComputeDataType": MagicMock(**{"numBytes.return_value": 4}),
            # Non-zero MXBlock triggers subtileShape=[1,1] for FP8
            "MXBlockA": 32,
            "MXBlockB": 32,
        },
    }


def create_writer_fp8(cfg, mi_wave_group=None):
    """Create a mock writer with FP8 kernel (MatrixInstK=128, bpe=1)."""
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

    readSize     = 2 * tileInfoA.subtileSize
    numASubtiles = tileInfoA.globalSubtileGrid[0] * tileInfoA.globalSubtileGrid[1]
    writer.ldsStartOffsetA = 0
    writer.ldsStartOffsetB = (
        (numASubtiles * tileInfoA.subtileSize + readSize - 1) // readSize
    ) * readSize

    return writer, kernel, tileInfoA, tileInfoB


# ---- LRA assembly generator ----

def generate_lra_asm_fp8(cfg):
    """Run lraTileAssignment for FP8 and return (lra_asm, writer, tileInfoA, tileInfoB, kernel)."""
    writer, kernel, tileInfoA, tileInfoB = create_writer_fp8(cfg)
    init_rocisa()

    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    prologue = generate_load_params(EXPORT_LOAD_PARAMS)
    module   = lraTileAssignment(writer, kernel)
    lra_asm  = f"{prologue}\n{module}"
    return lra_asm, writer, tileInfoA, tileInfoB, kernel


def export_register(writer, test_asm, export_reg, is_sgpr, cfg, tmp_path, label):
    """Assemble, run on GPU, return per-thread u32 results."""
    epilogue, allocated = generate_export_epilogue(writer, export_reg, is_sgpr)
    kernel_asm = generate_kernel_asm(f"{test_asm}\n{epilogue}", writer, EXPORT_ARGS)
    for v in allocated:
        writer.vgprPool.checkIn(v)

    raw = assemble_and_run(kernel_asm, tmp_path, label, NUM_THREADS * 4,
                           scalars=(cfg.stride_a, cfg.stride_b))
    return struct.unpack(f"{NUM_THREADS}I", raw)


# ---- Reference implementation ----

def compute_expected_lr_offset_fp8(thread_id, cfg, tileInfo, ldsStartOffsetB=None):
    """Python reference for FP8 lraTileAssignment block-swap swizzle.

    FP8 block-swap LR formula (zero LDS bank conflicts):
      M_row    = lane % 16
      K_group  = lane // 16   (0-3)
      swap_bit = (M_row >> 1) & 1            -- 1 for M_rows {2,3,6,7,10,11,14,15}
      finalColId = (K_group + 2*(M_row>=8)) % 4  -- shift K_group by 2 for M_rows 8-15

      Load 1: addr = M_row*depthUBytes + finalColId*16 + swap_bit*64
      Load 2: addr = Load1_addr XOR 64   (swap the 64-byte block)
    """
    bpe         = BPE_FP8
    depthUBytes = cfg.depth_u * bpe   # 128 for DU=128

    waveId = thread_id // WAVESIZE
    laneId = thread_id % WAVESIZE

    M_row   = laneId % 16
    K_group = laneId // 16

    swap_bit   = (M_row >> 1) & 1
    finalColId = (K_group + 2 * (M_row >> 3)) % 4   # >> 3 == >= 8

    rowOffset = M_row * depthUBytes

    # Load 1: col = finalColId + swap_bit * 4  (in 16-byte units)
    col_L1  = finalColId + swap_bit * 4
    offset0 = rowOffset + col_L1 * 16

    # Load 2: XOR 64 (flips the 64-byte block bit)
    offset1 = offset0 ^ 64

    offsets = [offset0, offset1]

    # Wave partitioning (same logic as FP4)
    partitionOffset = 0
    MT           = tileInfo.globalMMATileGrid[0] * tileInfo.mmaTileShape[0]
    subIterKBytes = depthUBytes

    if tileInfo.loadRatioGR >= 2.0:
        pass  # no partition needed
    elif tileInfo.loadRatioGR == 1.0:
        wavePartId      = (waveId & 1) if tileInfo.tc == 'A' else (waveId >> 1)
        partitionOffset = wavePartId * (MT * subIterKBytes // 2)
    elif tileInfo.loadRatioGR == 0.5:
        partitionOffset = waveId * (MT * subIterKBytes // 4)
    else:
        raise NotImplementedError(f"Unsupported loadRatioGR: {tileInfo.loadRatioGR}")

    # B matrix adds ldsStartOffsetB
    if tileInfo.tc == 'B':
        if ldsStartOffsetB is not None:
            partitionOffset += ldsStartOffsetB
        else:
            partitionOffset += cfg.mt_a * depthUBytes

    return [o + partitionOffset for o in offsets]


# ---- FP8 tile configs ----
#
# DU=128 = MatrixInstK → one subtile covers the full K dimension.
# 2x2 wave group: loadRatioGR=1.0 for A and B  → numLRPerSubtile=2
#
FP8_TILE_CONFIGS = [
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=512, stride_b=512),
]


# ---- Pytest tests ----

class TestLraTileAssignmentFP8Unit:
    """Non-GPU sanity checks for FP8 lraTileAssignment."""

    @pytest.fixture(params=FP8_TILE_CONFIGS, ids=lambda c: c.label)
    def lra_env(self, request):
        cfg = request.param
        lra_asm, writer, tileInfoA, tileInfoB, kernel = generate_lra_asm_fp8(cfg)
        return SimpleNamespace(cfg=cfg, lra_asm=lra_asm,
                               writer=writer, tileInfoA=tileInfoA,
                               tileInfoB=tileInfoB, kernel=kernel)

    def test_returns_valid_module(self, lra_env):
        assert lra_env.lra_asm is not None and len(lra_env.lra_asm) > 0

    def test_lr_offset_registers_allocated(self, lra_env):
        for tileInfo in [lra_env.tileInfoA, lra_env.tileInfoB]:
            assert hasattr(tileInfo, 'sharedVgprLROffset')
            assert len(tileInfo.sharedVgprLROffset) == tileInfo.numLRPerSubtile, \
                f"{tileInfo.tc}: expected {tileInfo.numLRPerSubtile} LR offset regs, " \
                f"got {len(tileInfo.sharedVgprLROffset)}"

    def test_num_lr_per_subtile_is_2(self, lra_env):
        """FP8 DU=128 always needs 2 ds_read_b128 (32 bytes / 16 bytes each)."""
        for tileInfo in [lra_env.tileInfoA, lra_env.tileInfoB]:
            assert tileInfo.numLRPerSubtile == 2, \
                f"{tileInfo.tc}: FP8 DU=128 expects numLRPerSubtile=2, got {tileInfo.numLRPerSubtile}"

    def test_tile_info_consistency(self, lra_env):
        for tileInfo in [lra_env.tileInfoA, lra_env.tileInfoB]:
            assert tileInfo.numLRPerSubtile >= 1
            assert tileInfo.numLRTotal >= 1
            assert tileInfo.loadRatioLR > 0
            assert tileInfo.bpe == BPE_FP8


@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestLraTileAssignmentFP8GPU:

    @pytest.fixture(params=FP8_TILE_CONFIGS, ids=lambda c: c.label)
    def lra_env(self, request, tmp_path):
        cfg = request.param
        lra_asm, writer, tileInfoA, tileInfoB, kernel = generate_lra_asm_fp8(cfg)
        return SimpleNamespace(cfg=cfg, lra_asm=lra_asm,
                               writer=writer, tileInfoA=tileInfoA,
                               tileInfoB=tileInfoB, kernel=kernel,
                               tmp_path=tmp_path)

    def _check_offsets(self, lra_env, tc):
        cfg      = lra_env.cfg
        tileInfo = lra_env.tileInfoA if tc == 'A' else lra_env.tileInfoB
        for idx, reg in enumerate(tileInfo.sharedVgprLROffset):
            results = export_register(lra_env.writer, lra_env.lra_asm, reg, False,
                                      cfg, lra_env.tmp_path,
                                      f"fp8_lr_offset{tc}_{idx}_v{reg}_{cfg.label}")
            for tid in range(NUM_THREADS):
                expected = compute_expected_lr_offset_fp8(
                    tid, cfg, tileInfo, lra_env.writer.ldsStartOffsetB)
                assert results[tid] == expected[idx], \
                    f"[{cfg.label}] {tc} LR offset[{idx}] v{reg} mismatch at tid={tid}: " \
                    f"got {results[tid]}, expected {expected[idx]}"

    def test_offset_a(self, lra_env):
        """Validate sharedVgprLROffset[0] and [1] for matrix A across all threads."""
        self._check_offsets(lra_env, 'A')

    def test_offset_b(self, lra_env):
        """Validate sharedVgprLROffset[0] and [1] for matrix B across all threads."""
        self._check_offsets(lra_env, 'B')


# ---- Standalone runner ----

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU test for FP8 lraTileAssignment")
    parser.add_argument("--grid",  action="store_true", help="Display offsets as 2D grid")
    parser.add_argument("--debug", action="store_true", help="Show expected grid and diffs (implies --grid)")
    args = parser.parse_args()
    if args.debug:
        args.grid = True

    for cfg in FP8_TILE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  FP8 LR Config: {cfg.label}")
        print(f"{'='*60}")

        lra_asm, writer, tileInfoA, tileInfoB, kernel = generate_lra_asm_fp8(cfg)
        print(f"  A: loadRatioGR={tileInfoA.loadRatioGR}, numLRPerSubtile={tileInfoA.numLRPerSubtile}")
        print(f"  B: loadRatioGR={tileInfoB.loadRatioGR}, numLRPerSubtile={tileInfoB.numLRPerSubtile}")
        print(f"  ldsStartOffsetB={writer.ldsStartOffsetB}")

        if args.debug:
            print("\n--- Generated Assembly ---")
            for line in lra_asm.split('\n'):
                print(line)
            print("--- End ---\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

            if not HAS_HIP:
                print("  HIP not available — assembly generated but not executed")
                continue

            for tc, tileInfo in [("A", tileInfoA), ("B", tileInfoB)]:
                for idx, reg in enumerate(tileInfo.sharedVgprLROffset):
                    results = export_register(writer, lra_asm, reg, False, cfg, tmp_path,
                                              f"fp8_lr_offset{tc}_{idx}_v{reg}_{cfg.label}")

                    if args.grid:
                        print_offset_grid(
                            f"FP8 Matrix {tc} LR GPU offset[{idx}] v{reg} ({cfg.label})",
                            results, WAVESIZE, NUM_WAVES)

                        if args.debug:
                            expected_all = [
                                compute_expected_lr_offset_fp8(
                                    tid, cfg, tileInfo, writer.ldsStartOffsetB)[idx]
                                for tid in range(NUM_THREADS)
                            ]
                            print_offset_grid(
                                f"FP8 Matrix {tc} LR EXPECTED offset[{idx}] ({cfg.label})",
                                expected_all, WAVESIZE, NUM_WAVES)

                            mismatches = sum(
                                1 for t in range(NUM_THREADS)
                                if results[t] != expected_all[t])
                            if mismatches:
                                print(f"\n  DIFF: {mismatches} mismatches")
                                for w in range(NUM_WAVES):
                                    row = []
                                    for lane in range(WAVESIZE):
                                        tid = w * WAVESIZE + lane
                                        if results[tid] != expected_all[tid]:
                                            row.append(f"t{tid}:{results[tid]}!={expected_all[tid]}")
                                    if row:
                                        print(f"  w{w}: {' '.join(row)}")
                            else:
                                print(f"\n  All match.")

                    errors = sum(
                        1 for tid in range(NUM_THREADS)
                        if results[tid] != compute_expected_lr_offset_fp8(
                            tid, cfg, tileInfo, writer.ldsStartOffsetB)[idx])
                    status = "PASS" if errors == 0 else f"FAIL ({errors} errors)"
                    print(f"  Matrix {tc} LR offset[{idx}] v{reg}: {status}")
