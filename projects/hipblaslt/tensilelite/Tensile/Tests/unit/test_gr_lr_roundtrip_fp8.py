#!/usr/bin/env python3
################################################################################
# End-to-end GPU roundtrip test: GR -> LDS -> LR for FP8
#
# Uses the production globalReadDoSubtile / localReadDoSubtile code paths with
# FP8 (bpe=1, MatrixInstK=128, block-swap swizzle).
#
# After the roundtrip, each vgprTile (4 VGPRs, 16 bytes per lane) should hold
# the 16 FP8 elements that the MFMA instruction expects for that (M_row, K_range):
#
#   lrIdx=0 tiles (Load1): K_lo = K_group*16 .. K_group*16+15
#   lrIdx=1 tiles (Load2): K_hi = K_group*16+64 .. K_group*16+79
#
#   where K_group = laneId // 16  (0-3)
#
# The block-swap GR swizzle and LR de-swizzle cancel out, so the net result is
# the same data layout as the natural (no-swizzle) order.
#
# GR swizzle (SubtileBasedKernel._grSwizzleColIds / _grComputeAllOffsets):
#   colId ^= 4 for odd ldsRowId        (block swap: interchange K_lo/K_hi blocks)
#   rotated_K_group = (K_group + 2*(waveId&1)) % 4
#
# LR swizzle (SubtileBasedKernel._lrComputeAllOffsets):
#   swap_bit   = (M_row >> 1) & 1
#   finalColId = (K_group + 2*(M_row >= 8)) % 4
#   Load1 = M_row*128 + finalColId*16 + swap_bit*64
#   Load2 = Load1 XOR 64               (picks the other K block)
#
# Usage:
#   pytest test_gr_lr_roundtrip_fp8.py -v -s
#   python test_gr_lr_roundtrip_fp8.py --debug --wave all
################################################################################

import os
import sys
import tempfile

import pytest
import numpy as np

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    WAVESIZE, NUM_THREADS,
    init_rocisa,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
)

from Tensile.Components.SubtileBasedKernel import (
    graTileAssignment,
    lraTileAssignment,
    globalReadDTLInitCommonSgpr,
    globalReadDoSubtile,
    localReadDoSubtile,
)
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SMovB32, SMovB64, SWaitCnt, SBarrier

# Re-use FP8 writer/kernel helpers from the GRA test
from test_graTileAssignment_fp8 import create_writer_fp8

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BPE_FP8       = 1    # 1 byte per FP8 element
MATRIX_INST_K = 128  # v_mfma_f32_16x16x128_fp8_fp8
TILE_SIZE_BYTES = WAVESIZE * 16  # per tile: 64 lanes × 4 VGPRs × 4 bytes = 1024 bytes

# ---------------------------------------------------------------------------
# Test configurations  (depth_u=128 = MatrixInstK for FP8)
# ---------------------------------------------------------------------------
CONFIGS = [
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=128, stride_b=128),
]

# ---------------------------------------------------------------------------
# Assembly generation
# ---------------------------------------------------------------------------

def generate_srd_setup_fp8():
    """Set up SRD buffer descriptors for FP8 (u8 element type)."""
    module = Module("SRD setup FP8")
    module.add(SMovB64(dst=sgpr("SrdA+0", 2), src=sgpr(4, 2), comment="SrdA base = input_A_ptr"))
    module.add(SMovB32(dst=sgpr("SrdA+2"), src="0xFFFFFFFF",   comment="SrdA NumRecords = max"))
    module.add(SMovB32(dst=sgpr("SrdA+3"), src="0x20000",      comment="SrdA OOB_SELECT=2"))
    module.add(SMovB64(dst=sgpr("SrdB+0", 2), src=sgpr(6, 2), comment="SrdB base = input_B_ptr"))
    module.add(SMovB32(dst=sgpr("SrdB+2"), src="0xFFFFFFFF",   comment="SrdB NumRecords = max"))
    module.add(SMovB32(dst=sgpr("SrdB+3"), src="0x20000",      comment="SrdB OOB_SELECT=2"))
    return module


def generate_export_asm_fp8(wave_id, tileInfoA, tileInfoB):
    """Generate assembly to export all vgprTile registers from a selected wave.

    Each FP8 vgprTile has 4 VGPRs = 16 bytes per lane.
    Output layout: [A tiles][B tiles], each tile = WAVESIZE * 16 bytes = 1024 bytes.
    """
    lines = []
    lines.append(f"  // ---- Wave-gated FP8 export (wave {wave_id}) ----")

    all_tile_vgprs = set()
    for t in tileInfoA.vgprTiles:
        for v in t:
            all_tile_vgprs.add(v)
    for t in tileInfoB.vgprTiles:
        for v in t:
            all_tile_vgprs.add(v)
    for v in tileInfoA.sharedVgprGROffset:
        all_tile_vgprs.add(v)
    for v in tileInfoB.sharedVgprGROffset:
        all_tile_vgprs.add(v)
    for v in tileInfoA.sharedVgprLROffset:
        all_tile_vgprs.add(v)
    for v in tileInfoB.sharedVgprLROffset:
        all_tile_vgprs.add(v)

    next_v = max(all_tile_vgprs | {0}) + 1
    tmp = next_v; next_v += 1
    if next_v % 2 != 0:
        next_v += 1
    addr_lo = next_v; next_v += 1
    addr_hi = next_v; next_v += 1

    lines.append(f"  v_lshrrev_b32 v{tmp}, 6, v0                // waveId")
    lines.append(f"  v_cmp_eq_u32 vcc, {wave_id}, v{tmp}")
    lines.append(f"  s_and_saveexec_b64 s[2:3], vcc              // gate to wave {wave_id}")
    lines.append(f"  v_and_b32 v{tmp}, 0x3F, v0                  // laneId")

    tile_index = 0
    all_tiles = list(tileInfoA.vgprTiles) + list(tileInfoB.vgprTiles)

    for tile in all_tiles:
        vgpr_start = tile.regList.regValues[0]
        num_regs = len(tile.regList.regValues)
        assert num_regs == 4, f"FP8 roundtrip: expected 4 VGPRs per tile, got {num_regs}"

        base_offset = tile_index * TILE_SIZE_BYTES
        lines.append(f"  // Export FP8 tile {tile_index}: v[{vgpr_start}:{vgpr_start+3}]")
        lines.append(f"  v_lshlrev_b32 v{addr_lo}, 4, v{tmp}       // laneId * 16")
        if base_offset > 0:
            lines.append(f"  v_add_u32 v{addr_lo}, {base_offset}, v{addr_lo}  // + tile base")
        lines.append(f"  v_mov_b32 v{addr_hi}, s9                   // output_ptr hi")
        lines.append(f"  v_add_co_u32 v{addr_lo}, vcc, s8, v{addr_lo}")
        lines.append(f"  v_addc_co_u32 v{addr_hi}, vcc, v{addr_hi}, 0, vcc")
        lines.append(f"  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{vgpr_start}:{vgpr_start+3}]")
        lines.append(f"  s_waitcnt vmcnt(0)")
        tile_index += 1

    lines.append(f"  s_or_b64 exec, exec, s[2:3]                // restore exec")
    return "\n".join(lines), next_v


def generate_roundtrip_kernel_fp8(cfg, wave_id=0):
    """Generate a complete FP8 GR->LDS->LR kernel using production code paths."""
    init_rocisa()

    writer, kernel, tileInfoA, tileInfoB = create_writer_fp8(cfg)

    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    writer.sgprs["SrdA"] = writer.sgprPool.checkOutAligned(4, 4, "SrdA", preventOverflow=False)
    writer.sgprs["SrdB"] = writer.sgprPool.checkOutAligned(4, 4, "SrdB", preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrA"] = writer.sgprPool.checkOut(1, "LocalWriteBaseAddrA", preventOverflow=False)
    writer.sgprs["LocalWriteDTLOffsetA"] = writer.sgprPool.checkOut(1, "LocalWriteDTLOffsetA", preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrB"] = writer.sgprPool.checkOut(1, "LocalWriteBaseAddrB", preventOverflow=False)
    writer.sgprs["LocalWriteDTLOffsetB"] = writer.sgprPool.checkOut(1, "LocalWriteDTLOffsetB", preventOverflow=False)
    writer.sgprs["SwapA"] = writer.sgprPool.checkOut(1, "SwapA", preventOverflow=False)
    writer.sgprs["SwapB"] = writer.sgprPool.checkOut(1, "SwapB", preventOverflow=False)

    readSize = 2 * tileInfoA.subtileSize
    numASubtiles = tileInfoA.globalSubtileGrid[0] * tileInfoA.globalSubtileGrid[1]
    numBSubtiles = tileInfoB.globalSubtileGrid[0] * tileInfoB.globalSubtileGrid[1]
    sizeA = ((numASubtiles * tileInfoA.subtileSize + readSize - 1) // readSize) * readSize
    sizeB = ((numBSubtiles * tileInfoB.subtileSize + readSize - 1) // readSize) * readSize
    lds_size = sizeA + sizeB
    writer.ldsTotalSize = lds_size

    tileInfoA.allocVgprTileRegisters(writer, kernel)
    tileInfoB.allocVgprTileRegisters(writer, kernel)

    gra_module = graTileAssignment(writer, kernel, useSwizzling=True)
    lra_module = lraTileAssignment(writer, kernel)
    dtl_module = globalReadDTLInitCommonSgpr(writer, kernel)
    gr_a_module = globalReadDoSubtile('A', writer, kernel)
    gr_b_module = globalReadDoSubtile('B', writer, kernel)

    wait_gr = SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1)
    barrier = SBarrier()

    lr_a_module = localReadDoSubtile('A', writer, kernel)
    lr_b_module = localReadDoSubtile('B', writer, kernel)
    wait_lr = SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1)

    export_asm, _next_v = generate_export_asm_fp8(wave_id, tileInfoA, tileInfoB)

    prologue = generate_load_params([
        (4, 4, 0x00, "input_A_ptr + input_B_ptr"),
        (8, 4, 0x10, "output_ptr + strideA + strideB"),
    ])
    srd_module = generate_srd_setup_fp8()

    inner_asm = "\n".join([
        str(prologue),
        str(srd_module),
        str(gra_module),
        str(lra_module),
        str(dtl_module),
        str(gr_a_module),
        str(gr_b_module),
        str(wait_gr),
        str(barrier),
        str(lr_a_module),
        str(lr_b_module),
        str(wait_lr),
        str(export_asm),
    ])

    args = (
        ("input_A_ptr", 8, "global_buffer", "u8"),
        ("input_B_ptr", 8, "global_buffer", "u8"),
        ("output_ptr",  8, "global_buffer", "u32"),
        ("strideA",     4, "by_value",      "u32"),
        ("strideB",     4, "by_value",      "u32"),
    )

    kernel_asm = generate_kernel_asm(inner_asm, writer, args, lds_size)

    num_tiles_a = len(tileInfoA.vgprTiles)
    num_tiles_b = len(tileInfoB.vgprTiles)
    output_size = (num_tiles_a + num_tiles_b) * TILE_SIZE_BYTES

    return kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size


# ---------------------------------------------------------------------------
# Expected output computation
# ---------------------------------------------------------------------------

def _build_tile_to_lr(tileInfo):
    """Build map from vgprTile index to (mmaId0, lrIdx).

    For FP8 with subtileShape=[1,1] and numLRPerSubtile=2:
      - localReadMap[0] = lrIdx=0 (K_lo block)
      - localReadMap[1] = lrIdx=1 (K_hi block)
    """
    tile_to_lr = {}
    for linearId, subtile in enumerate(tileInfo.localSubtiles):
        sId0, sId1 = tileInfo.getLocalSubtileIdFromLinearId(linearId)
        mmaId0 = sId0 * tileInfo.subtileShape[0]  # row partition index
        for lrIdx, tileIdx in enumerate(subtile.localReadMap):
            tile_to_lr[tileIdx] = (mmaId0, lrIdx)
    return tile_to_lr


def compute_expected_output_fp8(cfg, tileInfoA, tileInfoB, kernel, input_A, input_B, wave_id):
    """Compute expected vgprTile contents after GR->LDS->LR roundtrip for FP8.

    After the block-swap GR+LR swizzles cancel out, each vgprTile holds 16 FP8
    elements corresponding to a specific (M_row, K_range) in the input matrix:

      lrIdx=0 (Load1, K_lo): row M_row, K[K_group*16 : K_group*16+16]
      lrIdx=1 (Load2, K_hi): row M_row, K[K_group*16+64 : K_group*16+80]

    where M_row = laneId % 16, K_group = laneId // 16.

    Returns a list of numpy arrays (uint8), one per vgprTile (A first, then B).
    Each array is WAVESIZE * 16 bytes (16 FP8 elements per lane).
    """
    mi_wave_group = kernel["MIWaveGroup"]
    results = []

    for tc, tileInfo, input_data in [('A', tileInfoA, input_A), ('B', tileInfoB, input_B)]:
        stride = cfg.stride_a if tc == 'A' else cfg.stride_b
        input_matrix = input_data.reshape(-1, stride)

        # Wave offset in the tile M-dimension
        wave_offset_factor = wave_id % mi_wave_group[0] if tc == 'A' else wave_id // mi_wave_group[0]
        wave_row_offset = wave_offset_factor * tileInfo.localMMATileGrid[0] * 16

        tile_to_lr = _build_tile_to_lr(tileInfo)

        for tileIdx in range(len(tileInfo.vgprTiles)):
            tile_data = np.zeros(WAVESIZE * 16, dtype=np.uint8)

            if tileIdx not in tile_to_lr:
                results.append(tile_data)
                continue

            mmaId0, lrIdx = tile_to_lr[tileIdx]

            for lane in range(WAVESIZE):
                M_row   = lane % 16
                K_group = lane // 16

                row_in_input = wave_row_offset + mmaId0 * 16 + M_row

                # K_lo block for lrIdx=0, K_hi block for lrIdx=1
                col_in_input = K_group * 16 + lrIdx * 64

                if (row_in_input < input_matrix.shape[0] and
                        col_in_input + 16 <= input_matrix.shape[1]):
                    tile_data[lane * 16 : lane * 16 + 16] = (
                        input_matrix[row_in_input, col_in_input : col_in_input + 16])

            results.append(tile_data)

    return results


def compare_tiles_fp8(actual_bytes, expected_tiles, tileInfoA, tileInfoB, wave_id, debug=False):
    """Compare GPU output against expected FP8 tile data. Returns error count."""
    errors = 0
    num_tiles_a = len(tileInfoA.vgprTiles)
    num_tiles_b = len(tileInfoB.vgprTiles)
    total_tiles = num_tiles_a + num_tiles_b

    for tile_idx in range(total_tiles):
        tc = 'A' if tile_idx < num_tiles_a else 'B'
        local_idx = tile_idx if tile_idx < num_tiles_a else tile_idx - num_tiles_a

        offset = tile_idx * TILE_SIZE_BYTES
        actual = np.frombuffer(actual_bytes[offset : offset + TILE_SIZE_BYTES], dtype=np.uint8)
        expected = expected_tiles[tile_idx]

        if not np.array_equal(actual, expected):
            errors += 1
            if errors <= 8 or debug:
                for lane in range(WAVESIZE):
                    a_sl = actual[lane * 16 : lane * 16 + 16]
                    e_sl = expected[lane * 16 : lane * 16 + 16]
                    if not np.array_equal(a_sl, e_sl):
                        print(f"  MISMATCH wave {wave_id} {tc} tile {local_idx} lane {lane}:")
                        print(f"    expected: {e_sl.tolist()}")
                        print(f"    actual:   {a_sl.tolist()}")
                        if not debug:
                            break

    return errors


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGrLrRoundtripFP8:

    @pytest.fixture(params=CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    @pytest.fixture(params=[0, 1, 2, 3], ids=lambda w: f"wave{w}")
    def wave_id(self, request):
        return request.param

    def test_gr_lr_roundtrip_fp8(self, cfg, wave_id, tmp_path):
        """Verify FP8 GR -> LDS -> LR roundtrip using production code paths."""
        sys.stdout.flush()

        kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size = \
            generate_roundtrip_kernel_fp8(cfg, wave_id=wave_id)

        # Sequential FP8 values so each element is uniquely identifiable
        input_A = np.arange(1, cfg.mt_a * cfg.stride_a + 1, dtype=np.uint8)
        input_B = np.arange(129, cfg.mt_b * cfg.stride_b + 129, dtype=np.uint8)

        label = f"fp8_roundtrip_{cfg.label}_wave{wave_id}"
        output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                        inputs=(input_A, input_B),
                                        scalars=(cfg.stride_a, cfg.stride_b),
                                        lds_size=lds_size)

        expected_tiles = compute_expected_output_fp8(cfg, tileInfoA, tileInfoB, kernel,
                                                     input_A, input_B, wave_id)
        errors = compare_tiles_fp8(output_bytes, expected_tiles, tileInfoA, tileInfoB,
                                   wave_id, debug=False)

        assert errors == 0, \
            f"Wave {wave_id}, config {cfg.label}: {errors} tile mismatches"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FP8 GR/LR roundtrip GPU test")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed mismatch output and ASM")
    parser.add_argument("--wave", default="all",
                        help="Which wave to test: 0-3 or 'all' (default: all)")
    parser.add_argument("--config", type=int, default=None,
                        help="Config index to test (default: all)")
    args = parser.parse_args()

    if not HAS_HIP:
        print("HIP not available — cannot run GPU test")
        sys.exit(1)

    wave_list = [0, 1, 2, 3] if args.wave == "all" else [int(args.wave)]
    config_list = CONFIGS if args.config is None else [CONFIGS[args.config]]

    total_errors = 0
    total_tests = 0

    for cfg_idx, cfg in enumerate(config_list):
        print(f"\n{'='*60}")
        print(f"FP8 Config: {cfg.label}")
        print(f"  mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, depth_u={cfg.depth_u}")
        print(f"  stride_a={cfg.stride_a}, stride_b={cfg.stride_b}")

        for wave_id in wave_list:
            total_tests += 1
            print(f"\n  --- Wave {wave_id} ---")

            kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size, lds_size = \
                generate_roundtrip_kernel_fp8(cfg, wave_id=wave_id)

            num_tiles_a = len(tileInfoA.vgprTiles)
            num_tiles_b = len(tileInfoB.vgprTiles)
            print(f"  A tiles: {num_tiles_a}, B tiles: {num_tiles_b}, output: {output_size} bytes")
            print(f"  MIWaveGroup: {kernel['MIWaveGroup']}")
            print(f"  LDS size: {lds_size} bytes")

            if args.debug:
                print(f"\n--- Kernel ASM ---\n{kernel_asm}\n--- End ---\n")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

                input_A = np.arange(1, cfg.mt_a * cfg.stride_a + 1, dtype=np.uint8)
                input_B = np.arange(129, cfg.mt_b * cfg.stride_b + 129, dtype=np.uint8)

                label = f"fp8_roundtrip_{cfg.label}_wave{wave_id}"
                output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                                inputs=(input_A, input_B),
                                                scalars=(cfg.stride_a, cfg.stride_b),
                                                lds_size=lds_size)

                expected_tiles = compute_expected_output_fp8(
                    cfg, tileInfoA, tileInfoB, kernel, input_A, input_B, wave_id)
                errors = compare_tiles_fp8(output_bytes, expected_tiles,
                                           tileInfoA, tileInfoB, wave_id, debug=args.debug)

                if errors == 0:
                    print(f"  PASS")
                else:
                    print(f"  FAIL: {errors} tile mismatches")
                    total_errors += errors

    print(f"\n{'='*60}")
    print(f"Result: {total_tests} tests, {total_errors} errors")
    if total_errors > 0:
        print("FAILED")
        sys.exit(1)
    else:
        print("PASSED")
