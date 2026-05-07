#!/usr/bin/env python3
################################################################################
# End-to-end GPU MFMA test for FP8:
#   GR -> LDS -> LR -> v_mfma_f32_16x16x128_f8f6f4 -> export AGPRs -> verify
#
# Uses OCP FP8 E4M3FN format (cbsz:0 blgp:0) for both A and B operands.
#
# MFMA instruction:
#   v_mfma_f32_16x16x128_f8f6f4 a[acc:acc+3], v[B:B+7], v[A:A+7], a[acc:acc+3]
#     cbsz:0 blgp:0
#
# The 8-VGPR operand for A (or B) consists of two consecutive 4-VGPR tiles:
#   v[a_lo:a_lo+3]  = K_lo tile (FP8 elements at K[K_group*16 : K_group*16+16])
#   v[a_lo+4:a_lo+7] = K_hi tile (FP8 elements at K[K_group*16+64 : K_group*16+80])
#
# Output layout per lane:
#   Lane l: M_row = l % 16, N_col_group = l // 16
#   4 float32 accumulators: C[M_row, N_col_group*4 : N_col_group*4+4]
#
# Usage:
#   pytest test_mfma_fp8.py -v -s
#   python test_mfma_fp8.py --debug --wave 0
################################################################################

import os
import sys
import struct
import tempfile

import pytest
import numpy as np

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    WAVESIZE, NUM_WAVES, NUM_THREADS,
    init_rocisa,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
)
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SMovB32, SMovB64, SWaitCnt, SBarrier

from Tensile.Components.SubtileBasedKernel import (
    graTileAssignment,
    lraTileAssignment,
    globalReadDTLInitCommonSgpr,
    globalReadDoSubtile,
    localReadDoSubtile,
)

# Re-use FP8 writer/kernel helpers from the GRA test
from test_graTileAssignment_fp8 import create_writer_fp8

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BPE_FP8 = 1           # 1 byte per OCP FP8 element
MATRIX_INST_K = 128   # v_mfma_f32_16x16x128_f8f6f4
ACC_SIZE = 4          # 4 AGPRs per MFMA output tile (4 float32 per lane)
MFMA_WAIT_NOPS = 16   # s_nop 7 repeated 16 times (128 cycles) after MFMAs

# OCP FP8 E4M3FN byte encodings for integer values 1..8
# Verified: bias=7, value v → byte 0x(sign_exp_mant)
FP8_FN_TABLE = {
    1: 0x38,  # 0 0111 000 = 1.000 × 2^0 = 1.0
    2: 0x40,  # 0 1000 000 = 1.000 × 2^1 = 2.0
    3: 0x44,  # 0 1000 100 = 1.100 × 2^1 = 3.0
    4: 0x48,  # 0 1001 000 = 1.000 × 2^2 = 4.0
    5: 0x4A,  # 0 1001 010 = 1.010 × 2^2 = 5.0
    6: 0x4C,  # 0 1001 100 = 1.100 × 2^2 = 6.0
    7: 0x4E,  # 0 1001 110 = 1.110 × 2^2 = 7.0
    8: 0x50,  # 0 1010 000 = 1.000 × 2^3 = 8.0
}

# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    # 256×256, 2×2 wave group: 64 MFMAs per wave (8 A × 8 B)
    TileConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=128, stride_b=128),
]

# ---------------------------------------------------------------------------
# FP8 E4M3FN helpers
# ---------------------------------------------------------------------------

def fp8_e4m3fn_to_float32(byte_val) -> float:
    """Decode OCP FP8 E4M3FN byte to Python float.

    Format: sign(1) | exponent(4, bias=7) | mantissa(3)
    Special: 0x7F and 0xFF are NaN; no infinity.
    """
    byte_val = int(byte_val)  # avoid numpy.uint8 wraparound arithmetic
    if byte_val & 0x7F == 0x7F:
        return float('nan')
    sign = -1.0 if (byte_val >> 7) else 1.0
    exp_stored = (byte_val >> 3) & 0xF
    mantissa = byte_val & 0x7
    if exp_stored == 0:
        return sign * (mantissa / 8.0) * (2.0 ** -6)
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exp_stored - 7))


def make_fp8_input(num_rows: int, num_cols: int, seed: int = 42) -> np.ndarray:
    """Build a random uint8 FP8 E4M3FN matrix.

    NaN bytes (0x7F, 0xFF) are replaced with 0x00 to avoid NaN propagation
    in the expected float32 matmul.
    """
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=num_rows * num_cols, dtype=np.uint8)
    data[data == 0x7F] = 0x00
    data[data == 0xFF] = 0x00
    return data


# ---------------------------------------------------------------------------
# SRD setup (identical to roundtrip FP8 test)
# ---------------------------------------------------------------------------

def generate_srd_setup_fp8():
    module = Module("SRD setup FP8")
    module.add(SMovB64(dst=sgpr("SrdA+0", 2), src=sgpr(4, 2),
                       comment="SrdA base = input_A_ptr"))
    module.add(SMovB32(dst=sgpr("SrdA+2"), src="0xFFFFFFFF",
                       comment="SrdA NumRecords = max"))
    module.add(SMovB32(dst=sgpr("SrdA+3"), src="0x20000",
                       comment="SrdA OOB_SELECT=2"))
    module.add(SMovB64(dst=sgpr("SrdB+0", 2), src=sgpr(6, 2),
                       comment="SrdB base = input_B_ptr"))
    module.add(SMovB32(dst=sgpr("SrdB+2"), src="0xFFFFFFFF",
                       comment="SrdB NumRecords = max"))
    module.add(SMovB32(dst=sgpr("SrdB+3"), src="0x20000",
                       comment="SrdB OOB_SELECT=2"))
    return module


# ---------------------------------------------------------------------------
# MFMA assembly generation
# ---------------------------------------------------------------------------

def _collect_tile_vgprs(tileInfoA, tileInfoB):
    """Return the set of all VGPR indices used by A/B tile and offset registers."""
    used = set()
    for t in tileInfoA.vgprTiles:
        used.update(t.regList.regValues)
    for t in tileInfoB.vgprTiles:
        used.update(t.regList.regValues)
    used.update(tileInfoA.sharedVgprGROffset)
    used.update(tileInfoB.sharedVgprGROffset)
    used.update(tileInfoA.sharedVgprLROffset)
    used.update(tileInfoB.sharedVgprLROffset)
    return used


def generate_mfma_pairs(tileInfoA, tileInfoB, writer):
    """Enumerate MFMA pairs and allocate 4-AGPR accumulators.

    Iterates in the same (mma1 × mma0) order as emitMfmaCode so that
    the pair index matches the output buffer layout.

    Returns:
        list of dicts: {mma0, mma1, a_lo_vgpr, b_lo_vgpr, acc_start}
    """
    pairs = []
    for mma1 in range(tileInfoB.localMMATileGrid[0]):
        for mma0 in range(tileInfoA.localMMATileGrid[0]):
            # A: find which subtile owns mma tile (mma0, mmak=0)
            aSId0, aSId1 = tileInfoA.getLocalSubtileIdFromMMATile(mma0, 0)
            a_lin = tileInfoA.getLocalSubtileLinearId(aSId0, aSId1)
            subtileA = tileInfoA.localSubtiles[a_lin]
            a_lo_idx = subtileA.localReadMap[0]   # K_lo tile
            a_hi_idx = subtileA.localReadMap[1]   # K_hi tile
            a_lo = tileInfoA.vgprTiles[a_lo_idx].regList.regValues[0]
            a_hi = tileInfoA.vgprTiles[a_hi_idx].regList.regValues[0]
            assert a_hi == a_lo + 4, (
                f"A K_lo/K_hi tiles not consecutive at mma0={mma0}: "
                f"a_lo={a_lo}, a_hi={a_hi}"
            )

            # B: find which subtile owns mma tile (mma1, mmak=0)
            bSId0, bSId1 = tileInfoB.getLocalSubtileIdFromMMATile(mma1, 0)
            b_lin = tileInfoB.getLocalSubtileLinearId(bSId0, bSId1)
            subtileB = tileInfoB.localSubtiles[b_lin]
            b_lo_idx = subtileB.localReadMap[0]
            b_hi_idx = subtileB.localReadMap[1]
            b_lo = tileInfoB.vgprTiles[b_lo_idx].regList.regValues[0]
            b_hi = tileInfoB.vgprTiles[b_hi_idx].regList.regValues[0]
            assert b_hi == b_lo + 4, (
                f"B K_lo/K_hi tiles not consecutive at mma1={mma1}: "
                f"b_lo={b_lo}, b_hi={b_hi}"
            )

            acc = writer.agprPool.checkOutAligned(4, 4, "mfma_acc",
                                                  preventOverflow=False)
            pairs.append({
                'mma0': mma0, 'mma1': mma1,
                'a_lo': a_lo,
                'b_lo': b_lo,
                'acc': acc,
            })
    return pairs


def generate_mfma_asm(mfma_pairs):
    """Generate AGPR zero + MFMA instruction assembly for all pairs.

    Returns assembly string (no export).
    """
    lines = ["  // === Zero all MFMA accumulators ==="]
    for p in mfma_pairs:
        acc = p['acc']
        for k in range(ACC_SIZE):
            lines.append(f"  v_accvgpr_write a{acc + k}, 0")
    # Wait for AGPR writes to complete before first MFMA reads them as C input
    lines.append(f"  s_nop 7  // wait for accvgpr_write to settle")

    lines.append("  // === FP8 MFMA: v_mfma_f32_16x16x128_f8f6f4 cbsz:0 blgp:0 ===")
    lines.append("  // SRC0=B (N-dim, 8 VGPRs), SRC1=A (M-dim, 8 VGPRs), acc=4 AGPRs")
    for p in mfma_pairs:
        a = p['a_lo']
        b = p['b_lo']
        acc = p['acc']
        lines.append(
            f"  v_mfma_f32_16x16x128_f8f6f4 a[{acc}:{acc + 3}], "
            f"v[{b}:{b + 7}], v[{a}:{a + 7}], a[{acc}:{acc + 3}] "
            f"cbsz:0 blgp:0  "
            f"// C[mma0={p['mma0']}, mma1={p['mma1']}] += A @ B.T"
        )

    # Wait for all MFMAs to write their accumulators before reading them.
    # s_nop 7 = 8 cycles; repeat MFMA_WAIT_NOPS times.
    lines.append(f"  // Wait {MFMA_WAIT_NOPS} × 8 = {MFMA_WAIT_NOPS * 8} cycles for MFMA results")
    lines.extend([f"  s_nop 7"] * MFMA_WAIT_NOPS)
    return "\n".join(lines)


def generate_mfma_export_asm(mfma_pairs, tileInfoA, tileInfoB):
    """Generate assembly to export AGPR results from all waves in parallel.

    Each wave writes to its own region:
      output_ptr + wave_id * wave_region_size + pair_base + laneId * 16

    wave_region_size = num_pairs * WAVESIZE * ACC_SIZE * 4 bytes

    Returns: (asm_string, next_free_vgpr)
    """
    wave_region_size = len(mfma_pairs) * WAVESIZE * ACC_SIZE * 4

    # Determine first free VGPR above all tile and offset VGPRs
    used = _collect_tile_vgprs(tileInfoA, tileInfoB)
    next_v = max(used | {0}) + 1

    wave_off = next_v; next_v += 1
    lane     = next_v; next_v += 1
    # Align addr pair to 2
    if next_v % 2 != 0:
        next_v += 1
    addr_lo = next_v; next_v += 1
    addr_hi = next_v; next_v += 1
    # 4 VGPRs to receive v_accvgpr_read results (aligned to 4)
    while next_v % 4 != 0:
        next_v += 1
    r0, r1, r2, r3 = next_v, next_v + 1, next_v + 2, next_v + 3
    next_v += 4

    import math
    assert wave_region_size & (wave_region_size - 1) == 0, "wave_region_size must be power of 2"
    shift = int(math.log2(wave_region_size))

    lines = ["  // ---- MFMA export: all waves write in parallel ----"]
    lines.append(f"  v_lshrrev_b32 v{wave_off}, 6, v0              // wave_id = tid / 64")
    lines.append(f"  v_lshlrev_b32 v{wave_off}, {shift}, v{wave_off}  // wave_id * {wave_region_size}")
    lines.append(f"  v_and_b32 v{lane}, 0x3F, v0                   // lane_id = tid % 64")

    for pair_base, pair in enumerate(mfma_pairs):
        acc = pair['acc']
        pair_offset = pair_base * WAVESIZE * 16  # bytes within one wave's region

        lines.append(f"  // Pair mma0={pair['mma0']} mma1={pair['mma1']}: a[{acc}:{acc+3}]")
        lines.append(f"  v_accvgpr_read v{r0}, a{acc}")
        lines.append(f"  v_accvgpr_read v{r1}, a{acc + 1}")
        lines.append(f"  v_accvgpr_read v{r2}, a{acc + 2}")
        lines.append(f"  v_accvgpr_read v{r3}, a{acc + 3}")
        lines.append(f"  s_nop 1  // wait for accvgpr_read")
        # address = output_ptr + wave_offset + pair_offset + lane_id*16
        lines.append(f"  v_lshlrev_b32 v{addr_lo}, 4, v{lane}      // lane_id * 16 bytes")
        lines.append(f"  v_add_u32 v{addr_lo}, v{wave_off}, v{addr_lo}  // + wave_offset")
        if pair_offset > 0:
            lines.append(f"  v_add_u32 v{addr_lo}, {pair_offset}, v{addr_lo}  // + pair base")
        lines.append(f"  v_mov_b32 v{addr_hi}, s9                   // output_ptr hi")
        lines.append(f"  v_add_co_u32 v{addr_lo}, vcc, s8, v{addr_lo}")
        lines.append(f"  v_addc_co_u32 v{addr_hi}, vcc, v{addr_hi}, 0, vcc")
        lines.append(f"  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{r0}:{r3}]")
        lines.append(f"  s_waitcnt vmcnt(0)")

    return "\n".join(lines), next_v


# ---------------------------------------------------------------------------
# Full kernel generator
# ---------------------------------------------------------------------------

def generate_mfma_kernel_fp8(cfg):
    """Generate a complete FP8 GR->LDS->LR->MFMA kernel.

    All 4 waves export in parallel; output buffer layout:
      wave_id * wave_region_size + pair_idx * WAVESIZE*16 + lane_id*16

    Returns:
        (kernel_asm, writer, kernel, tileInfoA, tileInfoB, mfma_pairs,
         output_size, lds_size)
    """
    init_rocisa()

    writer, kernel, tileInfoA, tileInfoB = create_writer_fp8(cfg)

    # Reserve s0-s11 for hardware regs + kernarg loads
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    writer.sgprs["SrdA"] = writer.sgprPool.checkOutAligned(
        4, 4, "SrdA", preventOverflow=False)
    writer.sgprs["SrdB"] = writer.sgprPool.checkOutAligned(
        4, 4, "SrdB", preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrA"] = writer.sgprPool.checkOut(
        1, "LocalWriteBaseAddrA", preventOverflow=False)
    writer.sgprs["LocalWriteDTLOffsetA"] = writer.sgprPool.checkOut(
        1, "LocalWriteDTLOffsetA", preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrB"] = writer.sgprPool.checkOut(
        1, "LocalWriteBaseAddrB", preventOverflow=False)
    writer.sgprs["LocalWriteDTLOffsetB"] = writer.sgprPool.checkOut(
        1, "LocalWriteDTLOffsetB", preventOverflow=False)
    writer.sgprs["SwapA"] = writer.sgprPool.checkOut(1, "SwapA", preventOverflow=False)
    writer.sgprs["SwapB"] = writer.sgprPool.checkOut(1, "SwapB", preventOverflow=False)

    # LDS allocation (same formula as roundtrip test / KernelWriter.py)
    readSize = 2 * tileInfoA.subtileSize
    numASubtiles = tileInfoA.globalSubtileGrid[0] * tileInfoA.globalSubtileGrid[1]
    numBSubtiles = tileInfoB.globalSubtileGrid[0] * tileInfoB.globalSubtileGrid[1]
    sizeA = ((numASubtiles * tileInfoA.subtileSize + readSize - 1) // readSize) * readSize
    sizeB = ((numBSubtiles * tileInfoB.subtileSize + readSize - 1) // readSize) * readSize
    lds_size = sizeA + sizeB
    writer.ldsTotalSize = lds_size

    # Allocate VGPR tiles for A and B
    tileInfoA.allocVgprTileRegisters(writer, kernel)
    tileInfoB.allocVgprTileRegisters(writer, kernel)

    # Generate GRA, LRA, GR, LR modules (identical to roundtrip)
    gra_module  = graTileAssignment(writer, kernel, useSwizzling=True)
    lra_module  = lraTileAssignment(writer, kernel)
    dtl_module  = globalReadDTLInitCommonSgpr(writer, kernel)
    gr_a_module = globalReadDoSubtile('A', writer, kernel)
    gr_b_module = globalReadDoSubtile('B', writer, kernel)
    wait_gr     = SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1)
    barrier     = SBarrier()
    lr_a_module = localReadDoSubtile('A', writer, kernel)
    lr_b_module = localReadDoSubtile('B', writer, kernel)
    wait_lr     = SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1)

    # MFMA: enumerate pairs and allocate AGPRs
    mfma_pairs = generate_mfma_pairs(tileInfoA, tileInfoB, writer)
    mfma_asm   = generate_mfma_asm(mfma_pairs)

    # Export: all waves write in parallel to their own region
    export_asm, _ = generate_mfma_export_asm(mfma_pairs, tileInfoA, tileInfoB)

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
        mfma_asm,
        export_asm,
    ])

    args = (
        ("input_A_ptr", 8, "global_buffer", "u8"),
        ("input_B_ptr", 8, "global_buffer", "u8"),
        ("output_ptr",  8, "global_buffer", "f32"),
        ("strideA",     4, "by_value",      "u32"),
        ("strideB",     4, "by_value",      "u32"),
    )
    kernel_asm = generate_kernel_asm(inner_asm, writer, args, lds_size)

    wave_region_size = len(mfma_pairs) * WAVESIZE * ACC_SIZE * 4
    output_size = NUM_WAVES * wave_region_size  # all 4 waves
    return (kernel_asm, writer, kernel, tileInfoA, tileInfoB,
            mfma_pairs, output_size, lds_size)


# ---------------------------------------------------------------------------
# Expected MFMA output computation
# ---------------------------------------------------------------------------

def compute_expected_mfma_fp8(cfg, tileInfoA, tileInfoB, kernel,
                               input_A, input_B, wave_id):
    """Compute expected float32 MFMA accumulator values for each pair.

    The MFMA instruction computes (in AMD ISA notation):
      D[M, N] = SRC1[M, K] × SRC0[N, K] + C[M, N]
    where SRC0 = B (N-indexed) and SRC1 = A (M-indexed), with zero C.
    Equivalently: C_block = A_block @ B_block.T  (float32)

    Output layout per MFMA pair, per lane:
      Lane l: M_row = l % 16, N_col_group = l // 16
      4 values: C_block[M_row, N_col_group*4 : N_col_group*4 + 4]

    Returns:
        list of np.ndarray (float32, shape WAVESIZE×4), one per pair in
        (mma1, mma0) order.
    """
    mi_wave_group = kernel["MIWaveGroup"]

    # Decode FP8 E4M3FN bytes to float32
    A_f32 = np.array(
        [fp8_e4m3fn_to_float32(b) for b in input_A], dtype=np.float32
    ).reshape(cfg.mt_a, cfg.stride_a)
    B_f32 = np.array(
        [fp8_e4m3fn_to_float32(b) for b in input_B], dtype=np.float32
    ).reshape(cfg.mt_b, cfg.stride_b)

    # Wave-level row offsets
    wave_a_off = (wave_id % mi_wave_group[0]) * tileInfoA.localMMATileGrid[0] * 16
    wave_b_off = (wave_id // mi_wave_group[0]) * tileInfoB.localMMATileGrid[0] * 16

    results = []
    for mma1 in range(tileInfoB.localMMATileGrid[0]):
        for mma0 in range(tileInfoA.localMMATileGrid[0]):
            a_row = wave_a_off + mma0 * 16
            b_row = wave_b_off + mma1 * 16

            # 16-row slices, depth_u columns (the K dimension)
            A_blk = A_f32[a_row:a_row + 16, :cfg.depth_u]  # (16, 128)
            B_blk = B_f32[b_row:b_row + 16, :cfg.depth_u]  # (16, 128)

            # C = A @ B.T  (16×16, float32)
            C_blk = (A_blk @ B_blk.T).astype(np.float32)

            # Map to per-lane layout
            # Lane l: M_row = l%16, N_col_group = l//16 → 4 consecutive N cols
            lane_data = np.zeros(WAVESIZE * ACC_SIZE, dtype=np.float32)
            for lane in range(WAVESIZE):
                m = lane % 16
                ng = lane // 16
                lane_data[lane * ACC_SIZE:(lane + 1) * ACC_SIZE] = (
                    C_blk[m, ng * ACC_SIZE:(ng + 1) * ACC_SIZE]
                )
            results.append(lane_data)

    return results


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compute_expected_all_waves(cfg, tileInfoA, tileInfoB, kernel, input_A, input_B):
    """Compute expected output for all 4 waves, concatenated in wave order."""
    all_pairs = []
    for wave_id in range(NUM_WAVES):
        all_pairs.extend(
            compute_expected_mfma_fp8(cfg, tileInfoA, tileInfoB, kernel,
                                      input_A, input_B, wave_id)
        )
    return all_pairs


def compare_mfma_output(actual_bytes, expected_pairs, debug=False):
    """Compare GPU output float32 values against Python-computed expected.

    Returns error count.
    """
    errors = 0
    pair_size = WAVESIZE * ACC_SIZE * 4  # bytes per pair

    for pair_idx, expected in enumerate(expected_pairs):
        offset = pair_idx * pair_size
        actual = np.frombuffer(actual_bytes[offset:offset + pair_size],
                               dtype=np.float32)

        if not np.allclose(actual, expected, rtol=1e-3, atol=50.0):
            errors += 1
            if errors <= 4 or debug:
                for lane in range(WAVESIZE):
                    a_sl = actual[lane * ACC_SIZE:(lane + 1) * ACC_SIZE]
                    e_sl = expected[lane * ACC_SIZE:(lane + 1) * ACC_SIZE]
                    if not np.allclose(a_sl, e_sl, rtol=1e-4, atol=1.0):
                        wave = pair_idx // (len(expected_pairs) // NUM_WAVES)
                        local_pair = pair_idx % (len(expected_pairs) // NUM_WAVES)
                        print(f"  MISMATCH wave={wave} pair={local_pair} lane={lane}:")
                        print(f"    expected: {e_sl.tolist()}")
                        print(f"    actual:   {a_sl.tolist()}")
                        if not debug:
                            break

    return errors


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestMfmaFP8:
    """GPU tests for FP8 GR->LDS->LR->MFMA roundtrip."""

    @pytest.fixture(params=CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    def test_mfma_fp8(self, cfg, tmp_path):
        """Verify FP8 MFMA accumulator values after full GR->LDS->LR->MFMA pipeline.

        All 4 waves export in parallel; output is verified for each wave.
        """
        sys.stdout.flush()

        (kernel_asm, writer, kernel, tileInfoA, tileInfoB,
         mfma_pairs, output_size, lds_size) = generate_mfma_kernel_fp8(cfg)

        input_A = make_fp8_input(cfg.mt_a, cfg.stride_a, seed=42)
        input_B = make_fp8_input(cfg.mt_b, cfg.stride_b, seed=43)

        label = f"mfma_fp8_{cfg.label}"
        output_bytes = assemble_and_run(
            kernel_asm, tmp_path, label, output_size,
            inputs=(input_A, input_B),
            scalars=(cfg.stride_a, cfg.stride_b),
            lds_size=lds_size,
        )

        expected_pairs = compute_expected_all_waves(
            cfg, tileInfoA, tileInfoB, kernel, input_A, input_B)
        errors = compare_mfma_output(output_bytes, expected_pairs)

        assert errors == 0, (
            f"Config {cfg.label}: {errors} MFMA pair mismatches"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FP8 MFMA GPU test")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed mismatch output and kernel ASM")
    parser.add_argument("--config", type=int, default=None,
                        help="Config index to test (default: all)")
    args = parser.parse_args()

    if not HAS_HIP:
        print("HIP not available — cannot run GPU test")
        sys.exit(1)

    config_list  = CONFIGS if args.config is None else [CONFIGS[args.config]]

    total_errors = 0
    total_tests  = 0

    for cfg in config_list:
        total_tests += 1
        print(f"\n{'='*60}")
        print(f"FP8 MFMA Config: {cfg.label}")
        print(f"  mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, depth_u={cfg.depth_u}")
        print(f"  stride_a={cfg.stride_a}, stride_b={cfg.stride_b}")

        (kernel_asm, writer, kernel, tileInfoA, tileInfoB,
         mfma_pairs, output_size, lds_size) = generate_mfma_kernel_fp8(cfg)

        print(f"  MIWaveGroup: {kernel['MIWaveGroup']}")
        print(f"  A tiles: {len(tileInfoA.vgprTiles)}, B tiles: {len(tileInfoB.vgprTiles)}")
        print(f"  MFMA pairs/wave: {len(mfma_pairs)}, output: {output_size} bytes (all waves)")
        print(f"  LDS: {lds_size} bytes")

        if args.debug:
            for p in mfma_pairs:
                print(f"    pair mma0={p['mma0']} mma1={p['mma1']}: "
                      f"A v[{p['a_lo']}:{p['a_lo']+7}], "
                      f"B v[{p['b_lo']}:{p['b_lo']+7}], "
                      f"acc a[{p['acc']}:{p['acc']+3}]")
            print(f"\n--- Kernel ASM ---\n{kernel_asm}\n--- End ---\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {
                '__truediv__': lambda s, n: os.path.join(tmp_dir, n)
            })()

            input_A = make_fp8_input(cfg.mt_a, cfg.stride_a, seed=42)
            input_B = make_fp8_input(cfg.mt_b, cfg.stride_b, seed=43)

            output_bytes = assemble_and_run(
                kernel_asm, tmp_path, f"mfma_fp8_{cfg.label}", output_size,
                inputs=(input_A, input_B),
                scalars=(cfg.stride_a, cfg.stride_b),
                lds_size=lds_size,
            )

            expected_pairs = compute_expected_all_waves(
                cfg, tileInfoA, tileInfoB, kernel, input_A, input_B)
            errors = compare_mfma_output(output_bytes, expected_pairs, debug=args.debug)

            if errors == 0:
                print(f"  PASS")
            else:
                print(f"  FAIL: {errors} pair mismatches")
                total_errors += errors

    print(f"\n{'='*60}")
    print(f"Result: {total_tests} tests, {total_errors} errors")
    if total_errors > 0:
        print("FAILED")
        sys.exit(1)
    else:
        print("PASSED")
