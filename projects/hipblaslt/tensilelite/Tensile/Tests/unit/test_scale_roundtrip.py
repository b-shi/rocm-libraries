#!/usr/bin/env python3
################################################################################
# GPU tests for scale tensor LDS layout (MX FP4).
#
# TestScaleLdsDumpGPU — Direct LDS content verification:
#   1. Init LDS to 0xFFFFFFFF marker
#   2. Compute GR offsets (production code: strided 2D formula)
#   3. flat_load_dwordx4 from global scale buffer at GR offset
#   4. ds_write_b128 to LDS at GR offset
#   5. s_barrier
#   6. Each thread reads ds_read_b128 at tid*16 (sequential dump)
#   7. Export via global_store_dwordx4
#   8. Verify strided 2D pattern matches expected
#
# TestScaleRoundtripGPU — GR -> LDS -> LR roundtrip
#
# GR offset formula (production _graTileAssignmentScaleSwizzledCommon):
#   grOffset = (serial / numTPG) * StridesMXS + (serial % numTPG) * loadWidthGR
#   numTPG = (subtileSize * localSubtileGrid[1]) / loadWidthGR
#
# Usage:
#   pytest test_scale_roundtrip.py -v -s
#   python test_scale_roundtrip.py --lds-dump --debug
################################################################################

import math
import os
import struct
import sys
import tempfile
from dataclasses import dataclass

import pytest
import numpy as np

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    WAVESIZE, NUM_WAVES, NUM_THREADS,
    create_writer,
    init_rocisa,
    generate_kernel_asm,
    generate_load_params,
    assemble_and_run,
    print_offset_grid,
)

from test_lraTileAssignment import compute_expected_scale_lr_offset

from Tensile.Components.SubtileBasedKernel import (
    graTileAssignmentScaleSwizzled,
    lraTileAssignmentScaleSwizzled,
)


# ---------------------------------------------------------------------------
# Scale config with stride_mxsa/stride_mxsb
# ---------------------------------------------------------------------------
@dataclass
class ScaleConfig(TileConfig):
    stride_mxsa: int = 0
    stride_mxsb: int = 0

    @property
    def label(self):
        base = super().label
        return f"{base}_smxs{self.stride_mxsa}" if self.stride_mxsa else base


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
# Tight stride = numTPG * loadWidthGR (no gaps between groups).
# All configs must have localSubtileGrid[0] >= 1 for both MXSA and MXSB.
SCALE_LDS_CONFIGS = [
    # 2x2, numTPG=4, tight stride=64
    ScaleConfig(mt_a=256, mt_b=256, depth_u=64,  stride_a=64,  stride_b=64,  mxblock=32, stride_mxsa=64,  stride_mxsb=64),
    # 2x2, numTPG=8, tight stride=128
    ScaleConfig(mt_a=256, mt_b=256, depth_u=128, stride_a=128, stride_b=128, mxblock=32, stride_mxsa=128, stride_mxsb=128),
    # 2x2, non-square macro tile
    ScaleConfig(mt_a=96,  mt_b=256, depth_u=64,  stride_a=64,  stride_b=64,  mxblock=32, stride_mxsa=64,  stride_mxsb=64),
    # 1x4 wave group
    ScaleConfig(mt_a=80,  mt_b=256, depth_u=64,  stride_a=64,  stride_b=64,  mxblock=32, stride_mxsa=64,  stride_mxsb=64),
    # 4x1 wave group
    ScaleConfig(mt_a=256, mt_b=80,  depth_u=64,  stride_a=64,  stride_b=64,  mxblock=32, stride_mxsa=64,  stride_mxsb=64),
]

SCALE_ROUNDTRIP_CONFIGS = [
    ScaleConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=64,  stride_b=64,  mxblock=32, stride_mxsa=64, stride_mxsb=64),
    ScaleConfig(mt_a=80,  mt_b=256, depth_u=64, stride_a=64,  stride_b=64,  mxblock=32, stride_mxsa=64, stride_mxsb=64),
    ScaleConfig(mt_a=256, mt_b=80,  depth_u=64, stride_a=64,  stride_b=64,  mxblock=32, stride_mxsa=64, stride_mxsb=64),
    ScaleConfig(mt_a=96,  mt_b=256, depth_u=64, stride_a=128, stride_b=128, mxblock=32, stride_mxsa=64, stride_mxsb=64),
]

LDS_DUMP_SIZE = NUM_THREADS * 16  # 4096 bytes: 256 threads x 16 bytes each

# Load params and kernel args shared by all scale kernels
SCALE_LOAD_PARAMS = [
    (4, 4, 0x00, "input_A_ptr + input_B_ptr"),
    (8, 4, 0x10, "output_ptr + strideA + strideB"),
    (12, 2, 0x20, "strideMXSA + strideMXSB"),
]

SCALE_KERNEL_ARGS = (
    ("input_scale_A_ptr", 8, "global_buffer", "u8"),
    ("input_scale_B_ptr", 8, "global_buffer", "u8"),
    ("output_ptr",        8, "global_buffer", "u32"),
    ("strideA",           4, "by_value",      "u32"),
    ("strideB",           4, "by_value",      "u32"),
    ("strideMXSA",        4, "by_value",      "u32"),
    ("strideMXSB",        4, "by_value",      "u32"),
)


# ---------------------------------------------------------------------------
# Python reference for strided 2D GR offset
# ---------------------------------------------------------------------------
def compute_expected_scale_gr_offset(tid, cfg, scaleTileInfo, tc):
    """Python reference matching _graTileAssignmentScaleSwizzledCommon.

    grOffset = (serial / numTPG) * strideBytes + (serial % numTPG) * loadWidthGR
    """
    loadWidth = scaleTileInfo.loadWidthGR  # 16
    numTPG = (scaleTileInfo.subtileSize * scaleTileInfo.localSubtileGrid[1]) // loadWidth
    stride = cfg.stride_mxsa if tc == 'A' else cfg.stride_mxsb
    strideBytes = stride * scaleTileInfo.bpe  # bpe=1
    groupId = tid >> int(math.log2(numTPG))
    colInGroup = tid & (numTPG - 1)
    colOffset = colInGroup << (loadWidth.bit_length() - 1)
    return [groupId * strideBytes + colOffset]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_lds_sizes(kernel):
    """Compute scale LDS sizes matching KernelWriter formula."""
    numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
    loadWidthGR = 16
    sizeMXSA = loadWidthGR * WAVESIZE * numWaves  # 4096
    sizeMXSB = loadWidthGR * WAVESIZE * numWaves
    return sizeMXSA, sizeMXSB


def compute_input_size(cfg, scaleTileInfo, tc):
    """Max GR offset + loadWidthGR across all threads."""
    max_off = 0
    for tid in range(NUM_THREADS):
        offsets = compute_expected_scale_gr_offset(tid, cfg, scaleTileInfo, tc)
        max_off = max(max_off, offsets[0])
    return max_off + scaleTileInfo.loadWidthGR


def generate_block_input_data(size, block_size=256):
    """Each block_size-byte chunk gets a constant value, starting at 1."""
    return np.array([1 + i // block_size for i in range(size)], dtype=np.uint8)


def generate_input_data(size):
    """Deterministic byte array for scale input (used by roundtrip test)."""
    return np.array([(i * 7 + 13) & 0xFF for i in range(size)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Common kernel setup
# ---------------------------------------------------------------------------
def _setup_writer_and_gr(cfg):
    """Create writer, allocate registers, run scale GR offset code.

    Returns (writer, kernel, tileInfoA, tileInfoB, gra_module,
             mxsaTileInfo, mxsbTileInfo).
    """
    init_rocisa()
    writer, kernel, tileInfoA, tileInfoB = create_writer(cfg)

    writer.sgprPool.checkOut(14)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    writer.sgprs["StridesMXSA"] = 12
    writer.sgprs["StridesMXSB"] = 13

    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    mxsaTileInfo = getattr(writer.states.mxsa, 'tileInfo', None)
    mxsbTileInfo = getattr(writer.states.mxsb, 'tileInfo', None)
    if mxsaTileInfo:
        mxsaTileInfo.allocOffsetRegisters(writer, kernel)
    if mxsbTileInfo:
        mxsbTileInfo.allocOffsetRegisters(writer, kernel)

    gra_module = graTileAssignmentScaleSwizzled(writer, kernel)
    return writer, kernel, tileInfoA, tileInfoB, gra_module, mxsaTileInfo, mxsbTileInfo


# ---------------------------------------------------------------------------
# Direct LDS content verification (GR write test)
# ---------------------------------------------------------------------------
def generate_lds_dump_kernel(cfg, tc):
    """Kernel: LDS marker init, scale GR write, dump LDS.

    Output is a raw 4096-byte image of LDS[0..4095] with strided 2D pattern:
      grOffset = (serial/numTPG)*stride + (serial%numTPG)*16
    """
    writer, kernel, tileInfoA, tileInfoB, gra_module, \
        mxsaTileInfo, mxsbTileInfo = _setup_writer_and_gr(cfg)

    scaleTileInfo = mxsaTileInfo if tc == 'A' else mxsbTileInfo
    grOffReg = scaleTileInfo.sharedVgprGROffset[0]
    ptrLo = 4 if tc == 'A' else 6
    ptrHi = 5 if tc == 'A' else 7

    lds_bytes = LDS_DUMP_SIZE

    # Allocate temp registers
    vAddr = writer.vgprPool.checkOutAligned(2, 2, "addr", preventOverflow=False)
    vData = writer.vgprPool.checkOutAligned(4, 4, "data", preventOverflow=False)
    vTmp = writer.vgprPool.checkOut(1, "tmp", preventOverflow=False)

    asm = f"""\
  // ---- Init LDS to 0xFFFFFFFF marker ----
  v_lshlrev_b32 v{vTmp}, 4, v0
  v_mov_b32 v{vData}, 0xffffffff
  v_mov_b32 v{vData+1}, 0xffffffff
  v_mov_b32 v{vData+2}, 0xffffffff
  v_mov_b32 v{vData+3}, 0xffffffff
  ds_write_b128 v{vTmp}, v[{vData}:{vData+3}]
  s_waitcnt lgkmcnt(0)
  s_barrier
  // ---- GR write for scale {tc} (strided 2D) ----
  v_mov_b32 v{vAddr}, s{ptrLo}
  v_mov_b32 v{vAddr+1}, s{ptrHi}
  v_add_co_u32 v{vAddr}, vcc, v{vAddr}, v{grOffReg}
  v_addc_co_u32 v{vAddr+1}, vcc, v{vAddr+1}, 0, vcc
  flat_load_dwordx4 v[{vData}:{vData+3}], v[{vAddr}:{vAddr+1}]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  ds_write_b128 v{grOffReg}, v[{vData}:{vData+3}]
  s_waitcnt lgkmcnt(0)
  s_barrier
  // ---- Dump LDS: 16 bytes per thread ----
  v_lshlrev_b32 v{vTmp}, 4, v0
  ds_read_b128 v[{vData}:{vData+3}], v{vTmp}
  s_waitcnt lgkmcnt(0)
  global_store_dwordx4 v{vTmp}, v[{vData}:{vData+3}], s[8:9]"""

    prologue = generate_load_params(SCALE_LOAD_PARAMS)
    inner_asm = "\n".join([str(prologue), str(gra_module), asm])
    kernel_asm = generate_kernel_asm(inner_asm, writer, SCALE_KERNEL_ARGS, lds_bytes)
    return kernel_asm, writer, kernel, tileInfoA, tileInfoB, lds_bytes


def compute_expected_lds_content(cfg, scaleTileInfo, input_data, tc):
    """Expected LDS[0..4095] after strided 2D GR write.

    Each unique grOffset writes 16 bytes: LDS[grOff..grOff+15] = input[grOff..grOff+15].
    Uncovered positions remain as marker (0xFF).
    """
    loadWidth = scaleTileInfo.loadWidthGR
    lds = np.full(LDS_DUMP_SIZE, 0xFF, dtype=np.uint8)
    seen = set()
    for tid in range(NUM_THREADS):
        grOff = compute_expected_scale_gr_offset(tid, cfg, scaleTileInfo, tc)[0]
        if grOff in seen or grOff >= LDS_DUMP_SIZE:
            continue
        seen.add(grOff)
        end = min(grOff + loadWidth, LDS_DUMP_SIZE)
        src_end = min(grOff + loadWidth, len(input_data))
        lds[grOff:end] = input_data[grOff:src_end]
    return lds


def print_scale_gr_grid(label, offsets, numTPG, wavesize=WAVESIZE, num_waves=NUM_WAVES,
                        group_size=16):
    """Print GR offsets in LDS layout format, lanes grouped by group_size.

    Output format:
        Wave 0:
        Lane  0-15: 0 16 32 48 64 80 ...
        Lane 16-31: 256 272 ...
        ...
    """
    print(f"\n--- {label} ---")
    for w in range(num_waves):
        print(f"  Wave {w}:")
        for start in range(0, wavesize, group_size):
            end = start + group_size - 1
            base_tid = w * wavesize + start
            vals = " ".join(str(offsets[base_tid + i]) for i in range(group_size))
            print(f"    Lane {start:>2}-{end:<2}: {vals}")


def print_lds_blocks(label, data, block_size=256):
    """Print LDS content summarized per block_size-byte chunk."""
    print(f"\n  --- {label} ---")
    for offset in range(0, min(len(data), LDS_DUMP_SIZE), block_size):
        chunk = data[offset:min(offset + block_size, len(data))]
        unique = set(chunk)
        if len(unique) == 1:
            val = unique.pop()
            tag = " (marker)" if val == 0xFF else " (zero)" if val == 0 else ""
            print(f"  LDS [{offset:>5}..{offset+len(chunk)-1:>5}]: "
                  f"0x{val:02x} x{len(chunk)}{tag}")
        else:
            vals = " ".join(f"{v:02x}" for v in chunk[:8])
            print(f"  LDS [{offset:>5}..{offset+len(chunk)-1:>5}]: "
                  f"{vals}... ({len(unique)} unique vals)")


@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestScaleLdsDumpGPU:
    """Verify LDS content after scale GR write (strided 2D layout)."""

    @pytest.fixture(params=SCALE_LDS_CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    def _run_lds_dump(self, cfg, tc, tmp_path):
        kernel_asm, writer, kernel, tileInfoA, tileInfoB, lds_bytes = \
            generate_lds_dump_kernel(cfg, tc)

        scaleTileInfo = getattr(writer.states.mxsa, 'tileInfo', None) if tc == 'A' \
                        else getattr(writer.states.mxsb, 'tileInfo', None)
        otherScaleTileInfo = getattr(writer.states.mxsb, 'tileInfo', None) if tc == 'A' \
                             else getattr(writer.states.mxsa, 'tileInfo', None)

        input_size = compute_input_size(cfg, scaleTileInfo, tc)
        input_data = generate_block_input_data(input_size)
        other_tc = 'B' if tc == 'A' else 'A'
        other_size = compute_input_size(cfg, otherScaleTileInfo, other_tc)
        other_input = generate_block_input_data(other_size)

        if tc == 'A':
            inputs = (input_data, other_input)
        else:
            inputs = (other_input, input_data)

        label = f"lds_dump_{tc}_{cfg.label}"
        raw = assemble_and_run(kernel_asm, tmp_path, label, LDS_DUMP_SIZE,
                               inputs=inputs,
                               scalars=(cfg.stride_a, cfg.stride_b,
                                        cfg.stride_mxsa, cfg.stride_mxsb),
                               lds_size=lds_bytes)

        actual = np.frombuffer(raw, dtype=np.uint8)
        expected = compute_expected_lds_content(cfg, scaleTileInfo, input_data, tc)

        errors = 0
        for i in range(LDS_DUMP_SIZE):
            if actual[i] != expected[i]:
                errors += 1
                if errors <= 16:
                    print(f"  byte {i}: got 0x{actual[i]:02x}, expected 0x{expected[i]:02x}")

        if errors > 0:
            print_lds_blocks(f"ACTUAL scale {tc}", actual)
            print_lds_blocks(f"EXPECTED scale {tc}", expected)

        assert errors == 0, \
            f"Scale {tc} LDS content ({cfg.label}): {errors} errors"

    def test_lds_content_a(self, cfg, tmp_path):
        """Verify LDS content after GR write for scale A."""
        self._run_lds_dump(cfg, 'A', tmp_path)

    def test_lds_content_b(self, cfg, tmp_path):
        """Verify LDS content after GR write for scale B."""
        self._run_lds_dump(cfg, 'B', tmp_path)


# ---------------------------------------------------------------------------
# GR -> LDS -> LR roundtrip test
# ---------------------------------------------------------------------------
def generate_roundtrip_kernel(cfg, tc):
    """Generate a complete kernel using production scale GR/LR code paths."""
    writer, kernel, tileInfoA, tileInfoB, gra_module, \
        mxsaTileInfo, mxsbTileInfo = _setup_writer_and_gr(cfg)

    lra_module = lraTileAssignmentScaleSwizzled(writer, kernel)

    scaleTileInfo = mxsaTileInfo if tc == 'A' else mxsbTileInfo
    grOffReg = scaleTileInfo.sharedVgprGROffset[0]
    lrOffReg = scaleTileInfo.sharedVgprLROffset[0]
    ptrLo = 4 if tc == 'A' else 6
    ptrHi = 5 if tc == 'A' else 7

    sizeMXSA, sizeMXSB = compute_lds_sizes(kernel)
    # LR offset includes full ldsStartOffsetMXSA/B; subtract it to get
    # the local position within our compact test LDS [ScaleA | ScaleB].
    ldsBase = writer.ldsStartOffsetMXSA if tc == 'A' else writer.ldsStartOffsetMXSB
    lds_bytes = sizeMXSA + sizeMXSB

    vAddr = writer.vgprPool.checkOutAligned(2, 2, "addr", preventOverflow=False)
    vData = writer.vgprPool.checkOutAligned(4, 4, "data", preventOverflow=False)
    vLrAdj = writer.vgprPool.checkOut(1, "lr_adj", preventOverflow=False)
    vByteOff = writer.vgprPool.checkOut(1, "byte_off", preventOverflow=False)
    sTmp = writer.sgprPool.checkOut(1, "tmp", preventOverflow=False)

    # GR offset is linear (serial * 16), used for both global load and LDS write.
    # For scale B, shift the LDS write by sizeMXSA so B doesn't overwrite A.
    grLdsShift = 0 if tc == 'A' else sizeMXSA

    roundtrip_asm = f"""\
  // ---- Roundtrip for scale {tc} (DTL: 16-byte load/write) ----
  v_mov_b32 v{vAddr}, s{ptrLo}
  v_mov_b32 v{vAddr+1}, s{ptrHi}
  v_add_co_u32 v{vAddr}, vcc, v{vAddr}, v{grOffReg}
  v_addc_co_u32 v{vAddr+1}, vcc, v{vAddr+1}, 0, vcc
  flat_load_dwordx4 v[{vData}:{vData+3}], v[{vAddr}:{vAddr+1}]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  v_add_u32 v{vLrAdj}, {grLdsShift}, v{grOffReg}
  ds_write_b128 v{vLrAdj}, v[{vData}:{vData+3}]
  s_waitcnt lgkmcnt(0)
  s_barrier
  // LR offset from production includes ldsStartOffsetMXS{tc}; subtract to get local pos
  s_mov_b32 s{sTmp}, {ldsBase}
  v_sub_u32 v{vLrAdj}, v{lrOffReg}, s{sTmp}
  // Add grLdsShift to align with where we wrote in LDS
  v_add_u32 v{vLrAdj}, {grLdsShift}, v{vLrAdj}
  ds_read_u8 v{vData}, v{vLrAdj}
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32 v{vByteOff}, 2, v0
  global_store_dword v{vByteOff}, v{vData}, s[8:9]"""

    prologue = generate_load_params(SCALE_LOAD_PARAMS)

    inner_asm = "\n".join([
        str(prologue),
        str(gra_module),
        str(lra_module),
        roundtrip_asm,
    ])

    kernel_asm = generate_kernel_asm(inner_asm, writer, SCALE_KERNEL_ARGS, lds_bytes)
    return kernel_asm, writer, kernel, tileInfoA, tileInfoB, lds_bytes


def compute_expected_roundtrip(cfg, writer, input_data, tc, kernel):
    """Compute expected scale byte for each thread after the roundtrip.

    GR writes linear pattern to LDS; LR reads from (lrOffset - ldsBase).
    """
    scaleTileInfo = writer.states.mxsa.tileInfo if tc == 'A' else writer.states.mxsb.tileInfo
    ldsBase = writer.ldsStartOffsetMXSA if tc == 'A' else writer.ldsStartOffsetMXSB

    sizeMXSA, sizeMXSB = compute_lds_sizes(kernel)
    scaleLdsSize = sizeMXSA if tc == 'A' else sizeMXSB

    expected = [0] * NUM_THREADS
    for T in range(NUM_THREADS):
        lr_offset = compute_expected_scale_lr_offset(T, scaleTileInfo, kernel, ldsBase)[0]
        lds_pos = lr_offset - ldsBase

        assert 0 <= lds_pos < scaleLdsSize, \
            f"Thread {T}: LDS pos {lds_pos} (lr={lr_offset}, base={ldsBase}) out of range [0, {scaleLdsSize})"

        assert lds_pos < len(input_data), \
            f"Thread {T}: LDS pos {lds_pos} >= input size {len(input_data)}"

        expected[T] = int(input_data[lds_pos])

    return expected


def build_and_run_roundtrip(cfg, tc, tmp_path, debug=False):
    """Generate, assemble, run roundtrip for one matrix; return (results, expected)."""
    sys.stdout.flush()

    kernel_asm, writer, kernel, tileInfoA, tileInfoB, lds_bytes = \
        generate_roundtrip_kernel(cfg, tc)

    if debug:
        print(f"\n--- Kernel ASM (scale {tc}, {cfg.label}) ---")
        print(kernel_asm)
        print("--- End ---\n")

    scaleTileInfo = getattr(writer.states.mxsa, 'tileInfo', None) if tc == 'A' \
                    else getattr(writer.states.mxsb, 'tileInfo', None)
    otherScaleTileInfo = getattr(writer.states.mxsb, 'tileInfo', None) if tc == 'A' \
                         else getattr(writer.states.mxsa, 'tileInfo', None)

    input_size = compute_input_size(cfg, scaleTileInfo, tc)
    input_data = generate_input_data(input_size)

    other_tc = 'B' if tc == 'A' else 'A'
    other_input_size = compute_input_size(cfg, otherScaleTileInfo, other_tc)
    other_input = generate_input_data(other_input_size)

    if tc == 'A':
        input_a, input_b = input_data, other_input
    else:
        input_a, input_b = other_input, input_data

    out_size = NUM_THREADS * 4
    label = f"scale_roundtrip_{tc}_{cfg.label}"
    output_bytes = assemble_and_run(kernel_asm, tmp_path, label, out_size,
                                    inputs=(input_a, input_b),
                                    scalars=(cfg.stride_a, cfg.stride_b,
                                             cfg.stride_mxsa, cfg.stride_mxsb),
                                    lds_size=lds_bytes)

    results = struct.unpack(f"{NUM_THREADS}I", output_bytes)
    expected = compute_expected_roundtrip(cfg, writer, input_data, tc, kernel)
    return results, expected


@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestScaleRoundtripGPU:

    @pytest.fixture(params=SCALE_ROUNDTRIP_CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    def test_roundtrip_scale_a(self, cfg, tmp_path):
        """Verify GR -> LDS -> LR roundtrip for scale A."""
        results, expected = build_and_run_roundtrip(cfg, 'A', tmp_path)
        errors = 0
        for tid in range(NUM_THREADS):
            if results[tid] != expected[tid]:
                errors += 1
                if errors <= 8:
                    print(f"  MISMATCH scale A tid={tid}: got {results[tid]}, expected {expected[tid]}")
        assert errors == 0, f"Scale A roundtrip {cfg.label}: {errors}/{NUM_THREADS} mismatches"

    def test_roundtrip_scale_b(self, cfg, tmp_path):
        """Verify GR -> LDS -> LR roundtrip for scale B."""
        results, expected = build_and_run_roundtrip(cfg, 'B', tmp_path)
        errors = 0
        for tid in range(NUM_THREADS):
            if results[tid] != expected[tid]:
                errors += 1
                if errors <= 8:
                    print(f"  MISMATCH scale B tid={tid}: got {results[tid]}, expected {expected[tid]}")
        assert errors == 0, f"Scale B roundtrip {cfg.label}: {errors}/{NUM_THREADS} mismatches"


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------
def print_intermediate_values(cfg, writer, tc, kernel):
    """Print GR offset, LR offset, and expected value per thread."""
    scaleTileInfo = writer.states.mxsa.tileInfo if tc == 'A' \
                    else writer.states.mxsb.tileInfo
    ldsBase = writer.ldsStartOffsetMXSA if tc == 'A' else writer.ldsStartOffsetMXSB

    numTPG = (scaleTileInfo.subtileSize * scaleTileInfo.localSubtileGrid[1]) // scaleTileInfo.loadWidthGR

    print(f"\n  Intermediate values for scale {tc}:")
    print(f"  numTPG={numTPG}, stride_mxs={cfg.stride_mxsa if tc == 'A' else cfg.stride_mxsb}, ldsBase={ldsBase}")
    print(f"  subtileSize={scaleTileInfo.subtileSize}, localSubtileGrid={scaleTileInfo.localSubtileGrid}")
    print(f"  {'tid':>4} {'GR_off':>7} {'LR_off':>7} {'LR_adj':>7}")
    print(f"  {'-'*30}")
    for T in range(min(NUM_THREADS, 64)):  # first wave
        gr_off = compute_expected_scale_gr_offset(T, cfg, scaleTileInfo, tc)[0]
        lr_off = compute_expected_scale_lr_offset(T, scaleTileInfo, kernel, ldsBase)[0]
        print(f"  {T:>4} {gr_off:>7} {lr_off:>7} {lr_off - ldsBase:>7}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scale GR/LDS GPU tests")
    parser.add_argument("--lds-dump", action="store_true",
                        help="Run LDS dump test (verify LDS content after GR write)")
    parser.add_argument("--roundtrip", action="store_true",
                        help="Run GR-LR roundtrip test")
    parser.add_argument("--debug", action="store_true",
                        help="Print intermediate values and full asm (implies --grid)")
    parser.add_argument("--grid", action="store_true",
                        help="Display actual/expected as 2D matrix grids")
    parser.add_argument("--config", type=int, default=None, help="Config index (default: all)")
    parser.add_argument("--tc", default="AB", help="Matrix to test: A, B, or AB (default)")
    parser.add_argument("--list", action="store_true", help="List available configs and exit")
    args = parser.parse_args()

    # Default: run LDS dump if neither mode specified
    if not args.lds_dump and not args.roundtrip:
        args.lds_dump = True

    if args.list:
        print("  LDS dump configs:")
        for i, cfg in enumerate(SCALE_LDS_CONFIGS):
            print(f"    {i}: {cfg.label}  (mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, "
                  f"du={cfg.depth_u}, mx={cfg.mxblock}, smxsa={cfg.stride_mxsa})")
        print("  Roundtrip configs:")
        for i, cfg in enumerate(SCALE_ROUNDTRIP_CONFIGS):
            print(f"    {i}: {cfg.label}  (mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, "
                  f"du={cfg.depth_u}, mx={cfg.mxblock}, smxsa={cfg.stride_mxsa})")
        sys.exit(0)

    if args.debug:
        args.grid = True

    if not HAS_HIP:
        print("HIP not available")
        sys.exit(1)

    tc_list = list(args.tc)
    total_errors = 0

    # --- LDS dump mode ---
    if args.lds_dump:
        configs = SCALE_LDS_CONFIGS if args.config is None else [SCALE_LDS_CONFIGS[args.config]]
        for cfg in configs:
            for tc in tc_list:
                print(f"\n{'='*60}")
                print(f"  LDS dump: {cfg.label}, matrix: {tc}")
                print(f"{'='*60}")

                kernel_asm, writer, kernel, tileInfoA, tileInfoB, lds_bytes = \
                    generate_lds_dump_kernel(cfg, tc)

                scaleTileInfo = getattr(writer.states.mxsa, 'tileInfo', None) if tc == 'A' \
                                else getattr(writer.states.mxsb, 'tileInfo', None)
                otherScaleTileInfo = getattr(writer.states.mxsb, 'tileInfo', None) if tc == 'A' \
                                     else getattr(writer.states.mxsa, 'tileInfo', None)

                numTPG = (scaleTileInfo.subtileSize * scaleTileInfo.localSubtileGrid[1]) // scaleTileInfo.loadWidthGR
                print(f"  numTPG={numTPG}, stride_mxs={cfg.stride_mxsa if tc == 'A' else cfg.stride_mxsb}")

                input_size = compute_input_size(cfg, scaleTileInfo, tc)
                input_data = generate_block_input_data(input_size)
                other_tc = 'B' if tc == 'A' else 'A'
                other_size = compute_input_size(cfg, otherScaleTileInfo, other_tc)
                other_input = generate_block_input_data(other_size)

                if tc == 'A':
                    inputs = (input_data, other_input)
                else:
                    inputs = (other_input, input_data)

                expected = compute_expected_lds_content(cfg, scaleTileInfo, input_data, tc)

                if args.debug:
                    print_intermediate_values(cfg, writer, tc, kernel)
                    print_lds_blocks(f"EXPECTED scale {tc}", expected)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()
                    raw = assemble_and_run(kernel_asm, tmp_path,
                                           f"lds_dump_{tc}_{cfg.label}",
                                           LDS_DUMP_SIZE,
                                           inputs=inputs,
                                           scalars=(cfg.stride_a, cfg.stride_b,
                                                    cfg.stride_mxsa, cfg.stride_mxsb),
                                           lds_size=lds_bytes)

                actual = np.frombuffer(raw, dtype=np.uint8)

                if args.debug or args.grid:
                    print_lds_blocks(f"ACTUAL scale {tc}", actual)

                    # Print GR offsets grouped by numTPG
                    gr_offsets = [compute_expected_scale_gr_offset(t, cfg, scaleTileInfo, tc)[0]
                                 for t in range(NUM_THREADS)]
                    print_scale_gr_grid(f"Scale {tc} GR offsets ({cfg.label})",
                                        gr_offsets, numTPG)

                errors = 0
                for i in range(LDS_DUMP_SIZE):
                    if actual[i] != expected[i]:
                        errors += 1
                        if errors <= 16 or args.debug:
                            print(f"  byte {i}: got 0x{actual[i]:02x}, "
                                  f"expected 0x{expected[i]:02x}")

                print(f"  {'PASS' if errors == 0 else f'FAIL: {errors} errors'}")
                total_errors += errors

    # --- Roundtrip mode ---
    if args.roundtrip:
        configs = SCALE_ROUNDTRIP_CONFIGS if args.config is None else [SCALE_ROUNDTRIP_CONFIGS[args.config]]
        for cfg in configs:
            for tc in tc_list:
                print(f"\n{'='*60}")
                print(f"  Roundtrip: {cfg.label}, matrix: {tc}")
                print(f"{'='*60}")

                if args.debug:
                    kernel_asm, writer, kernel, tileInfoA, tileInfoB, _ = \
                        generate_roundtrip_kernel(cfg, tc)
                    print_intermediate_values(cfg, writer, tc, kernel)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()
                    results, expected = build_and_run_roundtrip(cfg, tc, tmp_path,
                                                                debug=args.debug)

                if args.grid:
                    print_offset_grid(f"Scale {tc} GPU result ({cfg.label})",
                                      results, WAVESIZE, NUM_WAVES)
                    print_offset_grid(f"Scale {tc} EXPECTED ({cfg.label})",
                                      expected, WAVESIZE, NUM_WAVES)

                errors = 0
                for tid in range(NUM_THREADS):
                    if results[tid] != expected[tid]:
                        errors += 1
                        if errors <= 8 or args.debug:
                            print(f"  MISMATCH tid={tid}: got {results[tid]}, "
                                  f"expected {expected[tid]}")

                print(f"  {'PASS' if errors == 0 else f'FAIL: {errors} errors'}")
                total_errors += errors

    print(f"\n{'='*60}")
    print(f"{'PASSED' if total_errors == 0 else f'FAILED ({total_errors} errors)'}")
    sys.exit(0 if total_errors == 0 else 1)
