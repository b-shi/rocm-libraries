#!/usr/bin/env python3
################################################################################
# End-to-end GPU roundtrip test for scale tensors (MX FP4).
#
# Verifies GR and LR offset consistency: data loaded from global memory
# via GR offsets, written to LDS, and read back via LR offsets should
# deliver the correct scale bytes.
#
# Flow per matrix (A or B):
#   1. Compute GR + LR offsets (production code)
#   2. flat_load_ubyte from global scale buffer at GR offset
#   3. ds_write_b8 to LDS at position = serial
#   4. s_barrier
#   5. ds_read_u8 from LDS at (LR offset - dataLdsSize)
#   6. Export result
#
# Usage:
#   pytest test_scale_roundtrip.py -v -s
#   python test_scale_roundtrip.py --debug
################################################################################

import os
import re
import sys
import ctypes
import struct

import pytest
import numpy as np

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    BPE, WAVESIZE, NUM_THREADS, GFX_TARGET,
    create_writer_for_gpu,
    init_rocisa,
    assemble_kernel,
    hip_check,
)

from test_graTileAssignment import compute_expected_scale_gr_offset
from test_lraTileAssignment import compute_expected_scale_lr_offset

from Tensile.Components.SubtileBasedKernel import (
    graTileAssignmentScaleSwizzled,
    lraTileAssignmentScaleSwizzled,
)

if HAS_HIP:
    from hip import hip  # type: ignore

# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
SCALE_ROUNDTRIP_CONFIGS = [
    # 2x2 wave group, dense stride
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=64,  stride_b=64,  mxblock=32),
    # 1x4 wave group
    TileConfig(mt_a=80,  mt_b=64,  depth_u=64, stride_a=64,  stride_b=64,  mxblock=32),
    # 4x1 wave group
    TileConfig(mt_a=64,  mt_b=80,  depth_u=64, stride_a=64,  stride_b=64,  mxblock=32),
    # Non-trivial stride (stride > depthU, tests stride division by mxBlock)
    TileConfig(mt_a=96,  mt_b=256, depth_u=64, stride_a=128, stride_b=128, mxblock=32),
]


# ---------------------------------------------------------------------------
# LDS size computation (mirrors lraTileAssignmentScaleSwizzled)
# ---------------------------------------------------------------------------
def compute_lds_sizes(cfg, tileInfoA, tileInfoB, kernel):
    """Compute LDS layout sizes matching production code."""
    MT0A = tileInfoA.globalMMATileGrid[0] * tileInfoA.mmaTileShape[0]
    MT0B = tileInfoB.globalMMATileGrid[0] * tileInfoB.mmaTileShape[0]
    dataLdsSize = (MT0A * cfg.depth_u * tileInfoA.bpe) + \
                  (MT0B * cfg.depth_u * tileInfoB.bpe)

    numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]

    scaleALdsRaw = MT0A * tileInfoA.scaleDepthU * tileInfoA.scaleBpe if tileInfoA.mxBlock > 0 else 0
    ldsAlignment = WAVESIZE * numWaves * (tileInfoA.scaleLoadWidth if tileInfoA.mxBlock > 0 else 1)
    scaleALdsSize = ((scaleALdsRaw + ldsAlignment - 1) // ldsAlignment) * ldsAlignment if scaleALdsRaw > 0 else 0

    scaleBLdsRaw = MT0B * tileInfoB.scaleDepthU * tileInfoB.scaleBpe if tileInfoB.mxBlock > 0 else 0
    scaleBLdsSize = ((scaleBLdsRaw + ldsAlignment - 1) // ldsAlignment) * ldsAlignment if scaleBLdsRaw > 0 else 0

    return dataLdsSize, scaleALdsSize, scaleBLdsSize


def compute_input_size(cfg, tileInfo):
    """Max GR offset + 1 across all threads."""
    max_off = 0
    for tid in range(NUM_THREADS):
        offsets = compute_expected_scale_gr_offset(tid, cfg, tileInfo)
        max_off = max(max_off, offsets[0])
    return max_off + 1


def generate_input_data(size):
    """Deterministic byte array for scale input."""
    return np.array([(i * 7 + 13) & 0xFF for i in range(size)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# ASM generation
# ---------------------------------------------------------------------------
def generate_scale_asm(cfg):
    """Run production GR + LR offset computation on the same writer."""
    init_rocisa()
    writer, kernel, tileInfoA, tileInfoB = create_writer_for_gpu(cfg)
    gra_module = graTileAssignmentScaleSwizzled(writer, kernel)
    lra_module = lraTileAssignmentScaleSwizzled(writer, kernel)
    combined_asm = str(gra_module) + "\n" + str(lra_module)
    return combined_asm, tileInfoA, tileInfoB, kernel


def generate_roundtrip_kernel(test_asm, tileInfoA, tileInfoB, cfg, tc, kernel):
    """Generate complete kernel asm for scale roundtrip test on one matrix."""
    tileInfo = tileInfoA if tc == 'A' else tileInfoB
    grOffReg = tileInfo.sharedVgprGROffset[0]
    lrOffReg = tileInfo.sharedVgprLROffset[0]

    # Input pointer SGPRs: A -> s[4:5], B -> s[6:7]
    ptrLo = 4 if tc == 'A' else 6
    ptrHi = 5 if tc == 'A' else 7

    # LDS sizes
    dataLdsSize, scaleALdsSize, scaleBLdsSize = compute_lds_sizes(
        cfg, tileInfoA, tileInfoB, kernel)
    scaleBase = dataLdsSize if tc == 'A' else (dataLdsSize + scaleALdsSize)
    lds_bytes = scaleALdsSize + scaleBLdsSize

    # Find highest register indices used by test_asm
    vgpr_indices = set(int(m) for m in re.findall(r'\bv(\d+)\b', test_asm))
    sgpr_indices = set(int(m) for m in re.findall(r'\bs(\d+)\b', test_asm))

    # Allocate temporary VGPRs (addr pair must be 64-bit aligned)
    base_v = max(vgpr_indices | {0}) + 1
    if base_v % 2 != 0:
        base_v += 1  # align to even for flat_load address pair
    vAddr0 = base_v
    vAddr1 = base_v + 1
    vData = base_v + 2
    vLrAdj = base_v + 3
    vByteOff = base_v + 4
    max_vgpr = base_v + 5
    max_vgpr = max(((max_vgpr + 3) // 4) * 4, 4)

    base_s = max(sgpr_indices | {11}) + 1
    sTmp = base_s
    max_sgpr = base_s + 1

    return f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{GFX_TARGET}"

.set vgprSerial, 0
.set sgprStrideA0I, 10
.set sgprStrideB1J, 11

.text
.protected test_kernel
.globl test_kernel
.p2align 8
.type test_kernel,@function

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset {max_vgpr}
  .amdhsa_next_free_vgpr {max_vgpr}
  .amdhsa_next_free_sgpr {max_sgpr}
  .amdhsa_group_segment_fixed_size {lds_bytes}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.text
test_kernel:
  // ---- Prologue: Load kernel arguments ----
  s_load_dwordx4 s[4:7], s[0:1], 0x00    // input_A_ptr(s4:5), input_B_ptr(s6:7)
  s_load_dwordx4 s[8:11], s[0:1], 0x10   // output_ptr(s8:9), strideA(s10), strideB(s11)
  s_waitcnt lgkmcnt(0)

  // ---- Generated GR + LR offset code ----
{test_asm}

  // ---- Roundtrip for scale {tc} ----

  // Step 1: Compute flat address = input_ptr + GR_offset
  v_mov_b32 v{vAddr0}, s{ptrLo}
  v_mov_b32 v{vAddr1}, s{ptrHi}
  v_add_co_u32 v{vAddr0}, vcc, v{vAddr0}, v{grOffReg}
  v_addc_co_u32 v{vAddr1}, vcc, v{vAddr1}, 0, vcc

  // Step 2: Load 1 byte from global memory
  flat_load_ubyte v{vData}, v[{vAddr0}:{vAddr1}]
  s_waitcnt vmcnt(0) lgkmcnt(0)

  // Step 3: Write to LDS at position = serial
  ds_write_b8 v0, v{vData}
  s_waitcnt lgkmcnt(0)
  s_barrier

  // Step 4: Read from LDS at adjusted LR offset (subtract dataLdsSize)
  s_mov_b32 s{sTmp}, {scaleBase}
  v_sub_u32 v{vLrAdj}, v{lrOffReg}, s{sTmp}
  ds_read_u8 v{vData}, v{vLrAdj}
  s_waitcnt lgkmcnt(0)

  // Step 5: Export result to output[serial]
  v_lshlrev_b32 v{vByteOff}, 2, v0
  global_store_dword v{vByteOff}, v{vData}, s[8:9]
  s_waitcnt vmcnt(0)
  s_endpgm

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: test_kernel
    .symbol: 'test_kernel.kd'
    .language: OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            input_scale_A_ptr
        .size:            8
        .offset:          0
        .value_kind:      global_buffer
        .value_type:      u8
        .address_space:   global
      - .name:            input_scale_B_ptr
        .size:            8
        .offset:          8
        .value_kind:      global_buffer
        .value_type:      u8
        .address_space:   global
      - .name:            output_ptr
        .size:            8
        .offset:          16
        .value_kind:      global_buffer
        .value_type:      u32
        .address_space:   global
      - .name:            strideA
        .size:            4
        .offset:          24
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB
        .size:            4
        .offset:          28
        .value_kind:      by_value
        .value_type:      u32
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .group_segment_fixed_size: {lds_bytes}
    .private_segment_fixed_size: 0
    .wavefront_size: {WAVESIZE}
    .sgpr_count: {max_sgpr}
    .vgpr_count: {max_vgpr}
    .max_flat_workgroup_size: {NUM_THREADS}
...
.end_amdgpu_metadata
""", lds_bytes


# ---------------------------------------------------------------------------
# GPU launch
# ---------------------------------------------------------------------------
def run_roundtrip_on_gpu(co_path, input_a, input_b, cfg, lds_bytes):
    """Launch roundtrip kernel with 3-buffer kernarg and LDS."""
    hip_check(hip.hipInit(0))

    module = hip_check(hip.hipModuleLoad(co_path.encode()))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"test_kernel"))

    # Allocate device buffers
    d_input_a = hip_check(hip.hipMalloc(len(input_a)))
    d_input_b = hip_check(hip.hipMalloc(len(input_b)))
    out_size = NUM_THREADS * 4
    d_output = hip_check(hip.hipMalloc(out_size))

    # Upload input data
    hip_check(hip.hipMemcpyHtoD(d_input_a, input_a.tobytes(), len(input_a)))
    hip_check(hip.hipMemcpyHtoD(d_input_b, input_b.tobytes(), len(input_b)))
    hip_check(hip.hipMemset(d_output, 0, out_size))

    # Kernarg struct
    class KernelArgs(ctypes.Structure):
        _fields_ = [
            ("input_a_ptr", ctypes.c_uint64),
            ("input_b_ptr", ctypes.c_uint64),
            ("output_ptr", ctypes.c_uint64),
            ("stride_a", ctypes.c_uint32),
            ("stride_b", ctypes.c_uint32),
        ]

    kargs = KernelArgs(int(d_input_a), int(d_input_b), int(d_output),
                       cfg.stride_a, cfg.stride_b)
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
        1, 1, 1,
        NUM_THREADS, 1, 1,
        lds_bytes,
        None,
        None,
        extra
    ))
    hip_check(hip.hipDeviceSynchronize())

    h_out = bytearray(out_size)
    hip_check(hip.hipMemcpyDtoH(h_out, d_output, out_size))

    hip_check(hip.hipFree(d_input_a))
    hip_check(hip.hipFree(d_input_b))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipModuleUnload(module))

    return struct.unpack(f"{NUM_THREADS}I", h_out)


# ---------------------------------------------------------------------------
# Python reference
# ---------------------------------------------------------------------------
def compute_expected_roundtrip(cfg, tileInfoA, tileInfoB, input_data, tc, kernel):
    """Compute expected scale byte for each thread after the roundtrip.

    Data path: thread T reads LDS[lrOff(T) - scaleBase]. That position
    was written by thread writer=lrOff(T)-scaleBase, who loaded
    input[grOff(writer)].
    """
    tileInfo = tileInfoA if tc == 'A' else tileInfoB
    otherTileInfo = tileInfoB if tc == 'A' else tileInfoA

    dataLdsSize, scaleALdsSize, _ = compute_lds_sizes(cfg, tileInfoA, tileInfoB, kernel)
    scaleBase = dataLdsSize if tc == 'A' else (dataLdsSize + scaleALdsSize)

    expected = [0] * NUM_THREADS
    for T in range(NUM_THREADS):
        lr_offset = compute_expected_scale_lr_offset(T, cfg, tileInfo, otherTileInfo)[0]
        writer = lr_offset - scaleBase

        assert 0 <= writer < NUM_THREADS, \
            f"Thread {T}: LR offset {lr_offset} - scaleBase {scaleBase} = {writer} out of range"

        gr_offset = compute_expected_scale_gr_offset(writer, cfg, tileInfo)[0]

        assert 0 <= gr_offset < len(input_data), \
            f"Writer thread {writer}: GR offset {gr_offset} >= input size {len(input_data)}"

        expected[T] = int(input_data[gr_offset])

    return expected


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_and_run_roundtrip(cfg, tc, tmp_path, debug=False):
    """Generate, assemble, run roundtrip for one matrix; return (results, expected)."""
    sys.stdout.flush()

    test_asm, tileInfoA, tileInfoB, kernel = generate_scale_asm(cfg)
    kernel_asm, lds_bytes = generate_roundtrip_kernel(test_asm, tileInfoA, tileInfoB, cfg, tc, kernel)

    if debug:
        print(f"\n--- Kernel ASM (scale {tc}, {cfg.label}) ---")
        print(kernel_asm)
        print("--- End ---\n")

    tileInfo = tileInfoA if tc == 'A' else tileInfoB
    input_size = compute_input_size(cfg, tileInfo)
    input_data = generate_input_data(input_size)

    # Use same input for both A and B (the kernel only reads one based on tc)
    other_tileInfo = tileInfoB if tc == 'A' else tileInfoA
    other_input_size = compute_input_size(cfg, other_tileInfo)
    other_input = generate_input_data(other_input_size)

    if tc == 'A':
        input_a, input_b = input_data, other_input
    else:
        input_a, input_b = other_input, input_data

    label = f"scale_roundtrip_{tc}_{cfg.label}"
    co_path = str(tmp_path / f"{label}.co")
    asm_path = str(tmp_path / f"{label}.s")
    with open(asm_path, "w") as f:
        f.write(kernel_asm)
    assemble_kernel(kernel_asm, co_path)

    results = run_roundtrip_on_gpu(co_path, input_a, input_b, cfg, lds_bytes)
    expected = compute_expected_roundtrip(cfg, tileInfoA, tileInfoB, input_data, tc, kernel)

    return results, expected


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------
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
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Scale GR-LR roundtrip GPU test")
    parser.add_argument("--debug", action="store_true", help="Print kernel asm")
    parser.add_argument("--config", type=int, default=None, help="Config index (default: all)")
    parser.add_argument("--tc", default="AB", help="Matrix to test: A, B, or AB (default)")
    args = parser.parse_args()

    if not HAS_HIP:
        print("HIP not available")
        sys.exit(1)

    configs = SCALE_ROUNDTRIP_CONFIGS if args.config is None else [SCALE_ROUNDTRIP_CONFIGS[args.config]]
    tc_list = list(args.tc)
    total_errors = 0

    for cfg in configs:
        for tc in tc_list:
            print(f"\n{'='*50}")
            print(f"Config: {cfg.label}, matrix: {tc}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()
                results, expected = build_and_run_roundtrip(cfg, tc, tmp_path, debug=args.debug)

                errors = 0
                for tid in range(NUM_THREADS):
                    if results[tid] != expected[tid]:
                        errors += 1
                        if errors <= 8 or args.debug:
                            print(f"  MISMATCH tid={tid}: got {results[tid]}, expected {expected[tid]}")

                if errors == 0:
                    print(f"  PASS")
                else:
                    print(f"  FAIL: {errors} mismatches")
                    total_errors += errors

    print(f"\n{'='*50}")
    print(f"{'PASSED' if total_errors == 0 else f'FAILED ({total_errors} errors)'}")
    sys.exit(0 if total_errors == 0 else 1)
