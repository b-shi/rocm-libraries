#!/usr/bin/env python3
################################################################################
# Debug VGPR Viewer
#
# Reads tensor_D.bin (raw binary dump from graTileAssignment debug kernels)
# and displays per-thread VGPR values as a wave x lane matrix.
#
# When DumpTensors: True is set in test.yaml, the kernel writes each thread's
# computed offset into D[tid]. This viewer reorganizes that flat 32x32 matrix
# into a 4-row x 64-column view matching the GPU's wave/lane structure.
#
# Usage:
#   python debug_vgpr_viewer.py tensor_D.bin
#   python debug_vgpr_viewer.py tensor_D.bin --validate --stride-a 256
#   python debug_vgpr_viewer.py tensor_D.bin --dtype f32 --num-threads 256 --wavesize 64
################################################################################

import argparse
import struct
import sys
import os


def read_tensor_bin(path, dtype="f32"):
    """Read raw binary file as flat array of floats/ints.

    Args:
        path: Path to binary file.
        dtype: Data type - f32, f16, i32.

    Returns:
        List of numeric values.
    """
    fmt_map = {
        "f32": ("<f", 4),
        "f16": ("<e", 2),
        "i32": ("<i", 4),
    }
    if dtype not in fmt_map:
        print(f"Error: unsupported dtype '{dtype}'. Use one of: {list(fmt_map.keys())}")
        sys.exit(1)

    fmt_char, elem_size = fmt_map[dtype]

    file_size = os.path.getsize(path)
    num_elems = file_size // elem_size
    if file_size % elem_size != 0:
        print(f"Warning: file size {file_size} is not a multiple of element size {elem_size}")

    with open(path, "rb") as f:
        raw = f.read()

    values = []
    for i in range(num_elems):
        val = struct.unpack(fmt_char, raw[i * elem_size:(i + 1) * elem_size])[0]
        values.append(val)

    return values


def display_wave_lane_matrix(values, num_threads=256, wavesize=64, source_name="tensor_D.bin"):
    """Display values as a wave x lane matrix.

    Args:
        values: Flat list of per-thread values (indexed by tid).
        num_threads: Total number of threads in the workgroup.
        wavesize: Number of lanes per wave.
        source_name: Filename for display header.
    """
    num_waves = num_threads // wavesize

    print(f"VGPR debug export ({num_threads} threads, {num_waves} waves x {wavesize} lanes)")
    print(f"Source: {source_name}")
    print()

    def fmt_val(v):
        if v == int(v):
            return str(int(v))
        return f"{v:.1f}"

    # Build formatted cell values and determine column widths
    label_w = max(len(f"wave{num_waves - 1}"), 4)
    col_widths = []
    cells = []  # cells[wave][lane]
    for lane in range(wavesize):
        w = len(str(lane))
        for wave in range(num_waves):
            tid = wave * wavesize + lane
            s = fmt_val(values[tid]) if tid < len(values) else "N/A"
            w = max(w, len(s))
        col_widths.append(w)

    for wave in range(num_waves):
        row = []
        for lane in range(wavesize):
            tid = wave * wavesize + lane
            row.append(fmt_val(values[tid]) if tid < len(values) else "N/A")
        cells.append(row)

    # Build horizontal rules
    def hrule(left, mid, right):
        parts = [left, "─" * (label_w + 2)]
        for lane in range(wavesize):
            parts.append(mid)
            parts.append("─" * (col_widths[lane] + 2))
        parts.append(right)
        return "".join(parts)

    top    = hrule("┌", "┬", "┐")
    sep    = hrule("├", "┼", "┤")
    bottom = hrule("└", "┴", "┘")

    # Header row
    hdr = f"│ {'':>{label_w}} │"
    for lane in range(wavesize):
        hdr += f" {str(lane):>{col_widths[lane]}} │"

    print(top)
    print(hdr)
    print(sep)

    # Data rows
    for wave in range(num_waves):
        row = f"│ {'wave' + str(wave):>{label_w}} │"
        for lane in range(wavesize):
            row += f" {cells[wave][lane]:>{col_widths[lane]}} │"
        print(row)

    print(bottom)


def validate_offsets(values, num_threads=256, wavesize=64, stride_a=256):
    """Compare values against expected graTileAssignment offsets.

    The expected offset pattern depends on the tile assignment logic.
    For the common case with MT0=256, DepthU=64, BPE=2, LoadWidth=16:
      block_size = (DepthU * BPE) / LoadWidth = (64 * 2) / 16 = 8
      block_id = tid / block_size = tid / 8
      block_offset = tid % block_size = tid % 8
      offset_a = block_id * stride_a + block_offset * LoadWidth

    Args:
        values: Flat list of per-thread values.
        num_threads: Total threads.
        wavesize: Lanes per wave.
        stride_a: Stride of matrix A in elements.
    """
    # Default parameters matching test_graTileAssignment_gpu.py
    depth_u = 64
    bpe = 2
    load_width = 16
    block_size = (depth_u * bpe) // load_width  # 8

    num_waves = num_threads // wavesize
    mismatches = 0
    total_checked = min(num_threads, len(values))

    print(f"\nValidation (stride_a={stride_a}, block_size={block_size}):")
    print(f"  Expected: offset = (tid / {block_size}) * {stride_a} + (tid % {block_size}) * {load_width}")
    print()

    for tid in range(total_checked):
        block_id = tid // block_size
        block_off = tid % block_size
        expected = block_id * stride_a + block_off * load_width
        actual = values[tid]

        if actual != expected:
            wave = tid // wavesize
            lane = tid % wavesize
            mismatches += 1
            if mismatches <= 16:
                print(f"  MISMATCH wave{wave} lane{lane:2d} (tid={tid:3d}): "
                      f"got {actual:>8}, expected {expected:>8}")

    if mismatches > 16:
        print(f"  ... and {mismatches - 16} more mismatches")

    if mismatches == 0:
        print(f"  All {total_checked} values match expected offsets.")
    else:
        print(f"\n  {mismatches}/{total_checked} mismatches found.")


def main():
    parser = argparse.ArgumentParser(
        description="Debug VGPR viewer: display tensor_D.bin as wave x lane matrix")
    parser.add_argument("binfile", help="Path to tensor_D.bin (raw binary, no header)")
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "i32"],
                        help="Data type of elements (default: f32)")
    parser.add_argument("--num-threads", type=int, default=256,
                        help="Number of threads in workgroup (default: 256)")
    parser.add_argument("--wavesize", type=int, default=64,
                        help="Lanes per wave (default: 64)")
    parser.add_argument("--validate", action="store_true",
                        help="Compare values against expected graTileAssignment offsets")
    parser.add_argument("--stride-a", type=int, default=256,
                        help="Stride of matrix A in elements, for validation (default: 256)")

    args = parser.parse_args()

    if not os.path.isfile(args.binfile):
        print(f"Error: file not found: {args.binfile}")
        sys.exit(1)

    values = read_tensor_bin(args.binfile, args.dtype)
    source_name = os.path.basename(args.binfile)

    display_wave_lane_matrix(values, args.num_threads, args.wavesize, source_name)

    if args.validate:
        validate_offsets(values, args.num_threads, args.wavesize, args.stride_a)


if __name__ == "__main__":
    main()
