// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <mxDataGen.hpp>

#include <cstdint>
#include <vector>

/**
 * @brief Returns true if a 4-bit FP4 E2M1 nibble represents zero.
 *
 * FP4 E2M1 values are packed two-per-byte (low nibble first).
 * Both 0x0 (+0) and 0x8 (-0) decode to zero.
 */
static bool isZeroNibble(uint8_t nibble)
{
    // FP4 E2M1: 0x0 = +0.0, 0x8 = -0.0
    return (nibble == 0x0) || (nibble == 0x8);
}

/**
 * @brief Count elements that decode to zero in a packed FP4 buffer.
 */
static size_t countZerosFP4(const uint8_t* packedData, size_t numPackedBytes)
{
    size_t zeros = 0;
    for(size_t i = 0; i < numPackedBytes; ++i)
    {
        uint8_t lo = packedData[i] & 0x0F;
        uint8_t hi = (packedData[i] >> 4) & 0x0F;
        if(isZeroNibble(lo))
            ++zeros;
        if(isZeroNibble(hi))
            ++zeros;
    }
    return zeros;
}

class MXDataGenFP4Test : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool>>
{
};

/**
 * @brief Verify that generateMXInput produces FP4 data with an acceptable zero frequency.
 *
 * FP4 E2M1 has 16 nibble values, 2 of which are zero (0x0 = +0, 0x8 = -0), giving a
 * naive baseline of 2/16 = 12.5%. MX block scaling slightly elevates this: the block
 * maximum is guaranteed non-zero, pushing small elements toward zero. Empirically the
 * zero frequency converges to ~12.89% for large matrices with bounded [-1, 1] input.
 */
TEST_P(MXDataGenFP4Test, ZeroFrequencyWithinBounds)
{
    auto [rows, cols, mxBlock, isTranspose] = GetParam();

    const uint64_t numElements  = rows * cols;
    const uint64_t numPacked    = (numElements + 1) / 2;
    const size_t   numScales    = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataBuffer(numPacked, 0);
    std::vector<uint8_t> scaleBuffer(numScales, 0);

    std::vector<size_t> emptySwizzle;
    std::vector<size_t> emptyTile;

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    dataBuffer.data(),
                    scaleBuffer.data(),
                    rows,
                    cols,
                    rows, // stride = rows (column-major)
                    isTranspose,
                    emptySwizzle,
                    emptyTile,
                    mxBlock,
                    1,
                    true,
                    "Bounded",
                    -1.0f,
                    1.0f);

    size_t zeros       = countZerosFP4(dataBuffer.data(), numPacked);
    double zeroPercent = 100.0 * static_cast<double>(zeros) / static_cast<double>(numElements);

    EXPECT_LT(zeroPercent, 13.0)
        << "Zero frequency " << zeroPercent << "% exceeds 13% upper bound for "
        << rows << "x" << cols << " FP4 matrix (transpose=" << isTranspose << ")";

    // Ensure non-trivial data was actually generated (not all zeros)
    EXPECT_GT(numElements - zeros, 0u)
        << "All elements are zero for " << rows << "x" << cols << " FP4 matrix";
}

INSTANTIATE_TEST_SUITE_P(
    FP4ZeroFrequency,
    MXDataGenFP4Test,
    ::testing::Values(
        // rows, cols, mxBlock, isTranspose
        std::make_tuple(128u,  128u,  32, true),
        std::make_tuple(256u,  256u,  32, true),
        std::make_tuple(2048u, 1026u, 32, true),
        std::make_tuple(2048u, 514u,  32, false)
    )
);

/**
 * @brief Regression guard: generateMXInput must be deterministic (fixed seed).
 *
 * Any post-generation overwrite of the MXSA/MXSB buffers (e.g., the general
 * tensor-init loop in initializeCPUInputs) desynchronises the CPU reference
 * from GPU data, causing intermittent single-element validation failures.
 * rows=K (must be mxBlock-aligned), cols=M/N (need not be).
 */
class MXGeneratorDeterminismTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool, bool>>
{
};

TEST_P(MXGeneratorDeterminismTest, GeneratorOutputIsDeterministic)
{
    auto [rows, cols, mxBlock, isTranspose, isMatrixA] = GetParam();

    const size_t numPacked = (rows * cols + 1) / 2;
    const size_t numScales = (rows / mxBlock) * cols;

    std::vector<uint8_t> data1(numPacked);
    std::vector<uint8_t> data2(numPacked);
    std::vector<uint8_t> scale1(numScales, 0x00);
    std::vector<uint8_t> scale2(numScales, 0xFF); // sentinel: catches no-write if scale1==scale2 passes

    std::vector<size_t> emptySwizzle, emptyTile;

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    data1.data(), scale1.data(),
                    rows, cols, rows, isTranspose,
                    emptySwizzle, emptyTile,
                    mxBlock, 1, isMatrixA, "Bounded", -1.f, 1.f);

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    data2.data(), scale2.data(),
                    rows, cols, rows, isTranspose,
                    emptySwizzle, emptyTile,
                    mxBlock, 1, isMatrixA, "Bounded", -1.f, 1.f);

    EXPECT_EQ(data1, data2)
        << "FP4 data is non-deterministic";
    EXPECT_EQ(scale1, scale2)
        << "Scale data is non-deterministic; any post-generation overwrite will corrupt validation";

    bool allZero = std::all_of(scale1.begin(), scale1.end(), [](uint8_t b){ return b == 0; });
    bool allOnes = std::all_of(scale1.begin(), scale1.end(), [](uint8_t b){ return b == 0xFF; });
    EXPECT_FALSE(allZero) << "Scale buffer is all-zero — generator did not write";
    EXPECT_FALSE(allOnes) << "Scale buffer is all-0xFF (max UE8M0 value) — generator likely failed; bounded [-1,1] input should produce varied scales";
}

INSTANTIATE_TEST_SUITE_P(
    GeneratorDeterminism,
    MXGeneratorDeterminismTest,
    ::testing::Values(
        // rows=K, cols=M or N  (tensorA.sizes()={K,M}, tensorB.sizes()={K,N})
        std::make_tuple(1024u, 128u, 32, true,  true),  // transposed A
        std::make_tuple(1024u, 128u, 32, false, false), // non-transposed B
        std::make_tuple(1024u, 204u, 32, true,  true),  // M=204, non-32-aligned (was failing)
        std::make_tuple(1024u, 213u, 32, true,  true)   // M=213, non-32-aligned (was failing)
    )
);

// ============================================================================
// PreSwizzle scale tests
//
// Verify generateMXInput with preSwizzle produces scale data that is a
// permutation of the unswizzled layout. gfx950 FP4 MX kernels expect:
//   preSwizzle = {swizzleTileMN=32, tileK=8, subTileK=MiK/mxBlock}
//   preTile    = {tileK=8, swizzleTileMN=32}
// swizzleTileMN=32 is fixed (2 SIMDs * 16 lanes); subTileK=4 for MiK=128, mxBlock=32.
// ============================================================================

// Params: {rows, cols, mxBlock, isTranspose, isMatrixA}
class MXPreSwizzleTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool, bool>>
{
};

/** @brief Verify preSwizzle produces a non-trivial permutation of scale data. */
TEST_P(MXPreSwizzleTest, ScaleIsPermutationOfUnswizzled)
{
    auto [rows, cols, mxBlock, isTranspose, isMatrixA] = GetParam();

    const std::vector<size_t> preSwizzle = {32, 8, 4};
    const std::vector<size_t> preTile    = {8, 32};

    const uint64_t numElements  = rows * cols;
    const uint64_t numPacked    = (numElements + 1) / 2;
    const size_t   numScales    = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataNoShuf(numPacked, 0);
    std::vector<uint8_t> scaleNoShuf(numScales, 0);
    std::vector<uint8_t> dataShuf(numPacked, 0);
    std::vector<uint8_t> scaleShuf(numScales, 0);

    // Generate without preSwizzle
    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    dataNoShuf.data(),
                    scaleNoShuf.data(),
                    rows, cols, rows,
                    isTranspose,
                    {}, {},
                    mxBlock, 1, isMatrixA,
                    "Bounded", -1.0f, 1.0f);

    // Generate with preSwizzle
    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    dataShuf.data(),
                    scaleShuf.data(),
                    rows, cols, rows,
                    isTranspose,
                    preSwizzle, preTile,
                    mxBlock, 1, isMatrixA,
                    "Bounded", -1.0f, 1.0f);

    // The scale buffers must be different
    EXPECT_NE(scaleNoShuf, scaleShuf)
        << "Scale data was not shuffled for " << rows << "x" << cols
        << " (transpose=" << isTranspose << ", isMatrixA=" << isMatrixA << ")";

    // The shuffled scale must be a permutation: same multiset of bytes
    std::vector<uint8_t> sortedNoShuf = scaleNoShuf;
    std::vector<uint8_t> sortedShuf   = scaleShuf;
    std::sort(sortedNoShuf.begin(), sortedNoShuf.end());
    std::sort(sortedShuf.begin(), sortedShuf.end());
    EXPECT_EQ(sortedNoShuf, sortedShuf)
        << "Pre-shuffled scale is not a permutation of the unshuffled scale for "
        << rows << "x" << cols;

    // Data buffer must be identical (preSwizzle only affects scale, not data)
    EXPECT_EQ(dataNoShuf, dataShuf)
        << "Data buffer changed unexpectedly with preSwizzle for "
        << rows << "x" << cols;
}

INSTANTIATE_TEST_SUITE_P(
    FP4PreSwizzle,
    MXPreSwizzleTest,
    ::testing::Values(
        // rows, cols, mxBlock, isTranspose, isMatrixA
        // Test size constraints for preSwizzle {32,8,4} + preTile {8,32}:
        //   rows % 256 == 0  (scaleRows = rows/mxBlock must be divisible by tileK=8)
        //   cols % 32  == 0  (scaleCols must be divisible by swizzleTileMN=32)        std::make_tuple(256u,  256u,  32, true,  true),   // scale A transposed
        std::make_tuple(256u,  256u,  32, false, false),  // scale B non-transposed
        std::make_tuple(512u,  256u,  32, true,  true),   // larger scale A
        std::make_tuple(256u,  512u,  32, false, false),  // larger scale B
        std::make_tuple(4096u, 16384u, 32, true, true)    // benchmark-scale problem
    )
);

// ============================================================================
// ScaleInitMode tests
//
// Verify that the scaleInitMethod parameter to generateMXInput correctly
// produces the expected scale patterns independent of data initialization.
// Scale layout: column-major, scaleRows = rows/mxBlock, scaleCols = cols.
// Tile dimensions: 32 rows x 8 cols.
// ============================================================================

static constexpr size_t kScaleTileRows = 32;
static constexpr size_t kScaleTileCols = 8;

// Helper: generate FP4 data+scale with a given scaleInitMethod
static void generateWithScaleInit(std::vector<uint8_t>& dataOut,
                                  std::vector<uint8_t>& scaleOut,
                                  uint64_t              rows,
                                  uint64_t              cols,
                                  int                   mxBlock,
                                  std::string_view      scaleInitMethod,
                                  int                   scaleBlockI = 0,
                                  int                   scaleBlockJ = 0)
{
    const size_t numPacked = (rows * cols + 1) / 2;
    const size_t numScales = (rows / mxBlock) * cols;

    dataOut.assign(numPacked, 0);
    scaleOut.assign(numScales, 0xAA); // sentinel to detect untouched bytes

    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    dataOut.data(),
                    scaleOut.data(),
                    rows, cols, rows,
                    true, // isTranspose
                    {}, {},
                    mxBlock, 1, true,
                    "Bounded", -1.0f, 1.0f,
                    scaleInitMethod,
                    scaleBlockI,
                    scaleBlockJ);
}

// Helper: get scale value at (r, c) in column-major layout
static uint8_t scaleAt(const std::vector<uint8_t>& scale,
                       size_t                       r,
                       size_t                       c,
                       size_t                       scaleRows)
{
    return scale[r + c * scaleRows];
}

// ---------------------------------------------------------------------------
// ScaleBlockSerial: each 32x8 tile gets an incrementing constant byte
// ---------------------------------------------------------------------------
class MXScaleBlockSerialTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int>>
{
};

TEST_P(MXScaleBlockSerialTest, TilesGetIncrementingValues)
{
    auto [rows, cols, mxBlock] = GetParam();

    std::vector<uint8_t> data, scale;
    generateWithScaleInit(data, scale, rows, cols, mxBlock, "MXScaleBlockSerial");

    const size_t scaleRows    = rows / mxBlock;
    const size_t scaleCols    = cols;
    const size_t numTileRows  = (scaleRows + kScaleTileRows - 1) / kScaleTileRows;
    const size_t numTileCols  = (scaleCols + kScaleTileCols - 1) / kScaleTileCols;

    uint8_t expectedValue = 0;
    for(size_t tr = 0; tr < numTileRows; tr++)
    {
        for(size_t tc = 0; tc < numTileCols; tc++)
        {
            size_t rEnd = std::min((tr + 1) * kScaleTileRows, scaleRows);
            size_t cEnd = std::min((tc + 1) * kScaleTileCols, scaleCols);

            for(size_t r = tr * kScaleTileRows; r < rEnd; r++)
            {
                for(size_t c = tc * kScaleTileCols; c < cEnd; c++)
                {
                    EXPECT_EQ(scaleAt(scale, r, c, scaleRows), expectedValue)
                        << "Mismatch at scale(" << r << "," << c
                        << ") tile(" << tr << "," << tc << ")";
                }
            }
            expectedValue = (expectedValue >= 0xFE) ? 0 : expectedValue + 1;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    ScaleBlockSerial,
    MXScaleBlockSerialTest,
    ::testing::Values(
        // rows, cols, mxBlock
        std::make_tuple(1024u, 256u, 32),   // 1 tile row, 32 tile cols
        std::make_tuple(2048u, 64u,  32),   // 2 tile rows, 8 tile cols
        std::make_tuple(256u,  256u, 32)    // small matrix
    )
);

// ---------------------------------------------------------------------------
// ScaleSparseBlock: all zeros except one tile filled with 0x7F
// ---------------------------------------------------------------------------
class MXScaleSparseBlockTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, int, int>>
{
};

TEST_P(MXScaleSparseBlockTest, OneTileIs0x7FRestZero)
{
    auto [rows, cols, mxBlock, blockI, blockJ] = GetParam();

    std::vector<uint8_t> data, scale;
    generateWithScaleInit(data, scale, rows, cols, mxBlock,
                          "MXScaleSparseBlock", blockI, blockJ);

    const size_t scaleRows   = rows / mxBlock;
    const size_t scaleCols   = cols;
    const size_t numTileRows = scaleRows / kScaleTileRows;
    const size_t numTileCols = scaleCols / kScaleTileCols;

    // Clamp target tile indices (matching mxDataGenerator logic)
    size_t targetI = (blockI >= 0) ? static_cast<size_t>(blockI) : 0;
    size_t targetJ = (blockJ >= 0) ? static_cast<size_t>(blockJ) : 0;
    if(targetI >= numTileRows)
        targetI = (numTileRows > 0) ? numTileRows - 1 : 0;
    if(targetJ >= numTileCols)
        targetJ = (numTileCols > 0) ? numTileCols - 1 : 0;

    size_t rStart = targetI * kScaleTileRows;
    size_t cStart = targetJ * kScaleTileCols;
    size_t rEnd   = std::min(rStart + kScaleTileRows, scaleRows);
    size_t cEnd   = std::min(cStart + kScaleTileCols, scaleCols);

    size_t countZero = 0;
    size_t count7F   = 0;

    for(size_t c = 0; c < scaleCols; c++)
    {
        for(size_t r = 0; r < scaleRows; r++)
        {
            uint8_t val      = scaleAt(scale, r, c, scaleRows);
            bool    inTarget = (r >= rStart && r < rEnd && c >= cStart && c < cEnd);

            if(inTarget)
            {
                EXPECT_EQ(val, 0x7F)
                    << "Target tile element at (" << r << "," << c << ") should be 0x7F";
                count7F++;
            }
            else
            {
                EXPECT_EQ(val, 0x00)
                    << "Non-target element at (" << r << "," << c << ") should be 0x00";
                countZero++;
            }
        }
    }

    EXPECT_GT(count7F, 0u) << "No elements found in target tile";
    EXPECT_GT(countZero, 0u) << "No zero elements outside target tile";
}

INSTANTIATE_TEST_SUITE_P(
    ScaleSparseBlock,
    MXScaleSparseBlockTest,
    ::testing::Values(
        // rows, cols, mxBlock, blockI, blockJ
        std::make_tuple(1024u, 64u, 32, 0, 0),   // first tile
        std::make_tuple(1024u, 64u, 32, 0, 3),   // different column tile
        std::make_tuple(2048u, 64u, 32, 1, 0),   // second tile row
        std::make_tuple(1024u, 64u, 32, 99, 99)  // out-of-range (clamped)
    )
);

// ---------------------------------------------------------------------------
// ScaleSparseBlockRandom: all zeros except one tile with random 0x00 or 0x7F
// ---------------------------------------------------------------------------
class MXScaleSparseBlockRandomTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, int, int>>
{
};

TEST_P(MXScaleSparseBlockRandomTest, OneTileIsRandom0x00Or0x7FRestZero)
{
    auto [rows, cols, mxBlock, blockI, blockJ] = GetParam();

    std::vector<uint8_t> data, scale;
    generateWithScaleInit(data, scale, rows, cols, mxBlock,
                          "MXScaleSparseBlockRandom", blockI, blockJ);

    const size_t scaleRows   = rows / mxBlock;
    const size_t scaleCols   = cols;
    const size_t numTileRows = scaleRows / kScaleTileRows;
    const size_t numTileCols = scaleCols / kScaleTileCols;

    size_t targetI = (blockI >= 0) ? static_cast<size_t>(blockI) : 0;
    size_t targetJ = (blockJ >= 0) ? static_cast<size_t>(blockJ) : 0;
    if(targetI >= numTileRows)
        targetI = (numTileRows > 0) ? numTileRows - 1 : 0;
    if(targetJ >= numTileCols)
        targetJ = (numTileCols > 0) ? numTileCols - 1 : 0;

    size_t rStart = targetI * kScaleTileRows;
    size_t cStart = targetJ * kScaleTileCols;
    size_t rEnd   = std::min(rStart + kScaleTileRows, scaleRows);
    size_t cEnd   = std::min(cStart + kScaleTileCols, scaleCols);

    bool   seenZeroInTile = false;
    bool   seen7FInTile   = false;

    for(size_t c = 0; c < scaleCols; c++)
    {
        for(size_t r = 0; r < scaleRows; r++)
        {
            uint8_t val      = scaleAt(scale, r, c, scaleRows);
            bool    inTarget = (r >= rStart && r < rEnd && c >= cStart && c < cEnd);

            if(inTarget)
            {
                EXPECT_TRUE(val == 0x00 || val == 0x7F)
                    << "Target tile element at (" << r << "," << c
                    << ") = 0x" << std::hex << (int)val
                    << " must be 0x00 or 0x7F";
                if(val == 0x00)
                    seenZeroInTile = true;
                if(val == 0x7F)
                    seen7FInTile = true;
            }
            else
            {
                EXPECT_EQ(val, 0x00)
                    << "Non-target element at (" << r << "," << c << ") should be 0x00";
            }
        }
    }

    // With 256 elements per tile and 50/50 probability, seeing only
    // one value is astronomically unlikely (p < 2^-255).
    EXPECT_TRUE(seenZeroInTile)
        << "Target tile has no 0x00 — random generation may be broken";
    EXPECT_TRUE(seen7FInTile)
        << "Target tile has no 0x7F — random generation may be broken";
}

INSTANTIATE_TEST_SUITE_P(
    ScaleSparseBlockRandom,
    MXScaleSparseBlockRandomTest,
    ::testing::Values(
        // rows, cols, mxBlock, blockI, blockJ
        std::make_tuple(1024u, 64u, 32, 0, 0),
        std::make_tuple(2048u, 64u, 32, 1, 3)
    )
);

// ---------------------------------------------------------------------------
// Scale init does not affect data buffer: FP4 data must be identical
// regardless of which scaleInitMethod is used.
// ---------------------------------------------------------------------------
TEST(MXScaleInitDecoupling, ScaleInitDoesNotAffectData)
{
    const uint64_t rows    = 1024;
    const uint64_t cols    = 64;
    const int      mxBlock = 32;

    std::vector<uint8_t> dataDefault, scaleDefault;
    std::vector<uint8_t> dataSerial,  scaleSerial;
    std::vector<uint8_t> dataSparse,  scaleSparse;

    generateWithScaleInit(dataDefault, scaleDefault, rows, cols, mxBlock, "");
    generateWithScaleInit(dataSerial,  scaleSerial,  rows, cols, mxBlock, "MXScaleBlockSerial");
    generateWithScaleInit(dataSparse,  scaleSparse,  rows, cols, mxBlock, "MXScaleSparseBlock");

    EXPECT_EQ(dataDefault, dataSerial)
        << "ScaleBlockSerial changed the FP4 data buffer";
    EXPECT_EQ(dataDefault, dataSparse)
        << "ScaleSparseBlock changed the FP4 data buffer";

    // Scale buffers must differ from each other
    EXPECT_NE(scaleDefault, scaleSerial)
        << "ScaleBlockSerial produced same scale as default";
    EXPECT_NE(scaleDefault, scaleSparse)
        << "ScaleSparseBlock produced same scale as default";
    EXPECT_NE(scaleSerial, scaleSparse)
        << "ScaleBlockSerial and ScaleSparseBlock produced same scale";
}

// ---------------------------------------------------------------------------
// PreSwizzle + ScaleInitMode: pre-swizzle applied to custom scale patterns
// must produce a permutation of the unswizzled result.
// ---------------------------------------------------------------------------
class MXScaleInitPreSwizzleTest
    : public ::testing::TestWithParam<std::string>
{
};

TEST_P(MXScaleInitPreSwizzleTest, PreSwizzleIsPermutationOfCustomScale)
{
    const std::string scaleInitMethod = GetParam();

    // Size must satisfy preSwizzle constraints:
    //   scaleRows % tileK(8) == 0  →  rows % (mxBlock*8) == 0  →  rows % 256 == 0
    //   scaleCols % swizzleTileMN(32) == 0  →  cols % 32 == 0
    const uint64_t rows    = 1024;
    const uint64_t cols    = 256;
    const int      mxBlock = 32;
    const size_t   numPacked = (rows * cols + 1) / 2;
    const size_t   numScales = (rows / mxBlock) * cols;

    const std::vector<size_t> preSwizzle = {32, 8, 4};
    const std::vector<size_t> preTile    = {8, 32};

    // Without preSwizzle
    std::vector<uint8_t> dataNoShuf(numPacked, 0);
    std::vector<uint8_t> scaleNoShuf(numScales, 0);
    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    dataNoShuf.data(), scaleNoShuf.data(),
                    rows, cols, rows, true,
                    {}, {},
                    mxBlock, 1, true,
                    "Bounded", -1.0f, 1.0f,
                    scaleInitMethod, 0, 0);

    // With preSwizzle
    std::vector<uint8_t> dataShuf(numPacked, 0);
    std::vector<uint8_t> scaleShuf(numScales, 0);
    generateMXInput((hipDataType)HIP_R_4F_E2M1,
                    dataShuf.data(), scaleShuf.data(),
                    rows, cols, rows, true,
                    preSwizzle, preTile,
                    mxBlock, 1, true,
                    "Bounded", -1.0f, 1.0f,
                    scaleInitMethod, 0, 0);

    EXPECT_NE(scaleNoShuf, scaleShuf)
        << "PreSwizzle did not change scale layout for " << scaleInitMethod;

    std::vector<uint8_t> sortedNoShuf = scaleNoShuf;
    std::vector<uint8_t> sortedShuf   = scaleShuf;
    std::sort(sortedNoShuf.begin(), sortedNoShuf.end());
    std::sort(sortedShuf.begin(), sortedShuf.end());
    EXPECT_EQ(sortedNoShuf, sortedShuf)
        << "PreSwizzle is not a permutation of unswizzled scale for " << scaleInitMethod;

    EXPECT_EQ(dataNoShuf, dataShuf)
        << "PreSwizzle changed data buffer for " << scaleInitMethod;
}

INSTANTIATE_TEST_SUITE_P(
    ScaleInitPreSwizzle,
    MXScaleInitPreSwizzleTest,
    ::testing::Values(
        "MXScaleBlockSerial",
        "MXScaleSparseBlock",
        "MXScaleSparseBlockRandom"
    )
);
