import io
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock
from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.SubtileBasedScheduler import SubtileBasedScheduler, SchedulerConfig, PrefetchMode, VGPRTileReUseStrategy, SubgroupOrdering
from rocisa import rocIsa
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType

# Initialize rocIsa for gfx950
ri = rocIsa.getInstance()
if not ri.isInit():
    import shutil
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init((9, 5, 0), asmpath)
ri.setKernel((9, 5, 0), 64)

def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock

def create_kernel(MT0=256, MT1=256, fp4=False):
    mxblock = 32 if fp4 else 0
    bpe = 1 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    depthU = 256 if fp4 else 64
    dtype = _mock_dtype(bpe)
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": _mock_dtype(4),
    }
    if fp4:
        problemType["MXBlockA"] = mxblock
        problemType["MXBlockB"] = mxblock
    kernel = {
        "DepthU": depthU,
        "_DepthUA": depthU,
        "_DepthUB": depthU,
        "MacroTileA": MT0,
        "MacroTileB": MT1,
        "MacroTile0": MT0,
        "MacroTile1": MT1,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": matrixInstK,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "SourceSwap": False,
        "MIArchVgpr": False,
        "ProblemType": problemType,
    }
    if fp4:
        kernel["_DepthUMXSA"] = depthU // mxblock
        kernel["_DepthUMXSB"] = depthU // mxblock
    return kernel

def create_mock_writer(kernel):
    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
    )
    # Allocate D tileInfo (same as KernelWriter line 3843)
    dTileInfo = TileInfo('D', kernel)
    dTileInfo.allocVgprTileRegisters(writer, kernel)
    writer.states.d = SimpleNamespace(tileInfo=dTileInfo)
    return writer

def create_writer_with_tiles(kernel, tiA, tiB, scaleTiA=None, scaleTiB=None):
    writer = create_mock_writer(kernel)
    writer.states.a = SimpleNamespace(tileInfo=tiA)
    writer.states.b = SimpleNamespace(tileInfo=tiB)
    writer.states.mxsa = SimpleNamespace(tileInfo=scaleTiA) if scaleTiA else SimpleNamespace()
    writer.states.mxsb = SimpleNamespace(tileInfo=scaleTiB) if scaleTiB else SimpleNamespace()
    tiA.allocOffsetRegisters(writer, kernel)
    tiB.allocOffsetRegisters(writer, kernel)
    if scaleTiA:
        scaleTiA.allocOffsetRegisters(writer, kernel)
    if scaleTiB:
        scaleTiB.allocOffsetRegisters(writer, kernel)
    return writer


def test_PGR2_64_64_1x1():
    MT0=MT1=64
    kernel = create_kernel(MT0,MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    # 2x2 partition grid
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH,
                          VGPRTileReUseStrategy.ACROSS_SUBGROUP,
                          SubgroupOrdering.COLUMN_MAJOR)
    s = SubtileBasedScheduler(tiA, tiB, cfg)

    assert len(s.preloopSteps)  > 0
    assert len(s.mainloopSteps) > 0
    assert len(s.ngllSteps)     > 0
    assert len(s.nllSteps)      > 0

    writer = create_writer_with_tiles(kernel, tiA, tiB)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.printSchedule()
    actual = buf.getvalue()

    expected = """\
SubtileGridA=2, SubtileGridB=2
Partition grid: 1 x 1
Partition size: 2 x 2
Prefetch: HALF_PREFETCH
Reuse: ACROSS_SUBGROUP
needsUnrolling: False
totalVGPRTiles: 8 (32 VGPRs)
totalScaleVGPRTiles: 0
hasScale: False

Ordering grid (COLUMN_MAJOR):
   0

PRELOOP:
  GR (MT 0):  A: [0, 1]  B: [0, 1]
  GR_INC
  WAIT_GR (MT 0) A: [0, 1]  B: [0, 1] — inflight SubtileLoads A=0 B=0
  SYNC
  LR (MT 0, subIterK 0) A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
  WAIT_LR
  SKIP_IF_LE(1, NLL)
  GR (MT 1):  A: [0, 1]  B: [0, 1]
  GR_INC
  SKIP_IF_LE(2, NGLL)

MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 0), (0, 1), (1, 0), (1, 1)]
        - USING  A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [WaitLROp]
      GR (MT n+2):  A: [0]  B: [0]
        before: [LR(MT n, sik 1), WaitLROp, SyncOp]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 0), (0, 1), (1, 0), (1, 1)]
        - USING  A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [none]
      LR (MT n+1, subIterK 0) A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        before: [WaitGROp, SyncOp, LR_INCOp]  after: [WaitLROp]
      GR (MT n+2):  A: [1]  B: [1]
        before: [none]  after: [GR_INCOp]

NGLL (No Global Load Loop):
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 0), (0, 1), (1, 0), (1, 1)]
        - USING  A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [WaitLROp, WaitLROp, SyncOp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 0), (0, 1), (1, 0), (1, 1)]
        - USING  A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [none]
      LR (MT n+1, subIterK 0) A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        before: [WaitGROp, SyncOp, LR_INCOp]  after: [WaitLROp]

NLL (No Load Loop):
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 0), (0, 1), (1, 0), (1, 1)]
        - USING  A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [WaitLROp, WaitLROp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 0), (0, 1), (1, 0), (1, 1)]
        - USING  A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [none]
"""

    assert actual == expected



def test_PGR2_64_64_2x2():
    MT0=MT1=64
    kernel = create_kernel(MT0,MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    # 2x2 partition grid
    lsgA = tiA.localSubtileGrid[0]//2
    lsgB = tiB.localSubtileGrid[0]//2

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH,
                          VGPRTileReUseStrategy.ACROSS_SUBGROUP,
                          SubgroupOrdering.COLUMN_MAJOR)
    s = SubtileBasedScheduler(tiA, tiB, cfg)

    assert len(s.preloopSteps)  > 0
    assert len(s.mainloopSteps) > 0
    assert len(s.ngllSteps)     > 0
    assert len(s.nllSteps)      > 0

    writer = create_writer_with_tiles(kernel, tiA, tiB)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.printSchedule()
    actual = buf.getvalue()

    expected = """\
SubtileGridA=2, SubtileGridB=2
Partition grid: 2 x 2
Partition size: 1 x 1
Prefetch: HALF_PREFETCH
Reuse: ACROSS_SUBGROUP
needsUnrolling: False
totalVGPRTiles: 8 (32 VGPRs)
totalScaleVGPRTiles: 0
hasScale: False

Ordering grid (COLUMN_MAJOR):
   0   2
   1   3

PRELOOP:
  GR (MT 0):  A: [0]  B: [0]
  GR (MT 0):  A: [1]  B: []
  GR (MT 0):  A: []  B: [1]
  GR_INC
  WAIT_GR (MT 0) A: [0, 1]  B: [0, 1] — inflight SubtileLoads A=0 B=0
  SYNC
  LR (MT 0, subIterK 0) A: {0: 0}  B: {0: 1}
  WAIT_LR
  SKIP_IF_LE(1, NLL)
  GR (MT 1):  A: [0]  B: [0]
  SKIP_IF_LE(2, NGLL)

MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 0)]
        - USING  A: {0: 0}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {0: 2}  B: {0: 3}
        before: [none]  after: [WaitLROp]
      GR (MT n+1):  A: [1]  B: []
        before: [none]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 0)]
        - USING  A: {0: 2}  B: {0: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {1: 4}  B: {}
        before: [WaitGROp, SyncOp]  after: [WaitLROp]
  Partition 1:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(1, 0)]
        - USING  A: {1: 4}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {1: 5}  B: {}
        before: [none]  after: [WaitLROp]
      GR (MT n+1):  A: []  B: [1]
        before: [none]  after: [GR_INCOp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(1, 0)]
        - USING  A: {1: 5}  B: {0: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {}  B: {1: 6}
        before: [WaitGROp, SyncOp]  after: [WaitLROp]
  Partition 2:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 1)]
        - USING  A: {0: 0}  B: {1: 6}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {}  B: {1: 7}
        before: [none]  after: [WaitLROp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 1)]
        - USING  A: {0: 2}  B: {1: 7}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {}  B: {}
        before: [none]  after: [WaitLROp]
  Partition 3:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(1, 1)]
        - USING  A: {1: 4}  B: {1: 6}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {}  B: {}
        before: [none]  after: [WaitLROp]
      GR (MT n+2):  A: [0]  B: [0]
        before: [LR(MT n, sik 1), WaitLROp, SyncOp]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(1, 1)]
        - USING  A: {1: 5}  B: {1: 7}
        before: [none]  after: [none]
      LR (MT n+1, subIterK 0) A: {0: 0}  B: {0: 1}
        before: [GR(MT n+1), WaitGROp, SyncOp, LR_INCOp]  after: [WaitLROp]

NGLL (No Global Load Loop):
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 0)]
        - USING  A: {0: 0}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {0: 2}  B: {0: 3}
        before: [none]  after: [WaitLROp]
      GR (MT n+1):  A: [1]  B: []
        before: [none]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 0)]
        - USING  A: {0: 2}  B: {0: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {1: 4}  B: {}
        before: [WaitGROp, SyncOp]  after: [WaitLROp]
  Partition 1:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(1, 0)]
        - USING  A: {1: 4}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {1: 5}  B: {}
        before: [none]  after: [WaitLROp]
      GR (MT n+1):  A: []  B: [1]
        before: [none]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(1, 0)]
        - USING  A: {1: 5}  B: {0: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {}  B: {1: 6}
        before: [WaitGROp, SyncOp]  after: [WaitLROp]
  Partition 2:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 1)]
        - USING  A: {0: 0}  B: {1: 6}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {}  B: {1: 7}
        before: [none]  after: [WaitLROp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 1)]
        - USING  A: {0: 2}  B: {1: 7}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {}  B: {}
        before: [none]  after: [WaitLROp]
  Partition 3:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(1, 1)]
        - USING  A: {1: 4}  B: {1: 6}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {}  B: {}
        before: [none]  after: [WaitLROp, WaitLROp, SyncOp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(1, 1)]
        - USING  A: {1: 5}  B: {1: 7}
        before: [none]  after: [none]
      LR (MT n+1, subIterK 0) A: {0: 0}  B: {0: 1}
        before: [GR(MT n+1), WaitGROp, SyncOp, LR_INCOp]  after: [WaitLROp]

NLL (No Load Loop):
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 0)]
        - USING  A: {0: 0}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {0: 2}  B: {0: 3}
        before: [none]  after: [WaitLROp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 0)]
        - USING  A: {0: 2}  B: {0: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {1: 4}  B: {}
        before: [WaitGROp, SyncOp]  after: [WaitLROp]
  Partition 1:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(1, 0)]
        - USING  A: {1: 4}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {1: 5}  B: {}
        before: [none]  after: [WaitLROp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(1, 0)]
        - USING  A: {1: 5}  B: {0: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {}  B: {1: 6}
        before: [WaitGROp, SyncOp]  after: [WaitLROp]
  Partition 2:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 1)]
        - USING  A: {0: 0}  B: {1: 6}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {}  B: {1: 7}
        before: [none]  after: [WaitLROp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 1)]
        - USING  A: {0: 2}  B: {1: 7}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: {}  B: {}
        before: [none]  after: [WaitLROp]
  Partition 3:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(1, 1)]
        - USING  A: {1: 4}  B: {1: 6}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: {}  B: {}
        before: [none]  after: [WaitLROp, WaitLROp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(1, 1)]
        - USING  A: {1: 5}  B: {1: 7}
        before: [none]  after: [none]
"""

    assert actual == expected


def test_PGR2_64_64_1x1_emitted_modules_links():
    MT0 = MT1 = 64
    kernel = create_kernel(MT0, MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH,
                          VGPRTileReUseStrategy.ACROSS_SUBGROUP,
                          SubgroupOrdering.COLUMN_MAJOR)
    s = SubtileBasedScheduler(tiA, tiB, cfg)
    writer = create_writer_with_tiles(kernel, tiA, tiB)

    s.allocVgprTiles(writer)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s.printEmittedModules(writer, kernel, "MAINLOOP", s.mainloopSteps)
        actual = buf.getvalue()
    finally:
        s.deallocVgprTiles(writer)

    expected = """\
MAINLOOP EmittedModules:
  Partition 0:
    subIterK=0:
      id=0 mfma: core=4 insts before=[-] after=[-]
      id=1 lr: core=4 insts before=[-] after=[-]
      id=2 gr: core=4 insts before=[4] after=[-]
      id=3 wait_lr: core=1 insts before=[1] after=[-]
      id=4 sync: core=1 insts before=[3] after=[-]
    subIterK=1:
      id=0 mfma: core=4 insts before=[-] after=[-]
      id=1 lr: core=4 insts before=[5] after=[6]
      id=2 gr: core=4 insts before=[-] after=[7]
      id=3 wait_gr: core=1 insts before=[-] after=[-]
      id=4 sync: core=1 insts before=[3] after=[-]
      id=5 lr_inc: core=6 insts before=[4] after=[-]
      id=6 wait_lr: core=1 insts before=[1] after=[-]
      id=7 gr_inc: core=10 insts before=[2] after=[-]
"""
    assert expected in actual


def test_PGR2_256_256_1x1_extract_paths_from_before_deps():
    MT0 = MT1 = 256
    kernel = create_kernel(MT0, MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH,
                          VGPRTileReUseStrategy.ACROSS_SUBGROUP,
                          SubgroupOrdering.COLUMN_MAJOR)
    s = SubtileBasedScheduler(tiA, tiB, cfg)
    writer = create_writer_with_tiles(kernel, tiA, tiB)

    s.allocVgprTiles(writer)
    try:
        dtileInfo = writer.states.d.tileInfo
        pss = s.mainloopSteps[0]
        dus0 = pss.subIterKSteps[0]
        dus1 = pss.subIterKSteps[1]

        emitted0 = s._buildEmittedModules(writer, kernel, dus0.modules, dtileInfo)
        emitted1 = s._buildEmittedModules(writer, kernel, dus1.modules, dtileInfo)
    finally:
        s.deallocVgprTiles(writer)

    sig0 = [(em.moduleId, em.opType, len(em.core), em.before, em.after) for em in emitted0]
    sig1 = [(em.moduleId, em.opType, len(em.core), em.before, em.after) for em in emitted1]

    assert sig0 == [
        (0, "mfma", 64, None, []),
        (1, "lr", 16, None, []),
        (2, "gr", 16, 4, []),
        (3, "wait_lr", 1, 1, []),
        (4, "sync", 1, 3, []),
    ]
    assert sig1 == [
        (0, "mfma", 64, None, []),
        (1, "lr", 16, 5, [6]),
        (2, "gr", 16, None, [7]),
        (3, "wait_gr", 1, None, []),
        (4, "sync", 1, 3, []),
        (5, "lr_inc", 6, 4, []),
        (6, "wait_lr", 1, 1, []),
        (7, "gr_inc", 10, 2, []),
    ]

    mfmaIdx0, pathOrders0 = SubtileBasedScheduler._extractPathsFromBeforeDeps(emitted0)
    mfmaIdx1, pathOrders1 = SubtileBasedScheduler._extractPathsFromBeforeDeps(emitted1)

    assert mfmaIdx0 == 0
    assert pathOrders0 == [[1, 3, 4, 2]]
    assert mfmaIdx1 == 0
    assert pathOrders1 == [[3, 4, 5, 1, 6], [2, 7]]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp4", action="store_true", help="Enable FP4 path with MX scales")
    args = parser.parse_args()

    MT0=MT1=256
    kernel = create_kernel(MT0, MT1, fp4=args.fp4)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)

    scaleTiA = TileInfo('MXSA', kernel) if args.fp4 else None
    scaleTiB = TileInfo('MXSB', kernel) if args.fp4 else None

    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH,
                          VGPRTileReUseStrategy.ACROSS_SUBGROUP,
                          SubgroupOrdering.COLUMN_MAJOR)
    s = SubtileBasedScheduler(tiA, tiB, cfg,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

    assert len(s.preloopSteps)  > 0
    assert len(s.mainloopSteps) > 0
    assert len(s.ngllSteps)     > 0
    assert len(s.nllSteps)      > 0

    writer = create_writer_with_tiles(kernel, tiA, tiB,
                                      scaleTiA=scaleTiA, scaleTiB=scaleTiB)

    print("=== INITIAL ===")
    s.printSchedule(mode="initial")
    print("\n=== ANNOTATED ===")
    s.printSchedule(mode="annotated")

    s.allocVgprTiles(writer)
    s.printEmittedModules(writer, kernel, "MAINLOOP", s.mainloopSteps)
    s.deallocVgprTiles(writer)

    s.generateCode(writer, kernel)
    # kernel = create_kernel()
    # tiA = TileInfo('A', kernel)
    # tiB = TileInfo('B', kernel)
    # lsgA = tiA.localSubtileGrid[0]
    # lsgB = tiB.localSubtileGrid[0]

    # configs = [
    #     # (f"lsg {lsgA}x{lsgB}, group {lsgA}x{lsgB}, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
    #     #     SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),
    #     (f"lsg {lsgA}x{lsgB}, group {lsgA//2}x{lsgB//2}, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
    #         SchedulerConfig(lsgA//2, lsgB//2, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),
    # ]

    # for name, cfg in configs:
    #     print(f"=== {name} ===")
    #     s = SubtileBasedScheduler(tiA, tiB, cfg)
    #     s.printSchedule()
    #     writer = create_writer_with_tiles(kernel, tiA, tiB)
    #     # s.generateCode(writer, kernel)
    #     # print()
