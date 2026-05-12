"""Pyodide/Emscripten-targeted setup.py for pybullet.

Replaces upstream setup.py when cross-compiling for WASM. Strips OpenGL,
X11, Cocoa, Win32 windows; networking (enet, clsocket); GUI (Gwen);
EGL plugin; VHACD; soft bodies. Keeps the core physics + URDF parser +
TinyRenderer (CPU rasterizer for getCameraImage).

If a source compiles but pulls in a missing symbol at link time, add it
back here. If it fails to compile, either fix the upstream include or
drop it from the list and accept reduced functionality.
"""
from __future__ import annotations

import os
import sys
import platform as _stdlib_platform

from setuptools import Extension, find_packages, setup

# Note: pyodide-build invokes this setup.py twice — once with the host
# Python (to query build requirements / metadata) and once under the
# Emscripten cross-compile env (to actually compile). We don't gate on
# sys.platform here because the metadata phase needs to succeed.

# ---------------------------------------------------------------------------
# Source list — curated for headless physics. Order matters for some
# compilers but not for the wasm-ld driver. Grouped by subdirectory for
# readability.

SRC = []

# --- Bullet core (these aggregate the LinearMath/Collision/Dynamics dirs)
SRC += [
    "src/btLinearMathAll.cpp",
    "src/btBulletCollisionAll.cpp",
    "src/btBulletDynamicsAll.cpp",
]

# --- Bullet3 common helpers
SRC += [
    "src/Bullet3Common/b3AlignedAllocator.cpp",
    "src/Bullet3Common/b3Logging.cpp",
    "src/Bullet3Common/b3Vector3.cpp",
]

# --- Bullet inverse dynamics (used by IK)
SRC += [
    "src/BulletInverseDynamics/IDMath.cpp",
    "src/BulletInverseDynamics/MultiBodyTree.cpp",
    "src/BulletInverseDynamics/details/MultiBodyTreeImpl.cpp",
    "src/BulletInverseDynamics/details/MultiBodyTreeInitCache.cpp",
]

# --- Bullet soft body (PhysicsServerCommandProcessor links the
# rigid+softbody collision config unconditionally, so we need it even
# for "rigid-only" use cases).
SRC += [
    "src/BulletSoftBody/btDefaultSoftBodySolver.cpp",
    "src/BulletSoftBody/btDeformableBackwardEulerObjective.cpp",
    "src/BulletSoftBody/btDeformableBodySolver.cpp",
    "src/BulletSoftBody/btDeformableContactConstraint.cpp",
    "src/BulletSoftBody/btDeformableContactProjection.cpp",
    "src/BulletSoftBody/btDeformableMultiBodyConstraintSolver.cpp",
    "src/BulletSoftBody/btDeformableMultiBodyDynamicsWorld.cpp",
    "src/BulletSoftBody/btSoftBody.cpp",
    "src/BulletSoftBody/btSoftBodyConcaveCollisionAlgorithm.cpp",
    "src/BulletSoftBody/btSoftBodyHelpers.cpp",
    "src/BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.cpp",
    "src/BulletSoftBody/btSoftMultiBodyDynamicsWorld.cpp",
    "src/BulletSoftBody/btSoftRigidCollisionAlgorithm.cpp",
    "src/BulletSoftBody/btSoftRigidDynamicsWorld.cpp",
    "src/BulletSoftBody/btSoftSoftCollisionAlgorithm.cpp",
    "src/BulletSoftBody/poly34.cpp",
    "src/BulletSoftBody/BulletReducedDeformableBody/btReducedDeformableBody.cpp",
    "src/BulletSoftBody/BulletReducedDeformableBody/btReducedDeformableBodyHelpers.cpp",
    "src/BulletSoftBody/BulletReducedDeformableBody/btReducedDeformableBodySolver.cpp",
    "src/BulletSoftBody/BulletReducedDeformableBody/btReducedDeformableContactConstraint.cpp",
]

# --- pybullet's Python binding
SRC += ["examples/pybullet/pybullet.c"]

# --- SharedMemory subsystem (DIRECT mode, no UDP/TCP/Win32/Posix shm)
SRC += [
    "examples/SharedMemory/PhysicsClient.cpp",
    "examples/SharedMemory/PhysicsClientC_API.cpp",
    "examples/SharedMemory/PhysicsDirect.cpp",
    "examples/SharedMemory/PhysicsDirectC_API.cpp",
    "examples/SharedMemory/PhysicsServer.cpp",
    "examples/SharedMemory/PhysicsServerCommandProcessor.cpp",
    "examples/SharedMemory/PhysicsServerSharedMemory.cpp",
    "examples/SharedMemory/PhysicsClientSharedMemory.cpp",
    "examples/SharedMemory/PhysicsClientSharedMemory_C_API.cpp",
    "examples/SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp",
    "examples/SharedMemory/InProcessMemory.cpp",
    "examples/SharedMemory/IKTrajectoryHelper.cpp",
    "examples/SharedMemory/b3PluginManager.cpp",
    "examples/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp",
]

# --- SharedMemory plugins (use in-process only)
SRC += [
    "examples/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp",
    "examples/SharedMemory/plugins/fileIOPlugin/fileIOPlugin.cpp",
    "examples/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp",
    "examples/SharedMemory/plugins/stablePDPlugin/BulletConversion.cpp",
    "examples/SharedMemory/plugins/stablePDPlugin/KinTree.cpp",
    "examples/SharedMemory/plugins/stablePDPlugin/MathUtil.cpp",
    "examples/SharedMemory/plugins/stablePDPlugin/RBDModel.cpp",
    "examples/SharedMemory/plugins/stablePDPlugin/RBDUtil.cpp",
    "examples/SharedMemory/plugins/stablePDPlugin/Shape.cpp",
    "examples/SharedMemory/plugins/stablePDPlugin/SpAlg.cpp",
    "examples/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp",
    "examples/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp",
]

# --- TinyRenderer (CPU-only rasterizer for getCameraImage in DIRECT mode)
SRC += [
    "examples/TinyRenderer/geometry.cpp",
    "examples/TinyRenderer/model.cpp",
    "examples/TinyRenderer/our_gl.cpp",
    "examples/TinyRenderer/tgaimage.cpp",
    "examples/TinyRenderer/TinyRenderer.cpp",
    # SimpleCamera is pure math (no OpenGL) and is pulled in by
    # TinyRendererVisualShapeConverter; including it keeps the linker happy.
    "examples/OpenGLWindow/SimpleCamera.cpp",
]

# --- URDF / MJCF / Collada importers (file-based, no GL deps)
SRC += [
    "examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp",
    "examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp",
    "examples/Importers/ImportURDFDemo/URDF2Bullet.cpp",
    "examples/Importers/ImportURDFDemo/UrdfParser.cpp",
    "examples/Importers/ImportURDFDemo/urdfStringSplit.cpp",
    "examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp",
    "examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp",
    # Wavefront2GLInstanceGraphicsShape is pure data conversion; its
    # SimpleOpenGL3App include is patched out by source_patches.py.
    "examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp",
    "examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp",
    "examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp",
]

# --- Utils
SRC += [
    "examples/Utils/b3Clock.cpp",
    "examples/Utils/b3ResourcePath.cpp",
    "examples/Utils/b3Quickprof.cpp",
    "examples/Utils/ChromeTraceUtil.cpp",
    "examples/Utils/RobotLoggingUtil.cpp",
]

# --- ThirdPartyLibs we actually need
SRC += [
    "examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
    "examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",
    "examples/ThirdPartyLibs/stb_image/stb_image.cpp",
    "examples/ThirdPartyLibs/stb_image/stb_image_write.cpp",
    "examples/ThirdPartyLibs/minizip/ioapi.c",
    "examples/ThirdPartyLibs/minizip/unzip.c",
    "examples/ThirdPartyLibs/minizip/zip.c",
    "examples/ThirdPartyLibs/zlib/adler32.c",
    "examples/ThirdPartyLibs/zlib/compress.c",
    "examples/ThirdPartyLibs/zlib/crc32.c",
    "examples/ThirdPartyLibs/zlib/deflate.c",
    "examples/ThirdPartyLibs/zlib/gzclose.c",
    "examples/ThirdPartyLibs/zlib/gzlib.c",
    "examples/ThirdPartyLibs/zlib/gzread.c",
    "examples/ThirdPartyLibs/zlib/gzwrite.c",
    "examples/ThirdPartyLibs/zlib/infback.c",
    "examples/ThirdPartyLibs/zlib/inffast.c",
    "examples/ThirdPartyLibs/zlib/inflate.c",
    "examples/ThirdPartyLibs/zlib/inftrees.c",
    "examples/ThirdPartyLibs/zlib/trees.c",
    "examples/ThirdPartyLibs/zlib/uncompr.c",
    "examples/ThirdPartyLibs/zlib/zutil.c",
    # BussIK — analytical IK solver used by IKTrajectoryHelper.
    "examples/ThirdPartyLibs/BussIK/Jacobian.cpp",
    "examples/ThirdPartyLibs/BussIK/LinearR2.cpp",
    "examples/ThirdPartyLibs/BussIK/LinearR3.cpp",
    "examples/ThirdPartyLibs/BussIK/LinearR4.cpp",
    "examples/ThirdPartyLibs/BussIK/MatrixRmn.cpp",
    "examples/ThirdPartyLibs/BussIK/Misc.cpp",
    "examples/ThirdPartyLibs/BussIK/Node.cpp",
    "examples/ThirdPartyLibs/BussIK/Tree.cpp",
    "examples/ThirdPartyLibs/BussIK/VectorRn.cpp",
]

# --- Bullet binary-format serialization (used by save/load)
SRC += [
    "Extras/Serialize/BulletFileLoader/bChunk.cpp",
    "Extras/Serialize/BulletFileLoader/bDNA.cpp",
    "Extras/Serialize/BulletFileLoader/bFile.cpp",
    "Extras/Serialize/BulletFileLoader/btBulletFile.cpp",
    "Extras/Serialize/BulletWorldImporter/btMultiBodyWorldImporter.cpp",
    "Extras/Serialize/BulletWorldImporter/btBulletWorldImporter.cpp",
    "Extras/Serialize/BulletWorldImporter/btWorldImporter.cpp",
]

# --- Extras inverse-dynamics extensions
SRC += [
    "Extras/InverseDynamics/CloneTreeCreator.cpp",
    "Extras/InverseDynamics/IDRandomUtil.cpp",
    "Extras/InverseDynamics/MultiBodyTreeDebugGraph.cpp",
    "Extras/InverseDynamics/User2InternalIndex.cpp",
    "Extras/InverseDynamics/CoilCreator.cpp",
    "Extras/InverseDynamics/MultiBodyNameMap.cpp",
    "Extras/InverseDynamics/RandomTreeCreator.cpp",
    "Extras/InverseDynamics/btMultiBodyTreeCreator.cpp",
    "Extras/InverseDynamics/DillCreator.cpp",
    "Extras/InverseDynamics/MultiBodyTreeCreator.cpp",
    "Extras/InverseDynamics/SimpleTreeCreator.cpp",
    "Extras/InverseDynamics/invdyn_bullet_comparison.cpp",
]

# ---------------------------------------------------------------------------
# Compile flags. Drop networking/EGL/threading. Keep precision and FileIO.
#
# Size optimizations (verified no RTTI / dynamic_cast / typeid / try-catch /
# throw in any Bullet or pybullet source we compile):
#   -fno-rtti       — Bullet's own CMake disables RTTI on MSVC by default
#                     (USE_MSVC_DISABLE_RTTI=ON). C++ files only; C is untouched.
#   -fno-exceptions — no try/catch in any compiled source.
#   -flto           — link-time optimization; lets the linker drop unreachable
#                     code across translation units. Emscripten supports it.
#
# Keep `-O3` (Pyodide's default); Bullet is compute-heavy and `-Oz` would
# slow it down without huge size wins.
CXX_FLAGS = (
    "-DGWEN_COMPILE_STATIC "
    "-DBT_USE_DOUBLE_PRECISION "
    "-DB3_DUMP_PYTHON_VERSION "
    "-DB3_ENABLE_FILEIO_PLUGIN "
    "-DB3_USE_ZIPFILE_FILEIO "
    "-DPYBULLET_USE_NUMPY "
    # Emscripten lacks pthread by default; disable Bullet's thread layer.
    "-DBT_THREADSAFE=0 "
    # Size optimizations — see comment block above for justification.
    "-fno-rtti "
    "-fno-exceptions "
    "-flto "
    # Suppress warnings that the noisy Bullet codebase emits.
    "-Wno-deprecated-declarations "
    "-Wno-deprecated-register "
    "-Wno-unused-but-set-variable "
    "-Wno-unused-variable "
    "-Wno-unused-function "
    "-Wno-tautological-compare "
    "-Wno-unused-command-line-argument "
)

INCLUDE_DIRS = [
    "src",
    "examples",
    "examples/SharedMemory",
    "examples/CommonInterfaces",
    "examples/ThirdPartyLibs",
    "examples/ThirdPartyLibs/glad",  # for occasional includes even if we skip glad sources
    "examples/ThirdPartyLibs/Wavefront",
    "Extras/Serialize/BulletFileLoader",
    "Extras/Serialize/BulletWorldImporter",
    "Extras/InverseDynamics",
]

try:
    import numpy as _np
    INCLUDE_DIRS.append(_np.get_include())
except ImportError:  # pragma: no cover — Pyodide should provide numpy
    pass

# ---------------------------------------------------------------------------
# Resource files for pybullet_data. Upstream ships ~150 MB of assets; we
# strip the ones our use case will never touch (RL policy checkpoints,
# benchmark meshes, random-URDF dumps). Predicators ships its own URDFs
# under predicators/envs/assets/urdf/, so we only need the basics like
# plane.urdf and cube.urdf that pybullet itself references in defaults.
need_files = []
datadir = "examples/pybullet/gym/pybullet_data"
setup_py_dir = os.path.dirname(os.path.realpath(__file__))
hh = setup_py_dir + "/" + datadir
allowed_exts = set(
    "yaml index meta data-00000-of-00001 png gif jpg urdf sdf obj txt mtl dae off stl STL xml gin npy".split())

# Path prefixes (relative to pybullet_data/) we won't ship.
SKIP_PREFIXES = (
    "data/policies/",          # humanoid3d RL checkpoints, ~30 MB
    "policies/",               # minitaur PPO checkpoints, ~15 MB
    "data/motions/",           # mocap reference motions for RL
    "MPL/",                    # massive sample data
    # Specific oversized single files:
)
SKIP_NAMES = {
    "samurai_monastry.obj",    # 10 MB
    "random_urdfs.zip",        # 5 MB
}
# Mesh subdirs that ship multi-megabyte STLs we don't need for basic envs.
SKIP_SUBSTRINGS = (
    "/laikago/",
    "/atlas/",
    "/quadruped/",
    "/humanoid/",
    "/biped/",
    "/r2d2_concave",
    "/dinnerware/",
    "/duck",                   # 235 KB but unused
)

for root, _dirs, files in os.walk(hh):
    for fn in files:
        ext = os.path.splitext(fn)[1][1:]
        if not ext or ext not in allowed_exts:
            continue
        full = os.path.join(root, fn)
        rel = full[1 + len(hh):]
        if any(rel.startswith(p) for p in SKIP_PREFIXES):
            continue
        if fn in SKIP_NAMES:
            continue
        if any(s in rel for s in SKIP_SUBSTRINGS):
            continue
        need_files.append(rel)
print(f"pybullet_data: shipping {len(need_files)} files (filtered)")

pybullet_ext = Extension(
    "pybullet",
    sources=SRC,
    extra_compile_args=CXX_FLAGS.split(),
    include_dirs=INCLUDE_DIRS,
)

setup(
    name="pybullet",
    version="3.2.7",
    description="Pyodide-targeted build of pybullet (headless physics only).",
    url="https://github.com/bulletphysics/bullet3",
    author="Erwin Coumans, Yunfei Bai, Jasmine Hsu",
    license="zlib",
    ext_modules=[pybullet_ext],
    package_dir={"": "examples/pybullet/gym"},
    packages=list(find_packages("examples/pybullet/gym")),
    package_data={"pybullet_data": need_files},
)
