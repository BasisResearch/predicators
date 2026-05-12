cmake_minimum_required(VERSION 3.18)
project(pybullet LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Curated headless source list. Overrides Bullet's own CMakeLists and
# upstream pybullet's setup.py; both build OpenGL examples / the
# ExampleBrowser / Gwen GUI / networking that don't apply to Pyodide.
set(PYBULLET_SOURCES
    src/btLinearMathAll.cpp
    src/btBulletCollisionAll.cpp
    src/btBulletDynamicsAll.cpp
    src/Bullet3Common/b3AlignedAllocator.cpp
    src/Bullet3Common/b3Logging.cpp
    src/Bullet3Common/b3Vector3.cpp

    src/BulletInverseDynamics/IDMath.cpp
    src/BulletInverseDynamics/MultiBodyTree.cpp
    src/BulletInverseDynamics/details/MultiBodyTreeImpl.cpp
    src/BulletInverseDynamics/details/MultiBodyTreeInitCache.cpp

    # PhysicsServerCommandProcessor links the rigid+softbody collision
    # config unconditionally — soft body is required even rigid-only.
    src/BulletSoftBody/btDefaultSoftBodySolver.cpp
    src/BulletSoftBody/btDeformableBackwardEulerObjective.cpp
    src/BulletSoftBody/btDeformableBodySolver.cpp
    src/BulletSoftBody/btDeformableContactConstraint.cpp
    src/BulletSoftBody/btDeformableContactProjection.cpp
    src/BulletSoftBody/btDeformableMultiBodyConstraintSolver.cpp
    src/BulletSoftBody/btDeformableMultiBodyDynamicsWorld.cpp
    src/BulletSoftBody/btSoftBody.cpp
    src/BulletSoftBody/btSoftBodyConcaveCollisionAlgorithm.cpp
    src/BulletSoftBody/btSoftBodyHelpers.cpp
    src/BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.cpp
    src/BulletSoftBody/btSoftMultiBodyDynamicsWorld.cpp
    src/BulletSoftBody/btSoftRigidCollisionAlgorithm.cpp
    src/BulletSoftBody/btSoftRigidDynamicsWorld.cpp
    src/BulletSoftBody/btSoftSoftCollisionAlgorithm.cpp
    src/BulletSoftBody/poly34.cpp
    src/BulletSoftBody/BulletReducedDeformableBody/btReducedDeformableBody.cpp
    src/BulletSoftBody/BulletReducedDeformableBody/btReducedDeformableBodyHelpers.cpp
    src/BulletSoftBody/BulletReducedDeformableBody/btReducedDeformableBodySolver.cpp
    src/BulletSoftBody/BulletReducedDeformableBody/btReducedDeformableContactConstraint.cpp

    examples/pybullet/pybullet.c

    # SharedMemory in DIRECT mode (no UDP/TCP/Win32/Posix shm).
    examples/SharedMemory/PhysicsClient.cpp
    examples/SharedMemory/PhysicsClientC_API.cpp
    examples/SharedMemory/PhysicsDirect.cpp
    examples/SharedMemory/PhysicsDirectC_API.cpp
    examples/SharedMemory/PhysicsServer.cpp
    examples/SharedMemory/PhysicsServerCommandProcessor.cpp
    examples/SharedMemory/PhysicsServerSharedMemory.cpp
    examples/SharedMemory/PhysicsClientSharedMemory.cpp
    examples/SharedMemory/PhysicsClientSharedMemory_C_API.cpp
    examples/SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp
    examples/SharedMemory/InProcessMemory.cpp
    examples/SharedMemory/IKTrajectoryHelper.cpp
    examples/SharedMemory/b3PluginManager.cpp
    examples/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp
    examples/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp
    examples/SharedMemory/plugins/fileIOPlugin/fileIOPlugin.cpp
    examples/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp
    examples/SharedMemory/plugins/stablePDPlugin/BulletConversion.cpp
    examples/SharedMemory/plugins/stablePDPlugin/KinTree.cpp
    examples/SharedMemory/plugins/stablePDPlugin/MathUtil.cpp
    examples/SharedMemory/plugins/stablePDPlugin/RBDModel.cpp
    examples/SharedMemory/plugins/stablePDPlugin/RBDUtil.cpp
    examples/SharedMemory/plugins/stablePDPlugin/Shape.cpp
    examples/SharedMemory/plugins/stablePDPlugin/SpAlg.cpp
    examples/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp
    examples/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp

    # TinyRenderer (CPU getCameraImage). SimpleCamera lives under
    # OpenGLWindow/ but is pure math; needed for the linker.
    examples/TinyRenderer/geometry.cpp
    examples/TinyRenderer/model.cpp
    examples/TinyRenderer/our_gl.cpp
    examples/TinyRenderer/tgaimage.cpp
    examples/TinyRenderer/TinyRenderer.cpp
    examples/OpenGLWindow/SimpleCamera.cpp

    # Wavefront2GLInstanceGraphicsShape's SimpleOpenGL3App include is
    # removed by patches/0001-wavefront-drop-opengl-include.patch.
    examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp
    examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp
    examples/Importers/ImportURDFDemo/URDF2Bullet.cpp
    examples/Importers/ImportURDFDemo/UrdfParser.cpp
    examples/Importers/ImportURDFDemo/urdfStringSplit.cpp
    examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp
    examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp
    examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp
    examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp
    examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp

    examples/Utils/b3Clock.cpp
    examples/Utils/b3ResourcePath.cpp
    examples/Utils/b3Quickprof.cpp
    examples/Utils/ChromeTraceUtil.cpp
    examples/Utils/RobotLoggingUtil.cpp

    examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp
    examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp
    examples/ThirdPartyLibs/stb_image/stb_image.cpp
    examples/ThirdPartyLibs/stb_image/stb_image_write.cpp
    examples/ThirdPartyLibs/minizip/ioapi.c
    examples/ThirdPartyLibs/minizip/unzip.c
    examples/ThirdPartyLibs/minizip/zip.c
    examples/ThirdPartyLibs/zlib/adler32.c
    examples/ThirdPartyLibs/zlib/compress.c
    examples/ThirdPartyLibs/zlib/crc32.c
    examples/ThirdPartyLibs/zlib/deflate.c
    examples/ThirdPartyLibs/zlib/gzclose.c
    examples/ThirdPartyLibs/zlib/gzlib.c
    examples/ThirdPartyLibs/zlib/gzread.c
    examples/ThirdPartyLibs/zlib/gzwrite.c
    examples/ThirdPartyLibs/zlib/infback.c
    examples/ThirdPartyLibs/zlib/inffast.c
    examples/ThirdPartyLibs/zlib/inflate.c
    examples/ThirdPartyLibs/zlib/inftrees.c
    examples/ThirdPartyLibs/zlib/trees.c
    examples/ThirdPartyLibs/zlib/uncompr.c
    examples/ThirdPartyLibs/zlib/zutil.c

    # BussIK is the analytical IK solver used by IKTrajectoryHelper.
    examples/ThirdPartyLibs/BussIK/Jacobian.cpp
    examples/ThirdPartyLibs/BussIK/LinearR2.cpp
    examples/ThirdPartyLibs/BussIK/LinearR3.cpp
    examples/ThirdPartyLibs/BussIK/LinearR4.cpp
    examples/ThirdPartyLibs/BussIK/MatrixRmn.cpp
    examples/ThirdPartyLibs/BussIK/Misc.cpp
    examples/ThirdPartyLibs/BussIK/Node.cpp
    examples/ThirdPartyLibs/BussIK/Tree.cpp
    examples/ThirdPartyLibs/BussIK/VectorRn.cpp

    Extras/Serialize/BulletFileLoader/bChunk.cpp
    Extras/Serialize/BulletFileLoader/bDNA.cpp
    Extras/Serialize/BulletFileLoader/bFile.cpp
    Extras/Serialize/BulletFileLoader/btBulletFile.cpp
    Extras/Serialize/BulletWorldImporter/btMultiBodyWorldImporter.cpp
    Extras/Serialize/BulletWorldImporter/btBulletWorldImporter.cpp
    Extras/Serialize/BulletWorldImporter/btWorldImporter.cpp

    Extras/InverseDynamics/CloneTreeCreator.cpp
    Extras/InverseDynamics/IDRandomUtil.cpp
    Extras/InverseDynamics/MultiBodyTreeDebugGraph.cpp
    Extras/InverseDynamics/User2InternalIndex.cpp
    Extras/InverseDynamics/CoilCreator.cpp
    Extras/InverseDynamics/MultiBodyNameMap.cpp
    Extras/InverseDynamics/RandomTreeCreator.cpp
    Extras/InverseDynamics/btMultiBodyTreeCreator.cpp
    Extras/InverseDynamics/DillCreator.cpp
    Extras/InverseDynamics/MultiBodyTreeCreator.cpp
    Extras/InverseDynamics/SimpleTreeCreator.cpp
    Extras/InverseDynamics/invdyn_bullet_comparison.cpp
)

Python3_add_library(pybullet MODULE WITH_SOABI ${PYBULLET_SOURCES})

target_include_directories(pybullet PRIVATE
    src
    examples
    examples/SharedMemory
    examples/CommonInterfaces
    examples/ThirdPartyLibs
    examples/ThirdPartyLibs/glad
    examples/ThirdPartyLibs/Wavefront
    Extras/Serialize/BulletFileLoader
    Extras/Serialize/BulletWorldImporter
    Extras/InverseDynamics
)

target_link_libraries(pybullet PRIVATE Python3::Module Python3::NumPy)

target_compile_definitions(pybullet PRIVATE
    BT_USE_DOUBLE_PRECISION
    B3_ENABLE_FILEIO_PLUGIN
    B3_USE_ZIPFILE_FILEIO
    PYBULLET_USE_NUMPY
    BT_THREADSAFE=0
)

# Bullet uses neither RTTI nor exceptions in any source we compile
# (verified across the curated set). Dropping both shrinks the binary.
# `-ffunction-sections` + `-fdata-sections` plus `--gc-sections` at link
# time mirror MSVC's `/Gy` + `/OPT:REF` defaults (which upstream's
# CMakeLists turns on by default for MSVC but never enables for
# gcc/clang). Reachability roots are determined by PyInit_pybullet.
target_compile_options(pybullet PRIVATE
    -fno-rtti
    -fno-exceptions
    -ffunction-sections
    -fdata-sections
)
target_link_options(pybullet PRIVATE -Wl,--gc-sections)

# Hide all non-exported symbols. Pyodide's recipe-level `exports: pyinit`
# already controls which symbols escape the wheel, but hiding at compile
# time also shrinks the dynamic relocation table and helps LTO inline
# more aggressively.
set_target_properties(pybullet PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

install(TARGETS pybullet LIBRARY DESTINATION ".")
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/examples/pybullet/gym/pybullet_data
    DESTINATION "."
)
