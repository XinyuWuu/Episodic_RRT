cmake_minimum_required ( VERSION 3.16 )
project ( CppEnv )
set ( CMAKE_CXX_STANDARD 20 )
set ( CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE )

set ( ExternalProjectDir "${CMAKE_CURRENT_BINARY_DIR}" )
include_directories ( BEFORE "${ExternalProjectDir}/include" )
link_directories ( BEFORE "${ExternalProjectDir}/lib" )
link_directories ( BEFORE "${ExternalProjectDir}/lib64" )
list ( INSERT CMAKE_PREFIX_PATH 0 "${ExternalProjectDir}" )
list ( APPEND CMAKE_INSTALL_RPATH "${ExternalProjectDir}/lib" )
list ( APPEND CMAKE_INSTALL_RPATH "${ExternalProjectDir}/lib64" )
list ( APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" )
set ( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

include ( FetchContent )
include ( ExternalProject )

ExternalProject_Add (
    openblaslib
    URL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.28/OpenBLAS-0.3.28.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Add (
    lapacklib
    URL https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Add (
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DEIGEN_USE_BLAS:STRING=ON
    -DEIGEN_USE_LAPACK:STRING=ON
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    DEPENDS
    openblaslib
    lapacklib
)
include_directories ( BEFORE "${ExternalProjectDir}/include/eigen3" )

ExternalProject_Add (
    armadillolib
    URL https://sourceforge.net/projects/arma/files/armadillo-14.2.0.tar.xz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DOPENBLAS_PROVIDES_LAPACK=ON
    DEPENDS
    eigen
    openblaslib
)

ExternalProject_Add (
    tinysplinelib
    URL https://github.com/msteinbeck/tinyspline/archive/refs/tags/v0.6.0.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Add (
    libccdlib
    GIT_REPOSITORY https://github.com/danfis/libccd.git
    GIT_TAG 7931e764a19ef6b21b443376c699bbc9c6d4fba8
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Add (
    fcllib
    URL https://github.com/flexible-collision-library/fcl/archive/refs/tags/0.7.0.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    DEPENDS
    eigen
    libccdlib
)

ExternalProject_Add (
    rerunlib
    URL https://github.com/rerun-io/rerun/releases/download/0.23.3/rerun_cpp_sdk.zip
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Add (
    ompllib
    URL https://github.com/ompl/omplapp/releases/download/1.6.0/omplapp-1.6.0-Source.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DOMPL_VERSIONED_INSTALL=OFF
    DEPENDS
    eigen
)

ExternalProject_Add (
    assimplib
    URL https://github.com/assimp/assimp/archive/refs/tags/v5.4.3.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Add (
    robinmaplib
    URL https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Add (
    yamllib
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
    PREFIX ${ExternalProjectDir}
    INSTALL_DIR ${ExternalProjectDir}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DYAML_BUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

add_library ( spline SHARED "spline.cpp" )
add_dependencies ( spline armadillolib tinysplinelib )
target_link_libraries ( spline PRIVATE "armadillo" )
target_link_libraries ( spline PRIVATE "tinysplinecxx" )

add_library ( ikfastur5e SHARED "ikfast/ur5e_ikfast61.cpp" )
add_dependencies ( ikfastur5e openblaslib lapacklib armadillolib )
target_compile_definitions ( ikfastur5e PRIVATE IKFAST_NO_MAIN )
target_compile_definitions ( ikfastur5e PRIVATE IKFAST_CLIBRARY )
target_link_libraries ( ikfastur5e PRIVATE openblas )
target_link_libraries ( ikfastur5e PRIVATE lapack )

add_library ( ur5eikfast SHARED "ikfast/ur5e.cpp" )
add_dependencies ( ur5eikfast ikfastur5e eigen armadillolib )
target_link_libraries ( ur5eikfast PRIVATE "ikfastur5e" )

add_library ( ur5e SHARED "ur5e.cpp" )
add_dependencies ( ur5e eigen assimplib fcllib )
target_link_libraries ( ur5e PRIVATE "assimp" )
target_link_libraries ( ur5e PRIVATE "fcl" )
target_link_libraries ( ur5e PRIVATE "ur5eikfast" )

add_library ( scene SHARED "scene.cpp" )
add_dependencies ( scene fcllib armadillolib )
target_link_libraries ( scene PRIVATE "armadillo" )
target_link_libraries ( scene PRIVATE "fcl" )
target_link_libraries ( scene PRIVATE "ccd" )
target_link_libraries ( scene PRIVATE "ur5e" )
target_link_libraries ( scene PRIVATE "ur5eikfast" )

add_library ( logger SHARED "logger.cpp" )
add_dependencies ( logger armadillolib rerunlib )
target_link_libraries ( logger PRIVATE "armadillo" )
target_link_libraries ( logger PRIVATE "rerun_sdk" )

add_library ( simulation SHARED "simulation.cpp" )
add_dependencies ( simulation fcllib armadillolib rerunlib robinmaplib )
target_link_libraries ( simulation PRIVATE "armadillo" )
target_link_libraries ( simulation PRIVATE "rerun_sdk" )
target_link_libraries ( simulation PRIVATE "logger" )
target_link_libraries ( simulation PRIVATE "spline" )
target_link_libraries ( simulation PRIVATE "ur5e" )
target_link_libraries ( simulation PRIVATE "ur5eikfast" )
target_link_libraries ( simulation PRIVATE "scene" )

install ( TARGETS logger spline scene ikfastur5e ur5eikfast ur5e simulation
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" )

add_executable ( test "test.cpp" )
add_dependencies ( test yamllib )
target_link_libraries ( test PRIVATE "openblas" )
target_link_libraries ( test PRIVATE "fcl" )
target_link_libraries ( test PRIVATE "ccd" )
target_link_libraries ( test PRIVATE "spline" )
target_link_libraries ( test PRIVATE "scene" )
target_link_libraries ( test PRIVATE "simulation" )
target_link_libraries ( test PRIVATE "ur5e" )
target_link_libraries ( test PRIVATE "ur5eikfast" )
target_link_libraries ( test PRIVATE "logger" )
target_link_libraries ( test PRIVATE "tinysplinecxx" )
target_link_libraries ( test PRIVATE "armadillo" )
target_link_libraries ( test PRIVATE "rerun_sdk" )
target_link_libraries ( test PRIVATE "ompl" )
target_link_libraries ( test PRIVATE "openvino" )
target_link_libraries ( test PRIVATE "yaml-cpp" )

install ( TARGETS test DESTINATION "${CMAKE_INSTALL_PREFIX}" )
