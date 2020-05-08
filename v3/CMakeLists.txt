cmake_minimum_required(VERSION 3.10.0...${CMAKE_VERSION})

project(LGR VERSION 3.0.0 LANGUAGES CXX)

option(LGR_ENABLE_CUDA "Build GPU support" OFF)
option(LGR_ENABLE_UNIT_TESTS "Enable unit tests" ON)
option(LGR_ENABLE_EFENCE "Build with ElectricFence support" OFF)

if (LGR_ENABLE_CUDA)
  enable_language(CUDA)
endif()

if (LGR_ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -Wall,-Wextra,-Werror,-Wno-noexcept-type --Werror cross-execution-space-call,deprecated-declarations --expt-extended-lambda")
else()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-float-equal")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
  endif()
endif()

set(LGR_SOURCES
    lgr_state.cpp
    lgr_vtk.cpp
    lgr_domain.cpp
    lgr_physics.cpp
    lgr_bar.cpp
    lgr_triangle.cpp
    lgr_tetrahedron.cpp
    lgr_element_specific.cpp
    lgr_composite_nodal_mass.cpp
    lgr_composite_h_min.cpp
    lgr_composite_gradient.cpp
    lgr_composite_tetrahedron.cpp
    lgr_meshing.cpp
    lgr_stabilized.cpp
    lgr_adapt.cpp
    )

set(OTM_SOURCES
    otm_apps.cpp
    otm_meshless.cpp
    otm_distance.cpp
    otm_tet2meshless.cpp
    otm_meshing.cpp
    otm_vtk.cpp
    )

option(LGR_ENABLE_EXODUS "Build the Exodus reader" OFF)

if (LGR_ENABLE_EXODUS)
  option(EXODUS_HAS_MPI "Whether Exodus was built with MPI support" OFF)
  find_package(SEACASExodus REQUIRED)
  if (EXODUS_HAS_MPI)
    find_package(MPI REQUIRED)
  endif()
  set(LGR_SOURCES ${LGR_SOURCES} lgr_exodus.cpp)
endif()

option(LGR_ENABLE_SEARCH "Build support for meshfree search via ArborX" OFF)

if (LGR_ENABLE_SEARCH)
  find_package(ArborX REQUIRED)
  set(OTM_SOURCES ${OTM_SOURCES} otm_search.cpp otm_arborx_search_impl.cpp)
endif()

if (LGR_ENABLE_CUDA)
  set_source_files_properties(${LGR_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(${OTM_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(lgrlib ${LGR_SOURCES} ${OTM_SOURCES})
set_property(TARGET lgrlib PROPERTY CXX_STANDARD "14")
set_property(TARGET lgrlib PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET lgrlib PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET lgrlib PROPERTY OUTPUT_NAME lgr)
target_include_directories(lgrlib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

if (LGR_ENABLE_EXODUS)
  target_compile_definitions(lgrlib PUBLIC -DLGR_ENABLE_EXODUS)
  target_link_libraries(lgrlib PUBLIC exodus)
  target_include_directories(lgrlib PUBLIC "${SEACASExodus_INCLUDE_DIRS}")
  target_include_directories(lgrlib PUBLIC "${SEACASExodus_TPL_INCLUDE_DIRS}")
  target_link_libraries(lgrlib PUBLIC "${SEACASExodus_TPL_LIBRARIES}")
  if (EXODUS_HAS_MPI)
    target_link_libraries(lgrlib PRIVATE MPI::MPI_CXX)
  endif()
endif()

if (LGR_ENABLE_SEARCH)
  # target_include_directories(lgrlib PUBLIC "${Kokkos_INCLUDE_DIRS}")
  # target_include_directories(lgrlib PUBLIC "${Kokkos_TPL_INCLUDE_DIRS}")
  target_link_libraries(lgrlib PUBLIC ArborX::ArborX)
endif()

if (LGR_ENABLE_CUDA)
  set_source_files_properties(lgr.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(otm.cpp PROPERTIES LANGUAGE CUDA)
endif()

set(LGR_LIBRARIES lgrlib)
if (LGR_ENABLE_EFENCE)
  set(LGR_LIBRARIES ${LGR_LIBRARIES}";efence")
endif()


add_executable(lgr lgr.cpp)
set_property(TARGET lgr PROPERTY CXX_STANDARD "14")
set_property(TARGET lgr PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET lgr PROPERTY CXX_EXTENSIONS OFF)
target_link_libraries(lgr ${LGR_LIBRARIES})

if (LGR_ENABLE_SEARCH)
  add_executable(otm otm.cpp)
  set_property(TARGET otm PROPERTY CXX_STANDARD "14")
  set_property(TARGET otm PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET otm PROPERTY CXX_EXTENSIONS OFF)
  target_link_libraries(otm ${LGR_LIBRARIES})
endif()

if (LGR_ENABLE_UNIT_TESTS)
  add_subdirectory(unit_tests)
endif()
