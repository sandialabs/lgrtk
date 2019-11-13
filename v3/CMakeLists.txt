cmake_minimum_required(VERSION 3.10.0...${CMAKE_VERSION})

project(LGR VERSION 3.0.0 LANGUAGES CXX)

option(LGR_ENABLE_CUDA "Build GPU support" OFF)
option(LGR_ENABLE_OTM "Build support for meshless OTM method" OFF)

if (LGR_ENABLE_CUDA)
  enable_language(CUDA)
endif()

if (LGR_ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -Wall,-Wextra,-Werror,-Wno-noexcept-type --Werror cross-execution-space-call,deprecated-declarations --expt-extended-lambda")
else()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Werror -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-float-equal")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
  endif()
endif()

set(LGR_SOURCES
    lgr.cpp
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

if (LGR_ENABLE_OTM)
  set(LGR_SOURCES ${LGR_SOURCES} otm/otm.cpp)

endif()

option(LGR_ENABLE_EXODUS "Build the Exodus reader" OFF)

if (LGR_ENABLE_EXODUS)
  find_package(SEACASExodus REQUIRED)
  set(LGR_SOURCES ${LGR_SOURCES} lgr_exodus.cpp)
endif()

if (LGR_ENABLE_CUDA)
  set_source_files_properties(${LGR_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_executable(lgr ${LGR_SOURCES})

set_property(TARGET lgr PROPERTY CXX_STANDARD "14")
set_property(TARGET lgr PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET lgr PROPERTY CXX_EXTENSIONS OFF)
target_include_directories(lgr
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
if (LGR_ENABLE_EXODUS)
  target_compile_definitions(lgr PRIVATE -DLGR_ENABLE_EXODUS)
  target_link_libraries(lgr PRIVATE exodus)
  target_include_directories(lgr PRIVATE "${SEACASExodus_INCLUDE_DIRS}")
  target_include_directories(lgr PRIVATE "${SEACASExodus_TPL_INCLUDE_DIRS}")
# target_link_libraries(lgr PRIVATE "${SEACASExodus_TPL_LIBRARIES}")
endif()
