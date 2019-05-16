cmake_minimum_required(VERSION 3.10.0...${CMAKE_VERSION})

project(LGR VERSION 3.0.0 LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Werror -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-float-equal")
endif()

set(LGR_SOURCES
    lgr.cpp
    lgr_state.cpp
    lgr_print.cpp
    lgr_vtk.cpp
    lgr_domain.cpp
    lgr_physics.cpp
    lgr_element_specific.cpp
    lgr_meshing.cpp
    lgr_memory_pool.cpp
    lgr_stabilized.cpp
    lgr_adapt.cpp
    lgr_threads.cpp
    )

option(LGR_ENABLE_EXODUS "Build the Exodus reader" OFF)

if (LGR_ENABLE_EXODUS)
  find_package(SEACASExodus REQUIRED)
  set(LGR_SOURCES ${LGR_SOURCES} lgr_exodus.cpp)
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
