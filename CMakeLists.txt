cmake_minimum_required(VERSION 3.12.0...${CMAKE_VERSION})

project(LGR VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-Weverything -Werror -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-float-equal")

add_executable(lgr
    lgr.cpp
    lgr_print.cpp
    lgr_vtk.cpp
    lgr_domain.cpp
    lgr_physics.cpp
    lgr_element_specific.cpp
    lgr_meshing.cpp
    )
set_property(TARGET lgr PROPERTY CXX_STANDARD "14")
set_property(TARGET lgr PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET lgr PROPERTY CXX_EXTENSIONS OFF)
target_include_directories(lgr
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
