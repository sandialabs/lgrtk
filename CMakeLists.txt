cmake_minimum_required(VERSION 3.0)

project(LGR VERSION 1.0.0 LANGUAGES CXX)
MESSAGE("\n\nConfiguring LGR Version ${LGR_VERSION_MAJOR}.${LGR_VERSION_MINOR}.${LGR_VERSION_PATCH}")

IF (DEFINED Trilinos_PREFIX)
  SET(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
ENDIF()

#silences warnings about the system not supporting shared libraries
#http://stackoverflow.com/questions/12264299/cmake-on-linux-target-platform-does-not-support-dynamic-linking
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Get Trilinos as one entity
FIND_PACKAGE(Trilinos REQUIRED)

IF(NOT Trilinos_FOUND)
  MESSAGE(FATAL_ERROR "Cannot find Trilinos - build exiting!!!")
ENDIF()

# Echo trilinos build info just for fun
MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_PREFIX = ${Trilinos_PREFIX}")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

# Important: We want to set the compilers and build flags from the Trilinos build. We do not need to check for valid compilers
# etc here in the LGR build, as it has already been done in the Trilinos build. So, we force the compilers here
# to be the same as the Trilinos build. We need the project statement after we force the compilers, or the compiler testing
# will occur.

set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
set(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})

set(THREAD_COUNT "" CACHE STRING "Default number of threads to use")
if(THREAD_COUNT)
  set(ALL_THREAD_ARGS "--kokkos-threads=${THREAD_COUNT}")
else()
  set(ALL_THREAD_ARGS "")
endif()

MESSAGE("-- BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")

SET(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS} )
SET(CMAKE_C_FLAGS ${Trilinos_C_COMPILER_FLAGS} )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUSTOM_CXX_FLAGS}")

MESSAGE("-- CXX Flags being used in this build are : ${CMAKE_CXX_FLAGS}")

# disallow in-source builds

get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)

get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)

if(${srcdir} STREQUAL ${bindir})

    message("######################################################")

    message("You are attempting to build in your Source Directory.")

    message("You must run cmake from a build directory.")

    message("######################################################")

    message(FATAL_ERROR "In-source builds are forbidden!")

endif()

FIND_PACKAGE(KokkosCore REQUIRED)
# for StaticCrsGraph
FIND_PACKAGE(KokkosContainers REQUIRED)
FIND_PACKAGE(Teuchos REQUIRED)

SET(OMEGA_H_PREFIX "" CACHE PATH "Omega_h install location")
IF(OMEGA_H_PREFIX)
  SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OMEGA_H_PREFIX})
ENDIF()
FIND_PACKAGE(Omega_h 9.12.0 REQUIRED)

FIND_PROGRAM(
  VTKDIFF
  NAMES vtkdiff
     PATHS ${OMEGA_H_PREFIX}/bin
     NO_DEFAULT_PATH
  )

IF(VTKDIFF)
  MESSAGE("-- VTKDIFF=${VTKDIFF}")
ELSE()
  MESSAGE(FATAL_ERROR "Cannot locate the executable for vtkdiff, exiting")
ENDIF()

# DAI: omega_h2 has a vtkdiff which by default compares all timesteps.
# old omega_h's vtkdiff only compared one timestep.
# at the time of transition, we will keep comparing just one time step
# this flag is needed to do that.
SET(VTKDIFF "${VTKDIFF} -onestep")

# We need to do this better, but for now lets see how well it works to find the right mpirun
# This needs to be the one on the front end for Cuda and MIC machines - it runs the utilities.

#SET(MPIEXEC mpiexec.hydra -hosts mic0 -env OMP_NUM_THREADS 56 -env KMP_AFFINITY=balanced -binding domain=omp)
IF(LGR_MPIEXEC)
  MESSAGE("-- User defined LGR_MPIEXEC=${LGR_MPIEXEC}")
  SEPARATE_ARGUMENTS(LGR_MPIEXEC)
ELSE()
  SET(LGR_MPIEXEC)

  #mpirun is frequently found alongside mpicxx, try looking for it there.
  get_filename_component(COMPILER_DIR ${CMAKE_CXX_COMPILER} PATH)

  FIND_FILE(
    LGR_MPIEXEC
    NAMES mpirun mpiexec
    PATHS "${COMPILER_DIR}" ENV PATH
    )

  IF(LGR_MPIEXEC)
    MESSAGE("-- Found LGR_MPIEXEC=${LGR_MPIEXEC}")
  ELSE()
    MESSAGE(FATAL_ERROR "Fatal Error: Cannot find mpirun or mpiexec for LGR_MPIEXEC")
  ENDIF()
ENDIF()

option( LGR_CHECK_FPE "Flag to enable LGR checks for floating point exceptions" OFF)
if ( LGR_CHECK_FPE)
  add_definitions(-DLGR_CHECK_FPE)
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    add_definitions(-DLGR_USE_GNU_FPE)
    message("-- Building with GNU FPE checking enabled.")
  else()
    message("-- Building with XMM FPE checking enabled.")
  endif()
else()
  message("-- Building without FPE checking.")
endif()

add_executable(
  lgr
  src/LGR.cpp
)
target_include_directories(lgr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

IF (DEFINED AMGX_PREFIX)
  ADD_DEFINITIONS(-DHAVE_AMGX)
  FIND_PATH(AMGX_INCLUDE_DIR NAMES amgx_c.h PATHS ${AMGX_PREFIX}/include)
  IF (NOT AMGX_INCLUDE_DIR)
    MESSAGE(FATAL_ERROR "Couldn't find AMGX header!")
  ENDIF()
  FIND_LIBRARY(AMGX_LIB NAMES amgxsh amgx PATHS ${AMGX_PREFIX}/lib)
  IF (NOT AMGX_LIB)
    MESSAGE(FATAL_ERROR "Couldn't find AMGX library!")
  ENDIF()
  SET(AMGX_INCLUDE_DIRS "${AMGX_INCLUDE_DIR}")
  SET(AMGX_LIBRARIES "${AMGX_LIB}")
  SET(AMGX_FOUND TRUE) # a bit of a hack; would be better to check that it actually exists!
  MESSAGE("-- Found AmgX!")
  MESSAGE("-- AMGX_INCLUDE_DIRS = ${AMGX_INCLUDE_DIRS}")
  MESSAGE("-- AMGX_LIBRARIES = ${AMGX_LIBRARIES}")
ENDIF()

IF (DEFINED ViennaCL_PREFIX)
  ADD_DEFINITIONS(-DHAVE_VIENNA_CL)
  SET(CMAKE_PREFIX_PATH ${ViennaCL_PREFIX} ${CMAKE_PREFIX_PATH})
  IF(VIENNACL_SKIP_FIND_PACKAGE)
    SET(VIENNACL_INCLUDE_DIRS ${ViennaCL_PREFIX}/include)
    SET(VIENNACL_LIBRARIES)  # will need/want to add OpenCL here if ever we support that
    SET(VIENNACL_FOUND TRUE) # a bit of a hack
  ELSE()
# for some reason, using the following line on my Mac causes the cmake configure to fail...
    FIND_PACKAGE(ViennaCL)
  ENDIF()
  IF (VIENNACL_FOUND)
    MESSAGE("-- Found ViennaCL!")
    MESSAGE("-- VIENNACL_INCLUDE_DIRS = ${VIENNACL_INCLUDE_DIRS}")
    MESSAGE("-- VIENNACL_LIBRARIES = ${VIENNACL_LIBRARIES}")
  ENDIF()
ENDIF()

SET(SOLVER_INTERFACE_LIBRARIES)

IF( AMGX_FOUND )
  SET(SOLVER_INTERFACE_LIBRARIES
      ${SOLVER_INTERFACE_LIBRARIES}
      ${AMGX_LIBRARIES}
     )
ENDIF()

IF( VIENNACL_FOUND )
  SET(SOLVER_INTERFACE_LIBRARIES
      ${SOLVER_INTERFACE_LIBRARIES}
      ${VIENNACL_LIBRARIES}
     )
ENDIF()

ADD_SUBDIRECTORY(src)

if (DEFINED Enable_MPMD)
  message(FATAL_ERROR "Enable_MPMD was renamed LGR_ENABLE_PLATO")
endif()
if (DEFINED PlatoEngine_INSTALL_DIR)
  message(FATAL_ERROR "PlatoEngine_INSTALL_DIR was renamed PLATO_PREFIX")
endif()
option(LGR_ENABLE_PLATO "Enable LGR Plato driver" OFF)
option(LGR_ENABLE_PLATO_MPMD "Enable the Plato interface MPMD executable" OFF)
if (LGR_ENABLE_PLATO_MPMD)
  FIND_PACKAGE(Epetra REQUIRED)
  set(PLATO_PREFIX "" CACHE STRING "Path to Plato install")
  find_library(PLATO_DATA_LAYER_LIB NAMES PlatoDataLayer PATHS ${PLATO_PREFIX}/lib)
  find_library(PLATO_INTERFACE_LIB NAMES PlatoInterface PATHS ${PLATO_PREFIX}/lib)
  find_library(PLATO_ANALYZE_LIB NAMES PlatoAnalyze PATHS ${PLATO_PREFIX}/lib)
  find_library(PLATO_PARSER_LIB NAMES PlatoParser PATHS ${PLATO_PREFIX}/lib)
  find_library(PLATO_EXCEPTIONS_LIB NAMES PlatoExceptions PATHS ${PLATO_PREFIX}/lib)
  find_package(Epetra REQUIRED)
  find_package(AztecOO REQUIRED)
  find_package(SEACASExodus REQUIRED)
  find_package(SEACASNemesis REQUIRED)
  find_package(Shards REQUIRED)
  find_package(Intrepid REQUIRED)
  find_package(TeuchosCore REQUIRED)
  find_package(STKMesh REQUIRED)
  add_executable(
    LGR_MPMD
    src/plato/LGR_App.cpp
    src/plato/LGR_MPMD.cpp
  )
  target_include_directories(LGR_MPMD PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(LGR_MPMD PRIVATE "${PLATO_PREFIX}/include")
  set(PLATO_LIBS
      ${PLATO_DATA_LAYER_LIB}
      ${PLATO_INTERFACE_LIB}
      ${PLATO_ANALYZE_LIB}
      ${PLATO_PARSER_LIB}
      ${PLATO_EXCEPTIONS_LIB}
      stk_mesh_base
      aztecoo
      epetra
      intrepid
      shards
      sacado
      teuchoscore
      nemesis
      exodus
     )

  target_link_libraries(LGR_MPMD ${PLATO_LIBS} ${PLATO_LIBS} lgrtk)

  option(LGR_ENABLE_PLATO_PYTHON "Enable the Plato Python interface" OFF)
  if (LGR_ENABLE_PLATO_PYTHON)
      find_package(PythonLibs)
      include_directories( ${PYTHON_INCLUDE_DIRS} )
      add_library( Plato SHARED src/plato/LGR_App.cpp src/plato/PlatoPythonModule.cpp )
      target_include_directories( Plato PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_include_directories( Plato PRIVATE "${PLATO_PREFIX}/include")
      target_link_libraries( Plato lgrtk ${PLATO_LIBS} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${PYTHON_LIBRARIES} )
      set_target_properties( Plato PROPERTIES PREFIX "" )
      target_compile_options( Plato PRIVATE "-lmpi" )
  endif()
endif()

list(REVERSE Trilinos_TPL_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
list(REVERSE Trilinos_TPL_LIBRARIES)
message(STATUS "Trilinos_TPL_LIBRARIES: ${Trilinos_TPL_LIBRARIES}")
message(STATUS "Trilinos_EXTRA_LD_FLAGS: ${Trilinos_EXTRA_LD_FLAGS}")

target_link_libraries(lgr lgrtk)

ENABLE_TESTING()
INCLUDE(CTest)

SET(LGR_RANKS_PER_NODE_FLAG "-npernode" CACHE STRING "Processes per node flag for LGR_MPIEXEC")
SET(LGR_RANKS_FLAG "-n" CACHE STRING "Number of processes flag for LGR_MPIEXEC")
SET(LGR_RANKS_PER_NODE "1" CACHE STRING "Maximum number of processes per node")
SET(LGR_VALGRIND "" CACHE STRING "Optional Valgrind call and arguments")

function(build_mpi_test_string RESULT_NAME num_mpi_proc)
  string(REPLACE " " ";" VALGRIND "${LGR_VALGRIND}")
  set(${RESULT_NAME} ${LGR_MPIEXEC} ${LGR_RANKS_PER_NODE_FLAG} ${LGR_RANKS_PER_NODE} ${LGR_RANKS_FLAG} ${num_mpi_proc} ${VALGRIND} ${ARGN} PARENT_SCOPE)
endfunction(build_mpi_test_string)

ADD_SUBDIRECTORY(unit_tests)
ADD_SUBDIRECTORY(tests)

set(LGR_KEY_INTS
    LGR_VERSION_MAJOR
    LGR_VERSION_MINOR
    LGR_VERSION_PATCH
   )
