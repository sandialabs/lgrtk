configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/LGRConfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/LGRConfig.hpp
  @ONLY)

SET(SOURCES
  MeshIO.cpp
  Run.cpp
  ParseInput.cpp
  AdaptRecon.cpp
  LowRmPotentialSolve.cpp
  LowRmRLCCircuit.cpp
  ElastostaticSolve.cpp
  LinearElasticMaterial.cpp
  VizOutput.cpp
  CmdLineOutput.cpp
  Basis.cpp
  CellTools.cpp
  ConductivityModels.cpp
  CrsMatrix.cpp
  Cubature.cpp
  ExactSolution.cpp
  ExplicitFunctors.cpp
  ExplicitLambdas.cpp
  Driver.cpp
  FEMesh.cpp
  FieldDB.cpp
  Fields.cpp
  InitialConditions.cpp
  LagrangianFineScale.cpp
  LagrangianNodalPressure.cpp
  LagrangianTimeIntegration.cpp
  MaterialModels.cpp
  IdealGas.cpp
  NeoHookean.cpp
  J2Plasticity.cpp
  MieGruneisen.cpp
  MatrixIO.cpp
  MeshFixture.cpp
  ParallelComm.cpp
  VectorContribution.cpp
  BoundaryCondition.cpp
  ContactForce.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_library(
  lgrtk
  ${SOURCES}
)
# for LGRConfig.hpp
target_include_directories(lgrtk PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

set(LINK_LIBS
    kokkoscore
    kokkoscontainers
    teuchoscore
    teuchoscomm
    teuchosparameterlist
    teuchosparser
    Omega_h::omega_h
    ${Trilinos_TPL_LIBRARIES}
    ${Trilinos_EXTRA_LD_FLAGS}
    ${SOLVER_INTERFACE_LIBRARIES}
    )

message("-- LINK_LIBS: ${LINK_LIBS}")

target_link_libraries(lgrtk PUBLIC ${LINK_LIBS})
IF( AMGX_FOUND )
  target_include_directories(lgrtk PUBLIC ${AMGX_INCLUDE_DIRS})
  target_include_directories(lgrtk PUBLIC ${PROJECT_SOURCE_DIR}/src/solverInterfaces/AmgX/)
ENDIF()

IF( VIENNACL_FOUND )
  target_include_directories(lgrtk PUBLIC ${VIENNACL_INCLUDE_DIRS})
  target_include_directories(lgrtk PUBLIC ${PROJECT_SOURCE_DIR}/src/solverInterfaces/ViennaCL/)
ENDIF()
