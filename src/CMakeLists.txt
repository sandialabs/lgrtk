configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/LGRConfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/LGRConfig.hpp
  @ONLY)

SET(LGRV1_SOURCES
  MeshIO.cpp
  AdaptRecon.cpp
  LowRmPotentialSolve.cpp
  LowRmRLCCircuit.cpp
  VizOutput.cpp
  CmdLineOutput.cpp
  CellTools.cpp
  ConductivityModels.cpp
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
  VectorContribution.cpp
  BoundaryCondition.cpp
  ContactForce.cpp
  ParseInput.cpp
  CrsMatrix.cpp
  Basis.cpp
  Cubature.cpp
  ParallelComm.cpp
)

SET(PLATO_SOURCES
  plato/LinearElasticMaterial.cpp
  plato/LinearElectroelasticMaterial.cpp
  plato/LinearThermoelasticMaterial.cpp
  plato/HeatEquationResidual.cpp
  plato/InternalThermalEnergy.cpp
  plato/InternalThermoelasticEnergy.cpp
  plato/InternalElectroelasticEnergy.cpp
  plato/InternalElasticEnergy.cpp
  plato/TemperatureAverage.cpp
  plato/FluxPNorm.cpp
  plato/StressPNorm.cpp
  plato/EMStressPNorm.cpp
  plato/EffectiveEnergy.cpp
  plato/ThermostaticResidual.cpp
  plato/ElastostaticResidual.cpp
  plato/HeatEquationProblem.cpp
  plato/PlatoProblem.cpp
  plato/alg/ParseInput.cpp
  plato/alg/CrsMatrix.cpp
  plato/alg/Basis.cpp
  plato/alg/Cubature.cpp
  plato/alg/ParallelComm.cpp
  plato/alg/AmgXConfigs.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/src)

set(SOURCES)
if(LGR_ENABLE_PLATO)
message("adding plato sources")
set(SOURCES ${SOURCES} ${PLATO_SOURCES})
endif()

if(LGR_ENABLE_LGRV1)
message("adding v1 sources")
set(SOURCES ${SOURCES} ${LGRV1_SOURCES})
endif()

add_library( lgrtk ${SOURCES})
if( CMAKE_INSTALL_PREFIX )
  install( TARGETS lgrtk DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
endif()
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

target_link_libraries(lgrtk PUBLIC ${LINK_LIBS})
IF( AMGX_FOUND )
  target_include_directories(lgrtk PUBLIC ${AMGX_INCLUDE_DIRS})
  target_include_directories(lgrtk PUBLIC ${PROJECT_SOURCE_DIR}/src/solverInterfaces/AmgX/)
ENDIF()

IF( VIENNACL_FOUND )
  target_include_directories(lgrtk PUBLIC ${VIENNACL_INCLUDE_DIRS})
  target_include_directories(lgrtk PUBLIC ${PROJECT_SOURCE_DIR}/src/solverInterfaces/ViennaCL/)
ENDIF()
