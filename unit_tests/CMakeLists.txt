if (NOT LGR_ENABLE_PLATO)
  set(SOURCES
    LGRTestHelpers.cpp
    CrsMatrixTests.cpp
    InitialConditionTests.cpp
    FieldDB.cpp
    IdealGas.cpp
    LowRmPotentialSolveTests.cpp
    LowRmRLCCircuitTests.cpp
    MatrixIOTests.cpp
    SampleTests.cpp
    SimplexCellToolsTests.cpp
    SimplexCubatureTests.cpp
    Teuchos_StandardUnitTestMain.cpp
    ConductivityModels.cpp
    TwoDMesh.cpp
    ElementHelpersReplacements.cpp
    mhdTests.cpp
    mhdPolynomialTests.cpp
    mhdMatrixAssemblyTests
    R3D.cpp
  )

  IF (AMGX_FOUND)
    set(SOURCES
        ${SOURCES}
        AmgXTests.cpp
  )
  ENDIF()

  IF (ViennaCL_FOUND)
    set(SOURCES
        ${SOURCES}
        ViennaCLTests.cpp
  )
  ENDIF()

  set(HEADERS
    LGRTestHelpers.hpp
  )

  add_executable(UnitTests
    ${SOURCES}
    ${HEADERS}
  )

  target_link_libraries(UnitTests PUBLIC lgrtk)
  target_include_directories(UnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/UnitTests)
  add_test(NAME runUnitTests COMMAND ${MPI_TEST})
endif()

if (LGR_ENABLE_PLATO)

#begin: Structural dynamics unit test
  set(StructuralDynamics_SOURCES
    StructuralDynamicsTest.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(StructuralDynamics_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(StructuralDynamicsUnitTests
    ${StructuralDynamics_SOURCES}
    ${StructuralDynamics_HEADERS}
  )

  target_link_libraries(StructuralDynamicsUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(StructuralDynamicsUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/StructuralDynamicsUnitTests)
  add_test(NAME runStructuralDynamicsUnitTests COMMAND ${ES_MPI_TEST})
#end: Structural dynamics unit test

#begin: Hamilton-Jacobi unit test
  set(HamiltonJacobi_SOURCES
    HamiltonJacobiTest.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(HamiltonJacobi_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(HamiltonJacobiUnitTests
    ${HamiltonJacobi_SOURCES}
    ${HamiltonJacobi_HEADERS}
  )

  target_link_libraries(HamiltonJacobiUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(HamiltonJacobiUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/HamiltonJacobiUnitTests)
  add_test(NAME runHamiltonJacobiUnitTests COMMAND ${ES_MPI_TEST})
#end: Hamilton-Jacobi unit test

#begin: Algebraic rocket model unit test
  set(AlgebraicRocket_SOURCES
    AlgebraicRocketTest.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(AlgebraicRocket_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(AlgebraicRocketUnitTests
    ${AlgebraicRocket_SOURCES}
    ${AlgebraicRocket_HEADERS}
  )

  target_link_libraries(AlgebraicRocketUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(AlgebraicRocketUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/AlgebraicRocketUnitTests)
  add_test(NAME runAlgebraicRocketUnitTests COMMAND ${ES_MPI_TEST})
#end: Algebraic rocket model unit test

#begin: AugLagStressTest unit test
  set(AugLagStressTest_SOURCES
    PlatoAugLagStressTest.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(AugLagStressTest_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(AugLagStressUnitTests
    ${AugLagStressTest_SOURCES}
    ${AugLagStressTest_HEADERS}
  )

  target_link_libraries(AugLagStressUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(AugLagStressUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/AugLagStressUnitTests)
  add_test(NAME runAugLagStressUnitTests COMMAND ${ES_MPI_TEST})
#end: AugLagStressTest unit test

#begin: MassPropertiesTest unit test
  set(MassPropertiesTest_SOURCES
    MassPropertiesTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(MassPropertiesTest_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(MassPropertiesUnitTests
    ${MassPropertiesTest_SOURCES}
    ${MassPropertiesTest_HEADERS}
  )

  target_link_libraries(MassPropertiesUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(MassPropertiesUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/MassPropertiesUnitTests)
  add_test(NAME runMassPropertiesUnitTests COMMAND ${ES_MPI_TEST})
#end: MassPropertiesTest unit test

#begin: PlatoMathHelpers unit test
  set(PlatoMathHelpers_SOURCES
    PlatoMathHelpersTest.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(PlatoMathHelpers_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(PlatoMathHelpersUnitTests
    ${PlatoMathHelpers_SOURCES}
    ${PlatoMathHelpers_HEADERS}
  )

  target_link_libraries(PlatoMathHelpersUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES}
    kokkoskernels
    )
  target_include_directories(PlatoMathHelpersUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/PlatoMathHelpersUnitTests)
  add_test(NAME runPlatoMathHelpersUnitTests COMMAND ${ES_MPI_TEST})
#end: PlatoMathHelpers unit test

  set(PlatoFunctors_SOURCES
    PlatoFunctorsTest.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(PlatoFunctors_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(PlatoFunctorsUnitTests
    ${PlatoFunctors_SOURCES}
    ${PlatoFunctors_HEADERS}
  )

  target_link_libraries(PlatoFunctorsUnitTests
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(PlatoFunctorsUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/PlatoFunctorsUnitTests)
  add_test(NAME runPlatoFunctorsUnitTests COMMAND ${ES_MPI_TEST})

  set(Derivatives_SOURCES
    DerivativeTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(HeatEquationTests_SOURCES
    HeatEquationTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(Derivatives_HEADERS
    PlatoTestHelpers.hpp
  )

  set(HeatEquationTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(DerivativeUnitTests
    ${Derivatives_SOURCES}
    ${Derivatives_HEADERS}
  )

  add_executable(HeatEquationUnitTests
    ${HeatEquationTests_SOURCES}
    ${HeatEquationTests_HEADERS}
  )

  set(ElectroelasticTests_SOURCES
    ElectroelasticTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(ElectroelasticTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(ElectroelasticUnitTests
    ${ElectroelasticTests_SOURCES}
    ${ElectroelasticTests_HEADERS}
  )

  target_link_libraries(ElectroelasticUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(ElectroelasticUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ElectroelasticUnitTests)
  add_test(NAME runElectroelasticUnitTests COMMAND ${ES_MPI_TEST})

  set(ThermoelasticTests_SOURCES
    ThermoelasticTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(ThermoelasticTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(ThermoelasticUnitTests
    ${ThermoelasticTests_SOURCES}
    ${ThermoelasticTests_HEADERS}
  )

  target_link_libraries(ThermoelasticUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(ThermoelasticUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ThermoelasticUnitTests)
  add_test(NAME runThermoelasticUnitTests COMMAND ${ES_MPI_TEST})

  set(TransientThermomechTests_SOURCES
    TransientThermomechTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(TransientThermomechTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(TransientThermomechUnitTests
    ${TransientThermomechTests_SOURCES}
    ${TransientThermomechTests_HEADERS}
  )

  target_link_libraries(TransientThermomechUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(TransientThermomechUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/TransientThermomechUnitTests)
  add_test(NAME runTransientThermomechUnitTests COMMAND ${ES_MPI_TEST})

  target_link_libraries(DerivativeUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(DerivativeUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/DerivativeUnitTests)
  add_test(NAME runDerivativeUnitTests COMMAND ${ES_MPI_TEST})

  target_link_libraries(HeatEquationUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(HeatEquationUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/HeatEquationUnitTests)
  add_test(NAME runHeatEquationUnitTests COMMAND ${ES_MPI_TEST})

  if(LGR_ENABLE_AD_TEST AND NOT AMGX_FOUND)
    set(AD_Tests_SOURCES
      ad_test.cpp
    )
    add_executable(ADTest
      ${AD_Tests_SOURCES}
    )
    target_link_libraries(ADTest ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${FASTPOSIT_LIBRARIES})
    target_include_directories(ADTest PRIVATE ${Trilinos_INCLUDE_DIRS} ${FASTPOSIT_INCLUDE_DIRS})
  endif()
endif()

if (LGR_ENABLE_PLATO_MPMD)
  set(LGRApp_HEADERS
    ${CMAKE_SOURCE_DIR}/src/plato/LGR_App.hpp
    PlatoTestHelpers.hpp
  )

  set(LGRApp_SOURCES
    LGRAppIntxTests.cpp
    LGRAppUnitTestMain.cpp
  )

  add_executable(LGRAppIntxTests
    ${LGRApp_SOURCES}
    ${LGRApp_HEADERS}
  )

  target_link_libraries(LGRAppIntxTests 
    LGR_App
    lgrtk
    ${PLATO_LIBS}
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )

  target_include_directories(LGRAppIntxTests PRIVATE "${PLATO_PREFIX}/include")
  target_include_directories(LGRAppIntxTests PRIVATE "${CMAKE_SOURCE_DIR}/src")
endif()
