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

if (LGR_ENABLE_PLATO)
  set(Elastostatics_SOURCES
    LGRTestHelpers.cpp
    ElastostaticSolveTests.cpp
    Teuchos_StandardUnitTestMain.cpp
  )

#begin: Structural dynamics unit test
  set(StructuralDynamics_SOURCES
    LGRTestHelpers.cpp
    StructuralDynamicsTest.cpp
    Teuchos_StandardUnitTestMain.cpp
  )

  set(StructuralDynamics_HEADERS
    LGRTestHelpers.hpp
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
    LGRTestHelpers.cpp
    HamiltonJacobiTest.cpp
    Teuchos_StandardUnitTestMain.cpp
  )

  set(HamiltonJacobi_HEADERS
    LGRTestHelpers.hpp
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
    LGRTestHelpers.cpp
    AlgebraicRocketTest.cpp
    Teuchos_StandardUnitTestMain.cpp
  )

  set(AlgebraicRocket_HEADERS
    LGRTestHelpers.hpp
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

#begin: Elastodynamics unit test
  set(Elastodynamics_SOURCES
    LGRTestHelpers.cpp
    ElastodynamicsTest.cpp
    Teuchos_StandardUnitTestMain.cpp
  )

  set(Elastodynamics_HEADERS
    LGRTestHelpers.hpp
    PlatoTestHelpers.hpp
  )

  add_executable(ElastodynamicsUnitTests
    ${Elastodynamics_SOURCES}
    ${Elastodynamics_HEADERS}
  )

  target_link_libraries(ElastodynamicsUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(ElastodynamicsUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ElastodynamicsUnitTests)
  add_test(NAME runElastodynamicsUnitTests COMMAND ${ES_MPI_TEST})
#end: Elastodynamics unit test

#begin: PlatoMathHelpers unit test
  set(PlatoMathHelpers_SOURCES
    LGRTestHelpers.cpp
    PlatoMathHelpersTest.cpp
    Teuchos_StandardUnitTestMain.cpp
  )

  set(PlatoMathHelpers_HEADERS
    LGRTestHelpers.hpp
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
    )
  target_include_directories(PlatoMathHelpersUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/PlatoMathHelpersUnitTests)
  add_test(NAME runPlatoMathHelpersUnitTests COMMAND ${ES_MPI_TEST})
#end: PlatoMathHelpers unit test

  set(PlatoFunctors_SOURCES
    LGRTestHelpers.cpp
    PlatoFunctorsTest.cpp
    Teuchos_StandardUnitTestMain.cpp
  )

  set(PlatoFunctors_HEADERS
    LGRTestHelpers.hpp
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
    LGRTestHelpers.cpp
    DerivativeTests.cpp
    Teuchos_StandardUnitTestMain.cpp
  )

  set(Elastostatics_HEADERS
    LGRTestHelpers.hpp
    PlatoTestHelpers.hpp
  )

  set(Derivatives_HEADERS
    LGRTestHelpers.hpp
  )

  add_executable(ElastostaticsUnitTests
    ${Elastostatics_SOURCES}
    ${Elastostatics_HEADERS}
  )

  add_executable(DerivativeUnitTests
    ${Derivatives_SOURCES}
    ${Derivatives_HEADERS}
  )

  target_link_libraries(ElastostaticsUnitTests ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} lgrtk)
  target_include_directories(ElastostaticsUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ElastostaticsUnitTests)
  add_test(NAME runElastostaticsUnitTests COMMAND ${ES_MPI_TEST})

  target_link_libraries(DerivativeUnitTests 
    lgrtk
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )
  target_include_directories(DerivativeUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/DerivativeUnitTests)
  add_test(NAME runDerivativeUnitTests COMMAND ${ES_MPI_TEST})
endif()

if (LGR_ENABLE_PLATO_MPMD)
  set(LGRApp_HEADERS
    ${CMAKE_SOURCE_DIR}/src/plato/LGR_App.hpp
    LGRTestHelpers.hpp
  )

  set(LGRApp_SOURCES
    ${CMAKE_SOURCE_DIR}/src/plato/LGR_App.cpp
    LGRAppIntxTests.cpp
    LGRTestHelpers.cpp
    Teuchos_LGRAppUnitTestMain.cpp
  )
  file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/IntxTests
       DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


  add_executable(LGRAppIntxTests
    ${LGRApp_SOURCES}
    ${LGRApp_HEADERS}
  )

  target_link_libraries(LGRAppIntxTests 
    lgrtk
    ${PLATO_LIBS}
    ${Trilinos_LIBRARIES} 
    ${Trilinos_TPL_LIBRARIES} 
    )

  target_include_directories(LGRAppIntxTests PRIVATE "${PLATO_PREFIX}/include")
  target_include_directories(LGRAppIntxTests PRIVATE "${CMAKE_SOURCE_DIR}/src")
endif()
