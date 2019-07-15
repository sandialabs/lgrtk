/*
 * PlatoMathHelpersTest.cpp
 *
 *  Created on: July 11, 2018
 */

#include <vector>

//#define COMPUTE_GOLD_
#ifdef COMPUTE_GOLD_
  #include <iostream>
  #include <fstream>
#endif

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "plato/PlatoMathHelpers.hpp"
#include "plato/Mechanics.hpp"
#include "plato/PhysicsScalarFunction.hpp"
#include "plato/VectorFunction.hpp"
#include "plato/ApplyProjection.hpp"
#include "plato/HyperbolicTangentProjection.hpp"

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"

#include <Omega_h_mesh.hpp>

namespace PlatoUnitTests
{


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_InvertLocalMatrices)
{
    const int N = 3; // Number of matrices to invert
    Plato::ScalarArray3D tMatrix("Matrix A", N, 2, 2);
    auto tHostMatrix = Kokkos::create_mirror(tMatrix);
    for (unsigned int i = 0; i < N; ++i)
    {
      const Plato::Scalar tScaleFactor = 1.0 / (1.0 + i);
      tHostMatrix(i,0,0) = -2.0 * tScaleFactor;
      tHostMatrix(i,1,0) =  1.0 * tScaleFactor;
      tHostMatrix(i,0,1) =  1.5 * tScaleFactor;
      tHostMatrix(i,1,1) = -0.5 * tScaleFactor;
    }
    Kokkos::deep_copy(tMatrix, tHostMatrix);

    Plato::ScalarArray3D tAInverse("A Inverse", N, 2, 2);
    auto tHostAInverse = Kokkos::create_mirror(tAInverse);
    for (unsigned int i = 0; i < N; ++i)
    {
      tHostAInverse(i,0,0) = 1.0;
      tHostAInverse(i,1,0) = 0.0;
      tHostAInverse(i,0,1) = 0.0;
      tHostAInverse(i,1,1) = 1.0;
    }
    Kokkos::deep_copy(tAInverse, tHostAInverse);

    using namespace KokkosBatched::Experimental;

    /// [template]AlgoType: Unblocked, Blocked, CompatMKL
    /// [in/out]A: 2d view
    /// [in]tiny: a magnitude scalar value compatible to the value type of A
    /// int SerialLU<Algo::LU::Unblocked>::invoke(const AViewType &A, const ScalarType tiny = 0)

    /// [template]SideType: Side::Left or Side::Right
    /// [template]UploType: Uplo::Upper or Uplo::Lower
    /// [template]TransType: Trans::NoTranspose or Trans::Transpose
    /// [template]DiagType: Diag::Unit or Diag::NonUnit
    /// [template]AlgoType: Unblocked, Blocked, CompatMKL
    /// [in]alpha: a scalar value
    /// [in]A: 2d view
    /// [in/out]B: 2d view
    /// int SerialTrsm<SideType,UploType,TransType,DiagType,AlgoType>
    ///    ::invoke(const ScalarType alpha, const AViewType &A, const BViewType &B);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,N), LAMBDA_EXPRESSION(const Plato::OrdinalType & n) {
      auto A    = Kokkos::subview(tMatrix  , n, Kokkos::ALL(), Kokkos::ALL());
      auto Ainv = Kokkos::subview(tAInverse, n, Kokkos::ALL(), Kokkos::ALL());

      SerialLU<Algo::LU::Blocked>::invoke(A);
      SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit   ,Algo::Trsm::Blocked>::invoke(1.0, A, Ainv);
      SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(1.0, A, Ainv);
    });

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar> > tGoldMatrixInverse = { {1.0, 3.0}, {2.0, 4.0} };

    Kokkos::deep_copy(tHostAInverse, tAInverse);
    for (unsigned int n = 0; n < N; ++n)
      for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
          {
            //printf("Matrix %d Inverse (%d,%d) = %f\n", n, i, j, tHostAInverse(n, i, j));
            const Plato::Scalar tScaleFactor = (1.0 + n);
            TEST_FLOATING_EQUALITY(tHostAInverse(n, i, j), tScaleFactor * tGoldMatrixInverse[i][j], tTolerance);
          }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, HyperbolicTangentProjection)
{
    const Plato::OrdinalType tNumNodesPerCell = 2;
    typedef Sacado::Fad::SFad<Plato::Scalar, tNumNodesPerCell> FadType;

    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tOutputVal("OutputVal", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tOutputGrad("OutputGrad", tNumNodesPerCell);
    Plato::ScalarMultiVectorT<FadType> tControl("Control", tNumCells, tNumNodesPerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tControl(aCellOrdinal, 0) = FadType(tNumNodesPerCell, 0, 1.0);
        tControl(aCellOrdinal, 1) = FadType(tNumNodesPerCell, 1, 1.0);
    }, "Set Controls");

    // SET EVALUATION TYPES FOR UNIT TEST
    Plato::HyperbolicTangentProjection tProjection;
    Plato::ApplyProjection<Plato::HyperbolicTangentProjection> tApplyProjection(tProjection);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        FadType tValue = tApplyProjection(aCellOrdinal, tControl);
        tOutputVal(aCellOrdinal) = tValue.val();
        tOutputGrad(0) = tValue.dx(0);
        tOutputGrad(1) = tValue.dx(1);
    }, "UnitTest: HyperbolicTangentProjection_GradZ");

    // TEST OUTPUT
    auto tHostVal = Kokkos::create_mirror(tOutputVal);
    Kokkos::deep_copy(tHostVal, tOutputVal);
    auto tHostGrad = Kokkos::create_mirror(tOutputGrad);
    Kokkos::deep_copy(tHostGrad, tOutputGrad);

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldVal = { 1.0 };
    std::vector<Plato::Scalar> tGoldGrad = { 4.539992985607449e-4, 4.539992985607449e-4 };
    TEST_FLOATING_EQUALITY(tHostVal(0), tGoldVal[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(0), tGoldGrad[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(1), tGoldGrad[1], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_ConditionalExpression)
{
    const Plato::OrdinalType tRange = 1;
    Plato::ScalarVector tOuput("Output", 2 /* number of outputs */);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), LAMBDA_EXPRESSION(Plato::OrdinalType tOrdinal)
    {
        Plato::Scalar tConditionalValOne = 5;
        Plato::Scalar tConditionalValTwo = 4;
        Plato::Scalar tConsequentValOne = 2;
        Plato::Scalar tConsequentValTwo = 3;
        tOuput(tOrdinal) = Plato::conditional_expression(tConditionalValOne, tConditionalValTwo, tConsequentValOne, tConsequentValTwo);

        tConditionalValOne = 3;
        tOuput(tOrdinal + 1) = Plato::conditional_expression(tConditionalValOne, tConditionalValTwo, tConsequentValOne, tConsequentValTwo);
    }, "Test inline conditional_expression function");

    auto tHostOuput = Kokkos::create_mirror(tOuput);
    Kokkos::deep_copy(tHostOuput, tOuput);
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tHostOuput(0), 3.0, tTolerance);
    TEST_FLOATING_EQUALITY(tHostOuput(1), 2.0, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_dot)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::fill(1.0, tVecA);
  Plato::ScalarVector tVecB("Vec B", tNumElems);
  Plato::fill(2.0, tVecB);

  const Plato::Scalar tOutput = Plato::dot(tVecA, tVecB);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(20., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_norm)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::fill(1.0, tVecA);

  const Plato::Scalar tOutput = Plato::norm(tVecA);
  constexpr Plato::Scalar tTolerance = 1e-6;
  TEST_FLOATING_EQUALITY(3.16227766016838, tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_sum)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec", tNumElems);
  Plato::fill(1.0, tVecA);

  Plato::Scalar tOutput = 0.0;
  Plato::local_sum(tVecA, tOutput);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(10., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_fill)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::fill(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_copy)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::fill(2.0, tSomeVector);

  Plato::ScalarVector tSomeOtherVector("some other vector", numVerts);
  Plato::copy(tSomeVector, tSomeOtherVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  auto tSomeOtherVectorHost = Kokkos::create_mirror_view(tSomeOtherVector);
  Kokkos::deep_copy(tSomeOtherVectorHost, tSomeOtherVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), tSomeOtherVectorHost(0), 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), tSomeOtherVectorHost(numVerts-1), 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_scale)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::fill(1.0, tSomeVector);
  Plato::scale(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_update)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tVector_A("vector a", numVerts);
  Plato::ScalarVector tVector_B("vector b", numVerts);
  Plato::fill(1.0, tVector_A);
  Plato::fill(2.0, tVector_B);
  Plato::update(2.0, tVector_A, 3.0, tVector_B);

  auto tVector_B_Host = Kokkos::create_mirror_view(tVector_B);
  Kokkos::deep_copy(tVector_B_Host, tVector_B);
  TEST_FLOATING_EQUALITY(tVector_B_Host(0), 8.0, 1e-17);
  TEST_FLOATING_EQUALITY(tVector_B_Host(numVerts-1), 8.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_MatrixTimesVectorPlusVector)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( mesh->nverts(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);

  // create mesh based displacement from host data
  //
  auto stateSize = spaceDim*mesh->nverts();
  Plato::ScalarVector u("state",stateSize);
  auto u_host = Kokkos::create_mirror_view(u);
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( int i = 0; i<stateSize; i++) u_host(i) = (disp += dval);
  Kokkos::deep_copy(u, u_host);

  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elastostatics'/>      \n"
    "  <Parameter name='Objective' type='string' value='My Internal Elastic Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='My Internal Elastic Energy'>                              \n"
    "    <Parameter name='Type' type='string' value='Scalar Function'/>            \n"
    "    <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Elastostatics'>                                        \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Model'>                                       \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                           \n"
    "      <Parameter name='Poissons Ratio' type='double' value='0.3'/>            \n"
    "      <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>          \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create objective
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Plato::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(*mesh, tMeshSets, tDataMap, *tParams, tParams->get<std::string>("Objective"));

  auto dfdx = eeScalarFunction.gradient_x(u,z);

  // create PDE constraint
  //
  Plato::VectorFunction<::Plato::Mechanics<spaceDim>>
    esVectorFunction(*mesh, tMeshSets, tDataMap, *tParams, tParams->get<std::string>("PDE Constraint"));

  auto dgdx = esVectorFunction.gradient_x(u,z);

#ifdef COMPUTE_GOLD_
  {
    auto dfdxHost = Kokkos::create_mirror_view(dfdx); 
    Kokkos::deep_copy(dfdxHost, dfdx);
    std::ofstream ofile;
    ofile.open("dfdx_before.dat");
    for(int i=0; i<dfdxHost.size(); i++) 
      ofile << std::setprecision(18) << dfdxHost(i) << std::endl;
    ofile.close();
  }
#endif

  Plato::MatrixTimesVectorPlusVector(dgdx, u, dfdx);

  auto dfdx_host = Kokkos::create_mirror_view(dfdx);
  Kokkos::deep_copy(dfdx_host, dfdx);

#ifdef COMPUTE_GOLD_
  {
    auto dfdxHost = Kokkos::create_mirror_view(dfdx); 
    Kokkos::deep_copy(dfdxHost, dfdx);
    std::ofstream ofile;
    ofile.open("dfdx_after.dat");
    for(int i=0; i<dfdxHost.size(); i++) 
      ofile << std::setprecision(18) << dfdxHost(i) << std::endl;
    ofile.close();
  }

  {
    std::ofstream ofile;
    ofile.open("u.dat");
    for(int i=0; i<u_host.size(); i++) 
      ofile << u_host(i) << std::endl;
    ofile.close();
  }

  {
    auto rowMapHost = Kokkos::create_mirror_view(dgdx->rowMap()); 
    Kokkos::deep_copy(rowMapHost, dgdx->rowMap());
    std::ofstream ofile;
    ofile.open("rowMap.dat");
    for(int i=0; i<rowMapHost.size(); i++) 
      ofile << rowMapHost(i) << std::endl;
    ofile.close();
  }

  {
    auto columnIndicesHost = Kokkos::create_mirror_view(dgdx->columnIndices());
    Kokkos::deep_copy(columnIndicesHost, dgdx->columnIndices());
    std::ofstream ofile;
    ofile.open("columnIndices.dat");
    for(int i=0; i<columnIndicesHost.size(); i++) 
      ofile << columnIndicesHost(i) << std::endl;
    ofile.close();
  }

  {
    auto entriesHost = Kokkos::create_mirror_view(dgdx->entries());
    Kokkos::deep_copy(entriesHost, dgdx->entries());
    std::ofstream ofile;
    ofile.open("entries.dat");
    for(int i=0; i<entriesHost.size(); i++) 
      ofile << std::setprecision(18) << entriesHost(i) << std::endl;
    ofile.close();
  }
#endif

  std::vector<Plato::Scalar> dfdx_gold = {
 97.7538461538461831, -62.6884615384616453, -29.1461538461538794, 
91.8173076923076934, -71.4288461538462229, 9.06923076923077076, 
6.02884615384615863, -18.8942307692307878, 30.8653846153846274, 
47.0480769230768630, 30.7442307692307608, -11.8846153846153673, 
-6.70384615384615579, 3.80192307692307541, 25.6788461538461377, 
-1.80000000000000249, 10.5923076923076849, 7.82307692307691660, 
-5.28461538461538716, 23.7230769230769098, -5.97692307692306990, 
-1.22307692307692806, 8.26153846153846061, -3.71538461538461373, 
-13.4019230769230564, 2.43461538461537774, -17.5557692307692150, 
44.2846153846154351, -5.57884615384621085, -19.1826923076923457, 
24.9403846153846658, -30.7211538461539249, -1.35000000000000453, 
3.30000000000000515, 4.96153846153846523, -1.61538461538461231, 
-0.0692307692307672085, 8.58461538461539853, -10.1769230769230710, 
-18.1442307692305782, 22.3730769230769155, 19.8634615384615891, 
-43.6442307692306954, 20.6711538461538638, 39.0692307692307992, 
-40.9730769230769027, 6.64038461538460645, 16.8865384615384571, 
-33.3403846153846359, 9.21923076923075158, -7.03269230769231868, 
5.44038461538462847, 6.82500000000002593, -15.3115384615384951, 
-5.98846153846156071, 2.59615384615385558, 3.15000000000001279, 
7.14807692307694431, -3.63461538461539790, 6.31730769230771649, 
11.7230769230769436, 15.8942307692308269, 1.21730769230769131, 
-60.2653846153847113, -29.1461538461537906, 6.47307692307693650, 
-34.1134615384616211, -19.6961538461538161, 21.7557692307692569, 
-0.230769230769231531, 0.461538461538463007, 0.876923076923080247, 
0.294230769230770239, 0.934615384615388844, 3.40961538461539604, 
-87.7384615384617348, 56.4865384615386503, -56.7057692307692918, 
13.1423076923077282, 6.58269230769232827, -12.8019230769231100};

  for(int iNode=0; iNode<int(dfdx_gold.size()); iNode++){
      TEST_FLOATING_EQUALITY(dfdx_host[iNode], dfdx_gold[iNode], 1e-13);
  }
}

} // namespace PlatoUnitTests
