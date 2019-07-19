
// J2 Local Plasticity Equations Unit Tests

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "plato/Plato_Diagnostics.hpp"

// #include "plato/LocalVectorFunctionInc.hpp"
// #include "plato/AbstractLocalVectorFunctionInc.hpp"
// #include "plato/J2PlasticityLocalResidual.hpp"
// #include "plato/J2PlasticityUtilities.hpp"
// #include "plato/ThermoPlasticityUtilities.hpp"
// #include "plato/SimplexPlasticity.hpp"
// #include "plato/SimplexThermoPlasticity.hpp"


namespace PlasticityTests
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_2D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    
    // Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    // auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    // tHostCauchyStrain(0, 0) = 3.0; tHostCauchyStrain(0, 1) = 2.0; tHostCauchyStrain(0, 2) = 1.0;
    // tHostCauchyStrain(0, 3) = 0.0; tHostCauchyStrain(0, 4) = 0.0; tHostCauchyStrain(0, 5) = 0.0;
    // tHostCauchyStrain(1, 0) = 0.5; tHostCauchyStrain(1, 1) = 0.2; tHostCauchyStrain(1, 2) = 0.8;
    // tHostCauchyStrain(1, 3) = 1.1; tHostCauchyStrain(1, 4) = 1.5; tHostCauchyStrain(1, 5) = 0.3;
    // tHostCauchyStrain(2, 0) = 1.64913808; tHostCauchyStrain(2, 1) = 0.61759347; tHostCauchyStrain(2, 2) = 0.33326845;
    // tHostCauchyStrain(2, 3) = 0.65938917; tHostCauchyStrain(2, 4) = -0.1840644; tHostCauchyStrain(2, 5) = 1.55789418;
    // Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    // Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    // Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    // {
    //     tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true);
    // }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-4;
    // std::vector<Plato::Scalar> tGold1 = {3.0, 2.0, 1.0};
    // auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    // Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    // for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
    //     TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityLocalResidual_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // using Residual = typename Plato::Evaluation<Plato::SimplexPlasticity<tSpaceDim>>::Residual;
    // using GlobalStateT = typename Residual::GlobalStateScalarType;
    // using LocalStateT = typename Residual::LocalStateScalarType;
    // using ConfigT = typename Residual::ConfigScalarType;
    // using ResultT = typename Residual::ResultScalarType;
    // using ControlT = typename Residual::ControlScalarType;

    // const     Plato::OrdinalType tNumCells     = tMesh->nelems();
    // constexpr Plato::OrdinalType tDofsPerCell  = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    // constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // // Create configuration workset
    // Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    // Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    // tWorksetBase.worksetConfig(tConfigWS);

    // // Create control workset
    // const Plato::OrdinalType tNumVerts = tMesh->nverts();
    // Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    // Plato::ScalarVector tControl("Controls", tNumVerts);
    // Plato::fill(1.0, tControl);
    // tWorksetBase.worksetControl(tControl, tControlWS);

    // // Create state workset
    // const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    // Plato::ScalarVector tState("States", tNumDofs);
    // Plato::fill(0.1, tState);
    // //Plato::fill(0.0, tState);
    // Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    //         { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");
    // Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    // tWorksetBase.worksetState(tState, tStateWS);

    // // Create result/output workset
    // Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // // ALLOCATE PLATO CRITERION
    // Plato::DataMap tDataMap;
    // Omega_h::MeshSets tMeshSets;
    

    // tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);


    // constexpr Plato::Scalar tTolerance = 1e-4;
    // std::vector<Plato::Scalar> tGold = {158.526064959, 4.77842781597};
    // auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    // Kokkos::deep_copy(tHostResultWS, tResultWS);
    // for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    // {
    //     TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    // }

    // auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    // TEST_FLOATING_EQUALITY(163.304492775, tObjFuncVal, tTolerance);
}


} // namespace AugLagStressTest
