
// J2 Local Plasticity Equations Unit Tests

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "plato/Plato_Diagnostics.hpp"

#include "plato/LocalVectorFunctionInc.hpp"
// #include "plato/J2PlasticityLocalResidual.hpp"


namespace PlasticityTests
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_UpdatePlasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 8;
    
    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumVoigtTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0)/ 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    Kokkos::deep_copy(tLocalState, tPrevLocalState);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tPenalizedHardeningModulusKinematic = 3.0;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.updatePlasticStrainAndBackstressPlasticStep(tCellOrdinal, tPrevLocalState, tYieldSurfaceNormal,
                                                                       tPenalizedHardeningModulusKinematic, tLocalState);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = 
      {{tHostPrevLocalState(0,0),tHostPrevLocalState(0,1),3.9798,5.4697,8.91918,7.9596,9.9394,11.9192},
       {tHostPrevLocalState(1,0),tHostPrevLocalState(1,1),9.9192,13.8788,25.6767,19.8384,25.7576,31.6767}};
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; ++tDofIndex)
            TEST_FLOATING_EQUALITY(tHostLocalState(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_UpdatePlasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumVoigtTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0)/ 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    Kokkos::deep_copy(tLocalState, tPrevLocalState);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tPenalizedHardeningModulusKinematic = 3.0;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.updatePlasticStrainAndBackstressPlasticStep(tCellOrdinal, tPrevLocalState, tYieldSurfaceNormal,
                                                                       tPenalizedHardeningModulusKinematic, tLocalState);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = 
      {{tHostPrevLocalState(0,0),tHostPrevLocalState(0,1),3.9798,5.4697,6.9596,10.899,12.8788,14.8586,10.9596,12.9394,14.9192,16.899,18.8788,20.8586},
       {tHostPrevLocalState(1,0),tHostPrevLocalState(1,1),9.9192,13.8788,17.8384,31.5959,37.5151,43.4343,25.8384,31.7576,37.6767,43.5959,49.5151,55.4343}};
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; ++tDofIndex)
            TEST_FLOATING_EQUALITY(tHostLocalState(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_UpdateElasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 8;
    
    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.updatePlasticStrainAndBackstressElasticStep(tCellOrdinal, tPrevLocalState, tLocalState);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-10;
    Kokkos::deep_copy(tHostPrevLocalState, tPrevLocalState);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
        for(Plato::OrdinalType tDofIndex = 2; tDofIndex < tNumLocalDofsPerCell; ++tDofIndex)
            TEST_FLOATING_EQUALITY(tHostLocalState(tCellIndex, tDofIndex), 
                                   tHostPrevLocalState(tCellIndex, tDofIndex), tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_UpdateElasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.updatePlasticStrainAndBackstressElasticStep(tCellOrdinal, tPrevLocalState, tLocalState);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-10;
    Kokkos::deep_copy(tHostPrevLocalState, tPrevLocalState);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Kokkos::deep_copy(tHostLocalState, tLocalState);

    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
        for(Plato::OrdinalType tDofIndex = 2; tDofIndex < tNumLocalDofsPerCell; ++tDofIndex)
            TEST_FLOATING_EQUALITY(tHostLocalState(tCellIndex, tDofIndex), 
                                   tHostPrevLocalState(tCellIndex, tDofIndex), tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_YieldSurfaceNormal2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 8;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumVoigtTerms);
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostDeviatoricStress(i, j) = (i + 1.0) * (j + 2.0)/ 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostDeviatoricStress(i, j));
        }
    Kokkos::deep_copy(tDeviatoricStress, tHostDeviatoricStress);
    
    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumVoigtTerms);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::ScalarVector tDevStressMinusBackstressNorm("deviatoric stress minus backstress", tNumCells);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeDeviatoricStressMinusBackstressNormalized(tCellOrdinal, tDeviatoricStress, tLocalState,
                                                                       tYieldSurfaceNormal, tDevStressMinusBackstressNorm);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-0.422116,-0.482418,-0.54272};
    auto tHostDevStressMinusBackstressNorm = Kokkos::create_mirror(tDevStressMinusBackstressNorm);
    Kokkos::deep_copy(tHostDevStressMinusBackstressNorm, tDevStressMinusBackstressNorm);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    Kokkos::deep_copy(tHostYieldSurfaceNormal, tYieldSurfaceNormal);

    Plato::OrdinalType tCellIndex = 0;
    TEST_FLOATING_EQUALITY(tHostDevStressMinusBackstressNorm(tCellIndex), 13.2665, tTolerance);
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostYieldSurfaceNormal(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_YieldSurfaceNormal3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumVoigtTerms);
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostDeviatoricStress(i, j) = (i + 1.0) * (j + 2.0)/ 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostDeviatoricStress(i, j));
        }
    Kokkos::deep_copy(tDeviatoricStress, tHostDeviatoricStress);
    
    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumVoigtTerms);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::ScalarVector tDevStressMinusBackstressNorm("deviatoric stress minus backstress", tNumCells);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeDeviatoricStressMinusBackstressNormalized(tCellOrdinal, tDeviatoricStress, tLocalState,
                                                                       tYieldSurfaceNormal, tDevStressMinusBackstressNorm);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-0.25878,-0.28286,-0.30693,-0.33100,-0.35508,-0.37915};
    auto tHostDevStressMinusBackstressNorm = Kokkos::create_mirror(tDevStressMinusBackstressNorm);
    Kokkos::deep_copy(tHostDevStressMinusBackstressNorm, tDevStressMinusBackstressNorm);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    Kokkos::deep_copy(tHostYieldSurfaceNormal, tYieldSurfaceNormal);

    Plato::OrdinalType tCellIndex = 0;
    TEST_FLOATING_EQUALITY(tHostDevStressMinusBackstressNorm(tCellIndex), 33.2319, tTolerance);
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostYieldSurfaceNormal(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_DeviatoricStress2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumVoigtTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostElasticStrain(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostElasticStrain(i, j));
        }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumVoigtTerms);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tShearModulus = 3.5;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeDeviatoricStress(tCellOrdinal, tElasticStrain, tShearModulus,
                                                   tDeviatoricStress);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.0,7.0,10.5};
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    Kokkos::deep_copy(tHostDeviatoricStress, tDeviatoricStress);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostDeviatoricStress(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_DeviatoricStress3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumVoigtTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostElasticStrain(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostElasticStrain(i, j));
        }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumVoigtTerms);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tShearModulus = 3.5;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.computeDeviatoricStress(tCellOrdinal, tElasticStrain, tShearModulus,
                                                   tDeviatoricStress);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-7.0,0.0,7.0,14.0,17.5,21.0};
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    Kokkos::deep_copy(tHostDeviatoricStress, tDeviatoricStress);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostDeviatoricStress(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_PlasticStrainResidualPlasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 8;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumVoigtTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0) / 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualPlasticStep(tCellOrdinal, tLocalState, 
                                                   tPrevLocalState, tYieldSurfaceNormal, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.0101021,-0.0681803,-0.146463};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 2; tDofIndex < 2 + tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-2], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_PlasticStrainResidualPlasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumVoigtTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0) / 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualPlasticStep(tCellOrdinal, tLocalState, 
                                                   tPrevLocalState, tYieldSurfaceNormal, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.0101021,-0.0681803,-0.146463,-0.224745,-0.303027,-0.381309};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 2; tDofIndex < 2 + tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-2], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_BackstressResidualPlasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 8;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumVoigtTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0) / 5.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tHardeningModulusKinematic = 3.2;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillBackstressTensorResidualPlasticStep(tCellOrdinal, tHardeningModulusKinematic,
                                                   tLocalState, tPrevLocalState, tYieldSurfaceNormal, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {-0.0451156,1.16667,1.33333};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 5; tDofIndex < 5 + tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-5], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_BackstressResidualPlasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 1.5;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal", tNumCells, tNumVoigtTerms);
    auto tHostYieldSurfaceNormal = Kokkos::create_mirror(tYieldSurfaceNormal);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumVoigtTerms; ++j)
        {
            tHostYieldSurfaceNormal(i, j) = (i + 1.0) * (j + 2.0) / 7.0;
            //printf("(%d,%d) = %f\n", i,j,tHostYieldSurfaceNormal(i, j));
        }
    Kokkos::deep_copy(tYieldSurfaceNormal, tHostYieldSurfaceNormal);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Plato::Scalar tHardeningModulusKinematic = 3.2;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillBackstressTensorResidualPlasticStep(tCellOrdinal, tHardeningModulusKinematic,
                                                   tLocalState, tPrevLocalState, tYieldSurfaceNormal, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {3.0,3.33333,3.66666,4.0,4.33333,4.66666};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 8; tDofIndex < 8 + tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-8], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_PlasticStrainResidualElasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 8;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualElasticStep(tCellOrdinal, tLocalState, 
                                                                      tPrevLocalState, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.5,2.0/3.0,0.833333};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 2; tDofIndex < 2 + tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-2], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_PlasticStrainResidualElasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tResult("plastic strain residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualElasticStep(tCellOrdinal, tLocalState, 
                                                                      tPrevLocalState, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {0.5,0.666666,0.83333,1.0,1.16666,1.33333};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 2; tDofIndex < 2 + tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-2], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_BackstressResidualElasticStep2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 8;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tResult("backstress residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillBackstressTensorResidualElasticStep(tCellOrdinal, tLocalState, 
                                                                   tPrevLocalState, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {1.0, 1.16666, 1.33333};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 5; tDofIndex < 5 + tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-5], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, J2PlasticityUtils_BackstressResidualElasticStep3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    
    Plato::ScalarMultiVector tLocalState("local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostLocalState(i, j) = (i + 1.0) * (j + 1.0) / 2.0;
            //printf("(%d,%d) = %f\n", i,j,tHostLocalState(i, j));
        }
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    Plato::ScalarMultiVector tPrevLocalState("previous local state", tNumCells, tNumLocalDofsPerCell);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumLocalDofsPerCell; ++j)
        {
            tHostPrevLocalState(i, j) = (i + 1.0) * (j + 1.0) / 3.0;
            //printf("(%d,%d) = %f\n", i,j,tHostPrevLocalState(i, j));
        }
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVector tResult("backstress residual", tNumCells, tNumLocalDofsPerCell);

    Plato::J2PlasticityUtilities<tSpaceDim> tJ2PlasticityUtils;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tJ2PlasticityUtils.fillBackstressTensorResidualElasticStep(tCellOrdinal, tLocalState, 
                                                                   tPrevLocalState, tResult);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold = {1.5, 1.666666, 1.8333333, 2.0, 2.1666666, 2.333333};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 8; tDofIndex < 8 + tNumVoigtTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostResult(tCellIndex, tDofIndex), tGold[tDofIndex-8], tTolerance);
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
