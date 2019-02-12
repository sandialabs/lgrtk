/*
 * PlatoAugLagStressTest.cpp
 *
 *  Created on: Feb 3, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "LinearElasticMaterial.hpp"

#include "plato/Simp.hpp"
#include "plato/Strain.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/LinearStress.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute cell/element mass, /f$ \sum_{i=1}^{N} \[M\] \{z\} /f$, where
 * /f$ \[M\] /f$ is the mass matrix, /f$ \{z\} /f$ is the control vector and
 * /f$ N /f$ is the number of nodes.
 * @param [in] aCellOrdinal cell/element index
 * @param [in] aBasisFunc 1D container of cell basis functions
 * @param [in] aCellControls 2D container of cell controls
 * @return cell/element penalized mass
**********************************************************************************/
template<typename ControlType, Plato::OrdinalType CellNumNodes>
DEVICE_TYPE inline ControlType
cell_mass(const Plato::OrdinalType & aCellOrdinal,
          const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunc,
          const Plato::ScalarMultiVectorT<ControlType> & aCellControls)
{
    Plato::Scalar tCellMass = 0.0;
    for(Plato::OrdinalType tIndex_I = 0; tIndex_I < CellNumNodes; tIndex_I++)
    {
        Plato::Scalar tNodalMass = 0.0;
        for(Plato::OrdinalType tIndex_J = 0; tIndex_J < CellNumNodes; tIndex_J++)
        {
            tNodalMass += aBasisFunc(tIndex_I) * aBasisFunc(tIndex_J)
                    * aCellControls(aCellOrdinal, tIndex_J);
        }
        tCellMass += tNodalMass;
    }
    return (tCellMass);
}

/******************************************************************************//**
 * @brief Compute average cell density
 * @param [in] aCellOrdinal cell/element index
 * @param [in] aCellControls 2D container of cell controls
 * @return average density for this cell/element
**********************************************************************************/
template<typename ControlType, Plato::OrdinalType NumControls>
DEVICE_TYPE inline ControlType
cell_density(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<ControlType> & aCellControls)
{
    ControlType tCellDensity = 0.0;
    for(Plato::OrdinalType tIndex = 0; tIndex < NumControls; tIndex++)
    {
        tCellDensity += aCellControls(aCellOrdinal, tIndex);
    }
    tCellDensity /= NumControls;
    return (tCellDensity);
}
// function cell_density

/******************************************************************************//**
 * @brief Compute Von Mises yield criterion
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class VonMisesYield
{
public:
    /******************************************************************************//**
     * @brief Constructor
    **********************************************************************************/
    VonMisesYield(){}

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    ~VonMisesYield(){}

    /******************************************************************************//**
     * @brief Compute Von Mises yield criterion
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aCauchyStress 2D container of cell Cauchy stresses
     * @param [out] aVonMisesStress 1D container of cell Von Mises yield stresses
    **********************************************************************************/
    template<typename InputStressScalarType, typename OutputStressScalarType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<InputStressScalarType> & aCauchyStress,
               const Plato::ScalarVectorT<OutputStressScalarType> & aVonMisesStress) const;
};
// class VonMisesYield

/******************************************************************************//**
 * @brief Von Mises yield criterion for 3D problems
**********************************************************************************/
template<>
template<typename InputStressScalarType, typename OutputStressScalarType>
DEVICE_TYPE inline void
VonMisesYield<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<InputStressScalarType> & aCauchyStress,
        const Plato::ScalarVectorT<OutputStressScalarType> & aVonMisesStress) const
{
    const Plato::Scalar tSigma11MinusSigma22 = aCauchyStress(aCellOrdinal, 0) - aCauchyStress(aCellOrdinal, 1);
    const Plato::Scalar tSigma22MinusSigma33 = aCauchyStress(aCellOrdinal, 1) - aCauchyStress(aCellOrdinal, 2);
    const Plato::Scalar tSigma33MinusSigma11 = aCauchyStress(aCellOrdinal, 2) - aCauchyStress(aCellOrdinal, 0);
    const Plato::Scalar tPrincipalStressContribution = tSigma11MinusSigma22 * tSigma11MinusSigma22 +
            tSigma22MinusSigma33 * tSigma22MinusSigma33 + tSigma33MinusSigma11 * tSigma33MinusSigma11;

    const Plato::Scalar tShearStressContribution = static_cast<Plato::Scalar>(3) *
            ( aCauchyStress(aCellOrdinal, 3) * aCauchyStress(aCellOrdinal, 3)
            + aCauchyStress(aCellOrdinal, 4) * aCauchyStress(aCellOrdinal, 4)
            + aCauchyStress(aCellOrdinal, 5) * aCauchyStress(aCellOrdinal, 5) );

    const Plato::Scalar tVonMises = static_cast<Plato::Scalar>(0.5) * ( tPrincipalStressContribution + tShearStressContribution);
    aVonMisesStress(aCellOrdinal) = pow(tVonMises, static_cast<Plato::Scalar>(0.5));
}

/******************************************************************************//**
 * @brief Von Mises yield criterion for 2D problems (i.e. general plane stress)
**********************************************************************************/
template<>
template<typename InputStressScalarType, typename OutputStressScalarType>
DEVICE_TYPE inline void
VonMisesYield<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<InputStressScalarType> & aCauchyStress,
        const Plato::ScalarVectorT<OutputStressScalarType> & aVonMisesStress) const
{
    const Plato::Scalar tSigma11TimesSigma11 = aCauchyStress(aCellOrdinal, 0) * aCauchyStress(aCellOrdinal, 0);
    const Plato::Scalar tSigma11TimesSigma22 = aCauchyStress(aCellOrdinal, 0) * aCauchyStress(aCellOrdinal, 1);
    const Plato::Scalar tSigma22TimesSigma22 = aCauchyStress(aCellOrdinal, 1) * aCauchyStress(aCellOrdinal, 1);
    const Plato::Scalar tSigma12TimesSigma12 = aCauchyStress(aCellOrdinal, 2) * aCauchyStress(aCellOrdinal, 2);

    const Plato::Scalar tVonMises = tSigma11TimesSigma11 - tSigma11TimesSigma22 + tSigma22TimesSigma22 +
            static_cast<Plato::Scalar>(3) * tSigma12TimesSigma12;
    aVonMisesStress(aCellOrdinal) = pow(tVonMises, static_cast<Plato::Scalar>(0.5));
}

/******************************************************************************//**
 * @brief Von Mises yield criterion for 1D problems (i.e. uniaxial stress)
**********************************************************************************/
template<>
template<typename InputStressScalarType, typename OutputStressScalarType>
DEVICE_TYPE inline void
VonMisesYield<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<InputStressScalarType> & aCauchyStress,
        const Plato::ScalarVectorT<OutputStressScalarType> & aVonMisesStress) const
{
    aVonMisesStress(aCellOrdinal) = aCauchyStress(aCellOrdinal, 0);
}

} // namespace Plato

namespace AugLagStressTest
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_CellDensity)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::ScalarMultiVector tCellControls("Control Workset", tNumCells, tNumNodesPerCell);
    auto tHostCellControls = Kokkos::create_mirror(tCellControls);
    tHostCellControls(0,0) = 1.00; tHostCellControls(1,0) = 0.93;
    tHostCellControls(0,1) = 0.90; tHostCellControls(1,1) = 1.00;
    tHostCellControls(0,2) = 0.95; tHostCellControls(1,2) = 0.89;
    tHostCellControls(0,3) = 0.89; tHostCellControls(1,3) = 0.91;
    Kokkos::deep_copy(tCellControls, tHostCellControls);

    Plato::ScalarVector tCellDensity("Cell Density", tNumCells);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        tCellDensity(tCellOrdinal) = Plato::cell_density<Plato::Scalar, tNumNodesPerCell>(tCellOrdinal, tCellControls);
    }, "Test cell density inline function");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.935, 0.9325};
    auto tHostCellDensity = Kokkos::create_mirror(tCellDensity);
    Kokkos::deep_copy(tHostCellDensity, tCellDensity);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellDensity(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0,0) = 1.096154; tHostCauchyStress(1,0) = 1.557692;
    tHostCauchyStress(0,1) = 1.557692; tHostCauchyStress(1,1) = 1.557692;
    tHostCauchyStress(0,2) = 1.096154; tHostCauchyStress(1,2) = 0.634615;
    tHostCauchyStress(0,3) = 0.461538; tHostCauchyStress(1,3) = 0.230769;
    tHostCauchyStress(0,4) = 0.230769; tHostCauchyStress(1,4) = 0.230769;
    tHostCauchyStress(0,5) = 0.461538; tHostCauchyStress(1,5) = 0.692308;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
    }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.965377, 1.315587};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0,0) = 1.096154; tHostCauchyStress(1,0) = 1.457692;
    tHostCauchyStress(0,1) = 1.557692; tHostCauchyStress(1,1) = 1.557692;
    tHostCauchyStress(0,2) = 1.096154; tHostCauchyStress(1,2) = 0.634615;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
    }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {2.350563425, 1.867844683};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises1D)
{
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0,0) = 1.096154; tHostCauchyStress(1,0) = 1.457692;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
    }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {1.096154, 1.457692};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_CriterionEval3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    std::vector<Plato::Scalar> tHostControl(tNumVerts, 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostControlView(tHostControl.data(),tHostControl.size());
    auto tDeviceControlView = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), tHostControlView);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tDeviceControlView, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    std::vector<Plato::Scalar> tHostStateData(tNumDofs, 0.1);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumDofs; tIndex++)
    { tHostStateData[tIndex] *= tIndex; }
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostStateView(tHostStateData.data(),tHostStateData.size());
    auto tDeviceStateView = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostStateView);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tDeviceStateView, tStateWS);

    // Create Lagrange multipliers workset
    std::vector<Plato::Scalar> tHostLambdaData(tNumCells, 0.1);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostLambdaView(tHostLambdaData.data(),tHostLambdaData.size());
    auto tDeviceLambdaView = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostLambdaView);

    // Create stress weight workset
    std::vector<Plato::Scalar> tHostMassMultiplierData(tNumCells, 0.01);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostMassMultiplierView(tHostMassMultiplierData.data(),tHostMassMultiplierData.size());
    auto tDeviceMassMultiplierView = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostMassMultiplierView);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    SIMP tPenaltySIMP;
    Strain<tSpaceDim> tCauchyStrain;
    Plato::VonMisesYield<tSpaceDim> tVonMises;
    LinearStress<tSpaceDim> tCauchyStress(tCellStiffMatrix);
    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;

    Plato::ScalarVectorT<Plato::Scalar> tResult("result", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tCellVonMises("von mises", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tCellVolume("cell volume", tNumCells);
    Plato::ScalarArray3DT<Plato::Scalar> tGradient("gradient", tNumCells, tNodesPerCell, tSpaceDim);
    Plato::ScalarMultiVectorT<Plato::Scalar> tCellCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tCellCauchyStress("stress", tNumCells, tNumVoigtTerms);

    constexpr Plato::Scalar tStressLimit = 1;
    constexpr Plato::Scalar tAugLagPenaltyParam = 0.1;

    // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    auto tCubWeight = tCubatureRule.getCubWeight();
    auto tBasisFunc = tCubatureRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(tCellOrdinal) *= tCubWeight;
        tCauchyStrain(tCellOrdinal, tCellCauchyStrain, tStateWS, tGradient);
        tCauchyStress(tCellOrdinal, tCellCauchyStress, tCellCauchyStrain);
        printf("Cell %d : Cauchy Stress 11 = %f\n", tCellOrdinal, tCellCauchyStress(tCellOrdinal, 0));
        printf("Cell %d : Cauchy Stress 22 = %f\n", tCellOrdinal, tCellCauchyStress(tCellOrdinal, 1));
        printf("Cell %d : Cauchy Stress 33 = %f\n", tCellOrdinal, tCellCauchyStress(tCellOrdinal, 2));
        printf("Cell %d : Cauchy Stress 23 = %f\n", tCellOrdinal, tCellCauchyStress(tCellOrdinal, 3));
        printf("Cell %d : Cauchy Stress 13 = %f\n", tCellOrdinal, tCellCauchyStress(tCellOrdinal, 4));
        printf("Cell %d : Cauchy Stress 12 = %f\n", tCellOrdinal, tCellCauchyStress(tCellOrdinal, 5));

        // Compute 3D Von Mises Yield Criterion
        tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
        printf("Cell %d : Von Mises stress = %f\n", tCellOrdinal, tCellVonMises(tCellOrdinal));

        // Compute Von Mises stress constraint residual
        const Plato::Scalar tVonMisesOverStressLimit = tCellVonMises(tCellOrdinal) / tStressLimit;
        const Plato::Scalar tVonMisesOverLimitMinusOne = tVonMisesOverStressLimit - static_cast<Plato::Scalar>(1.0);
        printf("Cell %d : Von Mises stress - limit = %f\n", tCellOrdinal, tVonMisesOverLimitMinusOne);

        Plato::Scalar tCellDensity = Plato::cell_density<Plato::Scalar, tNodesPerCell>(tCellOrdinal, tControlWS);
        const Plato::Scalar tPenalizedCellDensity = tPenaltySIMP(tCellDensity);
        printf("Cell %d : Penalized density = %f\n", tCellOrdinal, tPenalizedCellDensity);
        Plato::Scalar tPenalizedStressConstraint = tPenalizedCellDensity * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne;
        tPenalizedStressConstraint = tVonMisesOverStressLimit > static_cast<Plato::Scalar>(1.0) ?
                tPenalizedStressConstraint : static_cast<Plato::Scalar>(0.0);
        printf("Cell %d : Penalized stress contribution = %f\n", tCellOrdinal, tPenalizedStressConstraint);

        // Compute relaxed Von Mises stress constraint
        const Plato::Scalar tLambdaOverPenalty = static_cast<Plato::Scalar>(-1.0) * tDeviceLambdaView(tCellOrdinal) / tAugLagPenaltyParam;
        const Plato::Scalar tRelaxedStressConstraint = Omega_h::max2(tPenalizedStressConstraint, tLambdaOverPenalty);
        printf("Cell %d : Relaxed stress contribution = %f\n", tCellOrdinal, tRelaxedStressConstraint);

        // Compute Von Mises stress contribution to augmented Lagrangian function
        const Plato::Scalar tStressContribution = ( tDeviceLambdaView(tCellOrdinal) +
                static_cast<Plato::Scalar>(0.5) * tAugLagPenaltyParam * tRelaxedStressConstraint ) * tRelaxedStressConstraint;
        printf("Cell %d : Stress contribution = %f\n", tCellOrdinal, tStressContribution);

        // Compute mass contribution to augmented Lagrangian function
        Plato::Scalar tCellMass = Plato::cell_mass<Plato::Scalar, tNodesPerCell>(tCellOrdinal, tBasisFunc, tControlWS);
        tCellMass *= tCellVolume(tCellOrdinal);
        const Plato::Scalar tMassContribution = tDeviceMassMultiplierView(tCellOrdinal) * tCellMass;
        printf("Cell %d : Mass contribution = %f\n", tCellOrdinal, tMassContribution);

        // Compute augmented Lagrangian
        tResult(tCellOrdinal) = tMassContribution + tStressContribution;
        printf("Cell %d : Augmented Lag = %f\n", tCellOrdinal, tResult(tCellOrdinal));
    },"Compute Augmented Lagrangian Stress Func.");
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_UpdateMassMultipliers)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    std::vector<Plato::Scalar> tHostControl(tNumVerts, 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostControlView(tHostControl.data(),tHostControl.size());
    auto tDeviceControlView = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), tHostControlView);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tDeviceControlView, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    std::vector<Plato::Scalar> tHostStateData(tNumDofs, 0.1);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumDofs; tIndex++)
    { tHostStateData[tIndex] *= tIndex; }
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostStateView(tHostStateData.data(),tHostStateData.size());
    auto tDeviceStateView = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostStateView);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tDeviceStateView, tStateWS);

    // Create Lagrange multipliers workset
    std::vector<Plato::Scalar> tHostMassMultiplierMeasuresData(tNumCells, 0.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostMassMultiplierMeasuresView(tHostMassMultiplierMeasuresData.data(),tHostMassMultiplierMeasuresData.size());
    auto tMassMultiplierMeasures = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostMassMultiplierMeasuresView);

    // Create stress weight workset
    std::vector<Plato::Scalar> tHostMassMultiplierData(tNumCells, 0.01);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostMassMultiplierView(tHostMassMultiplierData.data(),tHostMassMultiplierData.size());
    auto tMassMultipliers = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostMassMultiplierView);

    // Create material
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    // Create Cauchy stress functors
    SIMP tPenaltySIMP;
    Strain<tSpaceDim> tCauchyStrain;
    Plato::VonMisesYield<tSpaceDim> tVonMises;
    LinearStress<tSpaceDim> tCauchyStress(tCellStiffMatrix);
    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;

    // Create test views
    Plato::ScalarVectorT<Plato::Scalar> tCellVonMises("von mises", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tCellVolume("cell volume", tNumCells);
    Plato::ScalarArray3DT<Plato::Scalar> tGradient("gradient", tNumCells, tNodesPerCell, tSpaceDim);
    Plato::ScalarMultiVectorT<Plato::Scalar> tCellCauchyStress("stress", tNumCells , tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tCellCauchyStrain("strain", tNumCells, tNumVoigtTerms);

    // Create problem constants
    constexpr Plato::Scalar tStressLimit = 1;
    constexpr Plato::Scalar tMassMultiplierLowerBound = 0.0;
    constexpr Plato::Scalar tMassMultiplierUpperBound = 4.0;

    // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    auto tCubWeight = tCubatureRule.getCubWeight();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        // Compute 3D Cauchy Stress
        tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(tCellOrdinal) *= tCubWeight;
        tCauchyStrain(tCellOrdinal, tCellCauchyStrain, tStateWS, tGradient);
        tCauchyStress(tCellOrdinal, tCellCauchyStress, tCellCauchyStrain);

        // Compute 3D Von Mises Yield Criterion
        tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
        const Plato::Scalar tVonMisesOverStressLimit = tCellVonMises(tCellOrdinal) / tStressLimit;

        // Compute mass multiplier measure
        Plato::Scalar tCellDensity = Plato::cell_density<Plato::Scalar, tNodesPerCell>(tCellOrdinal, tControlWS);
        tMassMultiplierMeasures(tCellOrdinal) = tVonMisesOverStressLimit * pow(tCellDensity, static_cast<Plato::Scalar>(0.5));

        const Plato::Scalar tOptionOne = static_cast<Plato::Scalar>(0.7) * tMassMultipliers(tCellOrdinal) - static_cast<Plato::Scalar>(0.1);
        const Plato::Scalar tOptionTwo = static_cast<Plato::Scalar>(2.5) * tMassMultipliers(tCellOrdinal) + static_cast<Plato::Scalar>(0.5);
        tMassMultipliers(tCellOrdinal) = tMassMultiplierMeasures(tCellOrdinal) > static_cast<Plato::Scalar>(1.0) ?
                Omega_h::max2(tOptionOne, tMassMultiplierLowerBound) : min(tOptionTwo, tMassMultiplierUpperBound);
    }, "Update Mass Multipliers");
}

} // namespace AugLagStressTest
