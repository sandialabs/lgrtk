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
#include "plato/PlatoMathHelpers.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Print input 1D container to terminal
 * @param [in] aInput 1D container
**********************************************************************************/
template<typename VecT>
inline void print(const VecT & aInput)
{
    Plato::OrdinalType tSize = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tSize), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex){
        printf("X[%d] = %f\n", aIndex, aInput(aIndex));
    }, "fill vector");
    printf("\n");
}

/******************************************************************************//**
 * @brief Compute cell/element mass, /f$ \sum_{i=1}^{N} \[M\] \{z\} /f$, where
 * /f$ \[M\] /f$ is the mass matrix, /f$ \{z\} /f$ is the control vector and
 * /f$ N /f$ is the number of nodes.
 * @param [in] aCellOrdinal cell/element index
 * @param [in] aBasisFunc 1D container of cell basis functions
 * @param [in] aCellControls 2D container of cell controls
 * @return cell/element penalized mass
**********************************************************************************/
template<Plato::OrdinalType CellNumNodes, typename ControlType>
DEVICE_TYPE inline ControlType
cell_mass(const Plato::OrdinalType & aCellOrdinal,
          const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunc,
          const Plato::ScalarMultiVectorT<ControlType> & aCellControls)
{
    ControlType tCellMass = 0.0;
    for(Plato::OrdinalType tIndex_I = 0; tIndex_I < CellNumNodes; tIndex_I++)
    {
        ControlType tNodalMass = 0.0;
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
 * @param [in] aNumControls number of controls
 * @param [in] aCellControls 2D container of cell controls
 * @return average density for this cell/element
**********************************************************************************/
template<Plato::OrdinalType NumControls, typename ControlType>
DEVICE_TYPE inline ControlType
cell_density(const Plato::OrdinalType & aCellOrdinal,
             const Plato::ScalarMultiVectorT<ControlType> & aCellControls)
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
    template<typename Inputype, typename ResultType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<Inputype> & aCauchyStress,
               const Plato::ScalarVectorT<ResultType> & aVonMisesStress) const;
};
// class VonMisesYield

/******************************************************************************//**
 * @brief Von Mises yield criterion for 3D problems
**********************************************************************************/
template<>
template<typename Inputype, typename ResultType>
DEVICE_TYPE inline void
VonMisesYield<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<Inputype> & aCauchyStress,
        const Plato::ScalarVectorT<ResultType> & aVonMisesStress) const
{
    Inputype tSigma11MinusSigma22 = aCauchyStress(aCellOrdinal, 0) - aCauchyStress(aCellOrdinal, 1);
    Inputype tSigma22MinusSigma33 = aCauchyStress(aCellOrdinal, 1) - aCauchyStress(aCellOrdinal, 2);
    Inputype tSigma33MinusSigma11 = aCauchyStress(aCellOrdinal, 2) - aCauchyStress(aCellOrdinal, 0);
    Inputype tPrincipalStressContribution = tSigma11MinusSigma22 * tSigma11MinusSigma22 +
            tSigma22MinusSigma33 * tSigma22MinusSigma33 + tSigma33MinusSigma11 * tSigma33MinusSigma11;

    Inputype tShearStressContribution = static_cast<Plato::Scalar>(3) *
            ( aCauchyStress(aCellOrdinal, 3) * aCauchyStress(aCellOrdinal, 3)
            + aCauchyStress(aCellOrdinal, 4) * aCauchyStress(aCellOrdinal, 4)
            + aCauchyStress(aCellOrdinal, 5) * aCauchyStress(aCellOrdinal, 5) );

    ResultType tVonMises = static_cast<Plato::Scalar>(0.5) * ( tPrincipalStressContribution + tShearStressContribution);
    aVonMisesStress(aCellOrdinal) = pow(tVonMises, static_cast<Plato::Scalar>(0.5));
}

/******************************************************************************//**
 * @brief Von Mises yield criterion for 2D problems (i.e. general plane stress)
**********************************************************************************/
template<>
template<typename Inputype, typename ResultType>
DEVICE_TYPE inline void
VonMisesYield<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<Inputype> & aCauchyStress,
        const Plato::ScalarVectorT<ResultType> & aVonMisesStress) const
{
    Inputype tSigma11TimesSigma11 = aCauchyStress(aCellOrdinal, 0) * aCauchyStress(aCellOrdinal, 0);
    Inputype tSigma11TimesSigma22 = aCauchyStress(aCellOrdinal, 0) * aCauchyStress(aCellOrdinal, 1);
    Inputype tSigma22TimesSigma22 = aCauchyStress(aCellOrdinal, 1) * aCauchyStress(aCellOrdinal, 1);
    Inputype tSigma12TimesSigma12 = aCauchyStress(aCellOrdinal, 2) * aCauchyStress(aCellOrdinal, 2);

    ResultType tVonMises = tSigma11TimesSigma11 - tSigma11TimesSigma22 + tSigma22TimesSigma22 +
            static_cast<Plato::Scalar>(3) * tSigma12TimesSigma12;
    aVonMisesStress(aCellOrdinal) = pow(tVonMises, static_cast<Plato::Scalar>(0.5));
}

/******************************************************************************//**
 * @brief Von Mises yield criterion for 1D problems (i.e. uniaxial stress)
**********************************************************************************/
template<>
template<typename Inputype, typename ResultType>
DEVICE_TYPE inline void
VonMisesYield<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<Inputype> & aCauchyStress,
        const Plato::ScalarVectorT<ResultType> & aVonMisesStress) const
{
    aVonMisesStress(aCellOrdinal) = aCauchyStress(aCellOrdinal, 0);
}

template< Plato::OrdinalType SpaceDim, typename StateT, typename ControlT, typename ConfigT, typename ResultT>
inline void evaluate(const Omega_h::Mesh & aMesh,
                     const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                     const Plato::ScalarMultiVectorT<ControlT> & aControlWS,
                     const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                     const Plato::ScalarVectorT<ResultT> & aResultWS)
{
    using StrainT = typename Plato::fad_type_t<Plato::SimplexMechanics<SpaceDim>, StateT, ConfigT>;

    // Create Lagrange multipliers workset
    const Plato::OrdinalType tNumCells = aMesh.nelems();
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
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<SpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<SpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    SIMP tPenaltySIMP;
    Strain<SpaceDim> tCauchyStrain;
    Plato::VonMisesYield<SpaceDim> tVonMises;
    LinearStress<SpaceDim> tCauchyStress(tCellStiffMatrix);
    Plato::ComputeGradientWorkset<SpaceDim> tComputeGradient;

    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<SpaceDim>::m_numNodesPerCell;
    Plato::ScalarVectorT<ResultT> tCellVonMises("von mises", tNumCells);
    Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
    Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, tNumNodesPerCell, SpaceDim);
    Plato::ScalarMultiVectorT<StrainT> tCellCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResultT> tCellCauchyStress("stress", tNumCells, tNumVoigtTerms);

    constexpr Plato::Scalar tStressLimit = 1;
    constexpr Plato::Scalar tAugLagPenaltyParam = 0.1;

    // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
    Plato::LinearTetCubRuleDegreeOne<SpaceDim> tCubatureRule;
    auto tCubWeight = tCubatureRule.getCubWeight();
    auto tBasisFunc = tCubatureRule.getBasisFunctions();

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tCellVolume);
        tCellVolume(tCellOrdinal) *= tCubWeight;
        tCauchyStrain(tCellOrdinal, tCellCauchyStrain, aStateWS, tGradient);
        tCauchyStress(tCellOrdinal, tCellCauchyStress, tCellCauchyStrain);

        // Compute 3D Von Mises Yield Criterion
        tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);

        // Compute Von Mises stress constraint residual
        ResultT tVonMisesOverStressLimit = tCellVonMises(tCellOrdinal) / tStressLimit;
        ResultT tVonMisesOverLimitMinusOne = tVonMisesOverStressLimit - static_cast<Plato::Scalar>(1.0);

        ControlT tCellDensity = Plato::cell_density<tNumNodesPerCell>(tCellOrdinal, aControlWS);
        ControlT tPenalizedCellDensity = tPenaltySIMP(tCellDensity);
        ResultT tSuggestedPenalizedStressConstraint = tPenalizedCellDensity * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne;
        ResultT tPenalizedStressConstraint = tVonMisesOverStressLimit > static_cast<ResultT>(1.0) ?
                tSuggestedPenalizedStressConstraint : static_cast<ResultT>(0.0);

        // Compute relaxed Von Mises stress constraint
        Plato::Scalar tLambdaOverPenalty = static_cast<Plato::Scalar>(-1.0) * tDeviceLambdaView(tCellOrdinal) / tAugLagPenaltyParam;
        ResultT tRelaxedStressConstraint = max(tPenalizedStressConstraint, tLambdaOverPenalty);

        // Compute Von Mises stress contribution to augmented Lagrangian function
        ResultT tStressContribution = ( tDeviceLambdaView(tCellOrdinal) +
                static_cast<Plato::Scalar>(0.5) * tAugLagPenaltyParam * tRelaxedStressConstraint ) * tRelaxedStressConstraint;

        // Compute mass contribution to augmented Lagrangian function
        ResultT tCellMass = Plato::cell_mass<tNumNodesPerCell>(tCellOrdinal, tBasisFunc, aControlWS);
        tCellMass *= tCellVolume(tCellOrdinal);
        ResultT tMassContribution = tDeviceMassMultiplierView(tCellOrdinal) * tCellMass;

        // Compute augmented Lagrangian
        aResultWS(tCellOrdinal) = tMassContribution + tStressContribution;
    },"Compute Augmented Lagrangian Stress Func.");
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
        tCellDensity(tCellOrdinal) = Plato::cell_density<tNumNodesPerCell>(tCellOrdinal, tCellControls);
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

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_CriterionEval_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using StateT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual::StateScalarType;
    using ConfigT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual::ConfigScalarType;
    using ResultT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual::ResultScalarType;
    using ControlT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    std::vector<Plato::Scalar> tHostControl(tNumVerts, 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostControlView(tHostControl.data(),tHostControl.size());
    auto tDeviceControlView = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), tHostControlView);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tDeviceControlView, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    std::vector<Plato::Scalar> tHostStateData(tNumDofs, 0.1);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumDofs; tIndex++)
    { tHostStateData[tIndex] *= tIndex; }
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostStateView(tHostStateData.data(),tHostStateData.size());
    auto tDeviceStateView = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostStateView);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tDeviceStateView, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = { 0.00166667, 0.0121221, 0.00166667, 0.0426097, 0.26095, 0.383705 };
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.702721, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_CriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using StateT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ::StateScalarType;
    using ConfigT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ::ConfigScalarType;
    using ResultT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ::ResultScalarType;
    using ControlT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    std::vector<Plato::Scalar> tHostControl(tNumVerts, 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostControlView(tHostControl.data(),tHostControl.size());
    auto tDeviceControlView = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), tHostControlView);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tDeviceControlView, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    std::vector<Plato::Scalar> tHostStateData(tNumDofs, 0.1);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumDofs; tIndex++)
    { tHostStateData[tIndex] *= tIndex; }
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostStateView(tHostStateData.data(),tHostStateData.size());
    auto tDeviceStateView = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostStateView);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tDeviceStateView, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST ASSEMBLED OBJ. FUNC VALUE AND PARTIAL W.R.T. CONTROLS ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    constexpr Plato::OrdinalType tNumControlFields = 1;
    Plato::ScalarVector tPartialZ("objective partial control", tNumVerts);
    Plato::VectorEntryOrdinal<tSpaceDim, tNumControlFields> tControlEntryOrdinal(&(*tMesh));
    Plato::assemble_scalar_gradient<tNodesPerCell>(tNumCells, tControlEntryOrdinal, tResultWS, tPartialZ);
    Plato::Scalar tObjFuncVal = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.702721, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FiniteDiff_CriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using StateT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ::StateScalarType;
    using ConfigT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ::ConfigScalarType;
    using ResultT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ::ResultScalarType;
    using ControlT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    std::vector<Plato::Scalar> tHostStateData(tNumDofs, 0.1);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumDofs; tIndex++)
    { tHostStateData[tIndex] *= tIndex; }
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostStateView(tHostStateData.data(),tHostStateData.size());
    auto tDeviceStateView = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostStateView);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tDeviceStateView, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // finite difference
    Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
    constexpr Plato::OrdinalType tNumControlFields = 1;
    Plato::ScalarVector tPartialZ("objective partial control", tNumVerts);
    Plato::VectorEntryOrdinal<tSpaceDim, tNumControlFields> tControlEntryOrdinal(&(*tMesh));
    Plato::assemble_scalar_gradient<tNodesPerCell>(tNumCells, tControlEntryOrdinal, tResultWS, tPartialZ);

    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueOne = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(-tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueTwo = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueThree = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueFour = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FiniteDiff_CriterionGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using StateT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian::StateScalarType;
    using ConfigT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian::ConfigScalarType;
    using ResultT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian::ResultScalarType;
    using ControlT = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerNode = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerNode;
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // finite difference
    Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
    Plato::ScalarVector tPartialU("objective partial state", tTotalNumDofs);
    Plato::VectorEntryOrdinal<tSpaceDim, tDofsPerNode> tStateEntryOrdinal(&(*tMesh));
    Plato::assemble_vector_gradient<tNodesPerCell, tDofsPerNode>(tNumCells, tStateEntryOrdinal, tResultWS, tPartialU);

    Plato::ScalarVector tStep("step", tTotalNumDofs);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::dot(tPartialU, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    Plato::ScalarVector tTrialState("trial state", tTotalNumDofs);
    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueOne = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(-tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueTwo = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueThree = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(-2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        Plato::evaluate<tSpaceDim>(*tMesh, tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueFour = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_UpdateMassMultipliers)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    std::vector<Plato::Scalar> tHostControl(tNumVerts, 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        tHostControlView(tHostControl.data(),tHostControl.size());
    auto tDeviceControlView = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), tHostControlView);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
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
    Plato::ScalarArray3DT<Plato::Scalar> tGradient("gradient", tNumCells, tNumNodesPerCell, tSpaceDim);
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
        Plato::Scalar tCellDensity = Plato::cell_density<tNumNodesPerCell>(tCellOrdinal, tControlWS);
        tMassMultiplierMeasures(tCellOrdinal) = tVonMisesOverStressLimit * pow(tCellDensity, static_cast<Plato::Scalar>(0.5));

        const Plato::Scalar tOptionOne = static_cast<Plato::Scalar>(0.7) * tMassMultipliers(tCellOrdinal) - static_cast<Plato::Scalar>(0.1);
        const Plato::Scalar tOptionTwo = static_cast<Plato::Scalar>(2.5) * tMassMultipliers(tCellOrdinal) + static_cast<Plato::Scalar>(0.5);
        tMassMultipliers(tCellOrdinal) = tMassMultiplierMeasures(tCellOrdinal) > static_cast<Plato::Scalar>(1.0) ?
                max(tOptionOne, tMassMultiplierLowerBound) : min(tOptionTwo, tMassMultiplierUpperBound);
    }, "Update Mass Multipliers");
}

} // namespace AugLagStressTest
