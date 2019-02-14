/*
 * Plato_AugLagStressCriterion.hpp
 *
 *  Created on: Feb 12, 2019
 */

#pragma once

#include <algorithm>

#include "ImplicitFunctors.hpp"
#include "LinearElasticMaterial.hpp"

#include "plato/Simp.hpp"
#include "plato/Strain.hpp"
#include "plato/LinearStress.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/Plato_VonMisesYield.hpp"
#include "plato/Plato_TopOptFunctors.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Augmented Lagrangian stress constraint criterion
**********************************************************************************/
template<typename EvaluationType>
class AugLagStressCriterion :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
        public AbstractScalarFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell/element */

    using AbstractScalarFunction<EvaluationType>::mMesh; /*!< mesh database */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */

    Plato::Scalar mStressLimit; /*!< stress limit/upper bound */
    Plato::Scalar mAugLagPenalty; /*!< augmented Lagrangian penalty */
    Plato::Scalar mAugLagPenaltyUpperBound; /*!< upper bound on augmented Lagrangian penalty */
    Plato::Scalar mMassMultipliersLowerBound; /*!< lower bound on mass multipliers */
    Plato::Scalar mMassMultipliersUpperBound; /*!< upper bound on mass multipliers */
    Plato::Scalar mMassNormalizationMultiplier; /*!< normalization multipliers for mass criterion */
    Plato::Scalar mInitialMassMultipliersValue; /*!< initial value for mass multipliers */
    Plato::Scalar mInitialLagrangeMultipliersValue; /*!< initial value for Lagrange multipliers */
    Plato::Scalar mAugLagPenaltyExpansionMultiplier; /*!< expansion parameter for augmented Lagrangian penalty */
    Plato::Scalar mMassMultiplierUpperBoundReductionParam; /*!< reduction parameter for upper bound on mass multipliers */

    Plato::ScalarVector mMassMultipliers; /*!< mass multipliers */
    Plato::ScalarVector mLagrangeMultipliers; /*!< Lagrange multipliers */
    Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

private:
    /******************************************************************************//**
     * @brief Allocate member data
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aInputParams);
        auto tMaterialModel = tMaterialModelFactory.create();
        mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();

        this->readInputs(aInputParams);

        Plato::fill(mInitialMassMultipliersValue, mMassMultipliers);
        Plato::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Read user inputs
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.get<Teuchos::ParameterList>("Stress Constraint");
        mStressLimit = tParams.get<Plato::Scalar>("Stress Limit", 1.0);
        mAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 0.1);
        mAugLagPenaltyUpperBound = tParams.get<Plato::Scalar>("Penalty Upper Bound", 100.0);
        mMassMultipliersLowerBound = tParams.get<Plato::Scalar>("Mass Multiplier Lower Bound", 0.0);
        mMassMultipliersUpperBound = tParams.get<Plato::Scalar>("Mass Multiplier Upper Bound", 4.0);
        mInitialMassMultipliersValue = tParams.get<Plato::Scalar>("Initial Mass Multiplier", 0.0);
        mMassNormalizationMultiplier = tParams.get<Plato::Scalar>("Mass Normalization Multiplier", 1.0);
        mInitialLagrangeMultipliersValue = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.01);
        mAugLagPenaltyExpansionMultiplier = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.05);
        mMassMultiplierUpperBoundReductionParam = tParams.get<Plato::Scalar>("Mass Multiplier Reduction Multiplier", 0.95);
    }

    /******************************************************************************//**
     * @brief Update augmented Lagrangian penalty and upper bound on mass multipliers.
    **********************************************************************************/
    void updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
        mMassMultipliersUpperBound = mMassMultipliersUpperBound * mMassMultiplierUpperBoundReductionParam;
    }

public:
    /******************************************************************************//**
     * @brief Primary constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     **********************************************************************************/
    AugLagStressCriterion(Omega_h::Mesh & aMesh,
                          Omega_h::MeshSets & aMeshSets,
                          Plato::DataMap & aDataMap,
                          Teuchos::ParameterList & aInputParams) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Von Mises Criterion"),
            mStressLimit(1),
            mAugLagPenalty(0.1),
            mAugLagPenaltyUpperBound(100),
            mMassMultipliersLowerBound(0),
            mMassMultipliersUpperBound(4),
            mMassNormalizationMultiplier(1.0),
            mInitialMassMultipliersValue(0.0),
            mInitialLagrangeMultipliersValue(0.01),
            mAugLagPenaltyExpansionMultiplier(1.05),
            mMassMultiplierUpperBoundReductionParam(0.95),
            mMassMultipliers("Mass Multipliers", aMesh.nelems()),
            mLagrangeMultipliers("Lagrange Multipliers", aMesh.nelems())
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * @brief Constructor tailored for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    AugLagStressCriterion(Omega_h::Mesh & aMesh, Omega_h::MeshSets & aMeshSets, Plato::DataMap & aDataMap) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Von Mises Criterion"),
            mStressLimit(1),
            mAugLagPenalty(0.1),
            mAugLagPenaltyUpperBound(100),
            mMassMultipliersLowerBound(0),
            mMassMultipliersUpperBound(4),
            mMassNormalizationMultiplier(1.0),
            mInitialMassMultipliersValue(0.0),
            mInitialLagrangeMultipliersValue(0.01),
            mAugLagPenaltyExpansionMultiplier(1.05),
            mMassMultiplierUpperBoundReductionParam(0.95),
            mMassMultipliers("Mass Multipliers", aMesh.nelems()),
            mLagrangeMultipliers("Lagrange Multipliers", aMesh.nelems())
    {
        Plato::fill(mInitialMassMultipliersValue, mMassMultipliers);
        Plato::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~AugLagStressCriterion()
    {
    }

    /******************************************************************************//**
     * @brief Return augmented Lagrangian penalty multiplier
     * @return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    Plato::Scalar getAugLagPenalty() const
    {
        return (mAugLagPenalty);
    }

    /******************************************************************************//**
     * @brief Return upper bound on mass multipliers
     * @return upper bound on mass multipliers
    **********************************************************************************/
    Plato::Scalar getMassMultipliersUpperBound() const
    {
        return (mMassMultipliersUpperBound);
    }

    /******************************************************************************//**
     * @brief Return mass multipliers
     * @return 1D view of mass multipliers
    **********************************************************************************/
    Plato::ScalarVector getMassMultipliers() const
    {
        return (mMassMultipliers);
    }

    /******************************************************************************//**
     * @brief Return Lagrange multipliers
     * @return 1D view of Lagrange multipliers
    **********************************************************************************/
    Plato::ScalarVector getLagrangeMultipliers() const
    {
        return (mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Set stress constraint limit/upper bound
     * @param [in] aInput stress constraint limit
    **********************************************************************************/
    void setStressLimit(const Plato::Scalar & aInput)
    {
        mStressLimit = aInput;
    }

    /******************************************************************************//**
     * @brief Set augmented Lagrangian function penalty multiplier
     * @param [in] aInput penalty multiplier
     **********************************************************************************/
    void setAugLagPenalty(const Plato::Scalar & aInput)
    {
        mAugLagPenalty = aInput;
    }

    /******************************************************************************//**
     * @brief Set mass multipliers
     * @param [in] aInput mass multipliers
     **********************************************************************************/
    void setMassMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mMassMultipliers.size());
        Plato::copy(aInput, mMassMultipliers);
    }

    /******************************************************************************//**
     * @brief Set Lagrange multipliers
     * @param [in] aInput Lagrange multipliers
     **********************************************************************************/
    void setLagrangeMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mLagrangeMultipliers.size());
        Plato::copy(aInput, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Set cell material stiffness matrix
     * @param [in] aInput cell material stiffness matrix
    **********************************************************************************/
    void setCellStiffMatrix(const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aInput)
    {
        mCellStiffMatrix = aInput;
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aStateWS,
                       const Plato::ScalarMultiVector & aControlWS,
                       const Plato::ScalarArray3D & aConfigWS) override
    {
        this->updateMultipliers(aStateWS, aControlWS, aConfigWS);
        this->updateAugLagPenaltyMultipliers();
    }

    /******************************************************************************//**
     * @brief Evaluate augmented Lagrangian stress constraint criterion
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                  const Plato::ScalarMultiVectorT<ControlT> & aControlWS,
                  const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                  Plato::ScalarVectorT<ResultT> & aResultWS,
                  Plato::Scalar aTimeStep = 0.0) const
    {
        using StrainT = typename Plato::fad_type_t<Plato::SimplexMechanics<mSpaceDim>, StateT, ConfigT>;

        SIMP tPenaltySIMP;
        Strain<mSpaceDim> tCauchyStrain;
        Plato::VonMisesYield<mSpaceDim> tVonMises;
        LinearStress<mSpaceDim> tCauchyStress(mCellStiffMatrix);
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;

        const Plato::OrdinalType tNumCells = mMesh.nelems();
        Plato::ScalarVectorT<ResultT> tCellVonMises("von mises", tNumCells);
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::ScalarMultiVectorT<StrainT> tCellCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tCellCauchyStress("stress", tNumCells, mNumVoigtTerms);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tStressLimit = mStressLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tMassMultipliers = mMassMultipliers;
        auto tLagrangeMultipliers = mLagrangeMultipliers;
        auto tMassNormalizationMultiplier = mMassNormalizationMultiplier;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
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

            ControlT tCellDensity = Plato::cell_density<mNumNodesPerCell>(tCellOrdinal, aControlWS);
            ControlT tPenalizedCellDensity = tPenaltySIMP(tCellDensity);
            ResultT tSuggestedPenalizedStressConstraint = tPenalizedCellDensity * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne;
            ResultT tPenalizedStressConstraint = tVonMisesOverStressLimit > static_cast<ResultT>(1.0) ?
                    tSuggestedPenalizedStressConstraint : static_cast<ResultT>(0.0);

            // Compute relaxed Von Mises stress constraint
            Plato::Scalar tLambdaOverPenalty =
                    static_cast<Plato::Scalar>(-1.0) * tLagrangeMultipliers(tCellOrdinal) / tAugLagPenalty;
            ResultT tRelaxedStressConstraint = max(tPenalizedStressConstraint, tLambdaOverPenalty);

            // Compute Von Mises stress contribution to augmented Lagrangian function
            ResultT tStressContribution = ( tLagrangeMultipliers(tCellOrdinal) +
                    static_cast<Plato::Scalar>(0.5) * tAugLagPenalty * tRelaxedStressConstraint ) * tRelaxedStressConstraint;

            // Compute mass contribution to augmented Lagrangian function
            ResultT tCellMass = Plato::cell_mass<mNumNodesPerCell>(tCellOrdinal, tBasisFunc, aControlWS);
            tCellMass *= tCellVolume(tCellOrdinal);
            ResultT tMassContribution = (tMassMultipliers(tCellOrdinal) * tCellMass) / tMassNormalizationMultiplier;

            // Compute augmented Lagrangian
            aResultWS(tCellOrdinal) = tMassContribution + tStressContribution;
        },"Compute Augmented Lagrangian Stress Func.");
    }

    /******************************************************************************//**
     * @brief Update Lagrange and mass multipliers
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateMultipliers(const Plato::ScalarMultiVector & aStateWS,
                           const Plato::ScalarMultiVector & aControlWS,
                           const Plato::ScalarArray3D & aConfigWS)
    {
        // Create Cauchy stress functors
        SIMP tPenaltySIMP;
        Strain<mSpaceDim> tCauchyStrain;
        Plato::VonMisesYield<mSpaceDim> tVonMises;
        LinearStress<mSpaceDim> tCauchyStress(mCellStiffMatrix);
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;

        // Create test views
        const Plato::OrdinalType tNumCells = mMesh.nelems();
        Plato::ScalarVector tCellVonMises("von mises", tNumCells);
        Plato::ScalarVector tCellVolume("cell volume", tNumCells);
        Plato::ScalarVector tMassMultiplierMeasures("mass multipliers measures", tNumCells);
        Plato::ScalarArray3D tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::ScalarMultiVector tCellCauchyStress("stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVector tCellCauchyStrain("strain", tNumCells, mNumVoigtTerms);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tStressLimit = mStressLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tMassMultipliers = mMassMultipliers;
        auto tLagrangeMultipliers = mLagrangeMultipliers;
        auto tMassMultiplierLowerBound = mMassMultipliersLowerBound;
        auto tMassMultiplierUpperBound = mMassMultipliersUpperBound;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tCubWeight = tCubatureRule.getCubWeight();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
        {
            // Compute 3D Cauchy Stress
            tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tCellVolume);
            tCellVolume(tCellOrdinal) *= tCubWeight;
            tCauchyStrain(tCellOrdinal, tCellCauchyStrain, aStateWS, tGradient);
            tCauchyStress(tCellOrdinal, tCellCauchyStress, tCellCauchyStrain);

            // Compute 3D Von Mises Yield Criterion
            tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
            const Plato::Scalar tVonMisesOverStressLimit = tCellVonMises(tCellOrdinal) / tStressLimit;

            // Compute mass multiplier measure
            Plato::Scalar tCellDensity = Plato::cell_density<mNumNodesPerCell>(tCellOrdinal, aControlWS);
            tMassMultiplierMeasures(tCellOrdinal) = tVonMisesOverStressLimit * pow(tCellDensity, static_cast<Plato::Scalar>(0.5));

            // Update mass multipliers
            const Plato::Scalar tOptionOne =
                    static_cast<Plato::Scalar>(0.7) * tMassMultipliers(tCellOrdinal) - static_cast<Plato::Scalar>(0.1);
            const Plato::Scalar tOptionTwo =
                    static_cast<Plato::Scalar>(2.5) * tMassMultipliers(tCellOrdinal) + static_cast<Plato::Scalar>(0.5);
            tMassMultipliers(tCellOrdinal) = tMassMultiplierMeasures(tCellOrdinal) > static_cast<Plato::Scalar>(1.0) ?
                    max(tOptionOne, tMassMultiplierLowerBound) : min(tOptionTwo, tMassMultiplierUpperBound);

            // Compute Von Mises stress constraint residual
            const Plato::Scalar tVonMisesOverLimitMinusOne = tVonMisesOverStressLimit - static_cast<Plato::Scalar>(1.0);
            const Plato::Scalar tPenalizedCellDensity = tPenaltySIMP(tCellDensity);
            const Plato::Scalar tSuggestedPenalizedStressConstraint =
                    tPenalizedCellDensity * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne;
            const Plato::Scalar tPenalizedStressConstraint = tVonMisesOverStressLimit > static_cast<Plato::Scalar>(1.0) ?
                    tSuggestedPenalizedStressConstraint : static_cast<Plato::Scalar>(0.0);

            // Compute relaxed stress constraint
            const Plato::Scalar tLambdaOverPenalty =
                    static_cast<Plato::Scalar>(-1.0) * tLagrangeMultipliers(tCellOrdinal) / tAugLagPenalty;
            const Plato::Scalar tRelaxedStressConstraint = max(tPenalizedStressConstraint, tLambdaOverPenalty);

            // Update Lagrange multipliers
            const Plato::Scalar tSuggestedLagrangeMultiplier =
                    tLagrangeMultipliers(tCellOrdinal) + tAugLagPenalty * tRelaxedStressConstraint;
            tLagrangeMultipliers(tCellOrdinal) = max(tSuggestedLagrangeMultiplier, 0.0);
        }, "Update Multipliers");
    }
};
// class AugLagStressCriterion

}// namespace Plato
