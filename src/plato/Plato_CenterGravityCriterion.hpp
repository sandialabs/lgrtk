/*
 * Plato_CenterGravityCriterion.hpp
 *
 *  Created on: Apr 17, 2019
 */

#pragma once

#include "plato/Simp.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/Plato_StructuralMass.hpp"
#include "plato/Plato_TopOptFunctors.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute centroid coordinates per cell/element
 * @param [in] aCellOrdinal cell/element ordinal
 * @param [in] aConfigWS 3D container with the coordinates per cell/element
 * @param [out] aCentroid 1D container with the centroid coordinates per cell/element
**********************************************************************************/
template<Plato::OrdinalType SpaceDim, typename InputT, typename OutputT>
DEVICE_TYPE inline void compute_simplex_centroid(const Plato::OrdinalType & aCellOrdinal,
                                                 Plato::ScalarArray3DT<InputT> aConfigWS,
                                                 Plato::ScalarMultiVectorT<OutputT> aCentroid)
{
    constexpr Plato::OrdinalType tNumNodesPerCell = SpaceDim + static_cast<Plato::OrdinalType>(1);
    for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aCentroid(aCellOrdinal, tDim) += (aConfigWS(aCellOrdinal, tNodeIndex, tDim) / tNumNodesPerCell);
        }
    }
}
// function compute_simplex_centroid

/******************************************************************************//**
 * @brief Augmented Lagrangian center of gravity constraint criterion with mass objective
**********************************************************************************/
template<typename EvaluationType>
class CenterGravityCriterion :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
        public AbstractScalarFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell/element */

    using AbstractScalarFunction<EvaluationType>::mMesh; /*!< mesh database */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */

    Plato::Scalar mPenalty; /*!< penalty parameter in SIMP model */
    Plato::Scalar mAugLagPenalty; /*!< augmented Lagrangian penalty */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */
    Plato::Scalar mCellMaterialDensity; /*!< material density (note: constant for all the elements/cells) */
    Plato::Scalar mAugLagPenaltyUpperBound; /*!< upper bound on augmented Lagrangian penalty */
    Plato::Scalar mMassNormalizationMultiplier; /*!< normalization multipliers for mass criterion */
    Plato::Scalar mInitialLagrangeMultipliersValue; /*!< initial value for Lagrange multipliers */
    Plato::Scalar mAugLagPenaltyExpansionMultiplier; /*!< expansion parameter for augmented Lagrangian penalty */

    Plato::ScalarVector mLagrangeMultipliers; /*!< Lagrange multipliers */
    Plato::ScalarVector mTargetCenterGravity; /*!< target center of gravity */

private:
    /******************************************************************************//**
     * @brief Allocate member data
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        auto tMaterialModelInputs = aInputParams.get<Teuchos::ParameterList>("Material Model");
        mCellMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);

        this->readInputs(aInputParams);

        Plato::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Read user inputs
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.get<Teuchos::ParameterList>("Center Gravity Constraint");
        mPenalty = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
        mAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 0.25);
        mMinErsatzValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
        mAugLagPenaltyUpperBound = tParams.get<Plato::Scalar>("Penalty Upper Bound", 500.0);
        mMassNormalizationMultiplier = tParams.get<Plato::Scalar>("Mass Normalization Multiplier", 1.0);
        mInitialLagrangeMultipliersValue = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.01);
        mAugLagPenaltyExpansionMultiplier = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.5);
    }

    /******************************************************************************//**
     * @brief Update augmented Lagrangian penalty and upper bound on mass multipliers.
    **********************************************************************************/
    void updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
    }

public:
    /******************************************************************************//**
     * @brief Primary constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     **********************************************************************************/
    CenterGravityCriterion(Omega_h::Mesh & aMesh,
                           Omega_h::MeshSets & aMeshSets,
                           Plato::DataMap & aDataMap,
                           Teuchos::ParameterList & aInputParams) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Center Gravity Constraint"),
            mPenalty(3),
            mAugLagPenalty(0.25),
            mMinErsatzValue(1e-9),
            mCellMaterialDensity(1.0),
            mAugLagPenaltyUpperBound(500),
            mMassNormalizationMultiplier(1.0),
            mInitialLagrangeMultipliersValue(0.01),
            mAugLagPenaltyExpansionMultiplier(1.5),
            mLagrangeMultipliers("Lagrange Multipliers", mSpaceDim),
            mTargetCenterGravity("Target Center of Gravity", mSpaceDim)
    {
        this->initialize(aInputParams);
        this->computeInitialStructuralMass();
    }

    /******************************************************************************//**
     * @brief Constructor tailored for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    CenterGravityCriterion(Omega_h::Mesh & aMesh, Omega_h::MeshSets & aMeshSets, Plato::DataMap & aDataMap) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Center Gravity Constraint"),
            mPenalty(3),
            mAugLagPenalty(0.25),
            mMinErsatzValue(1e-9),
            mCellMaterialDensity(1.0),
            mAugLagPenaltyUpperBound(500),
            mMassNormalizationMultiplier(1.0),
            mInitialLagrangeMultipliersValue(0.01),
            mAugLagPenaltyExpansionMultiplier(1.5),
            mLagrangeMultipliers("Lagrange Multipliers", mSpaceDim),
            mTargetCenterGravity("Target Center of Gravity", mSpaceDim)
    {
        this->computeInitialStructuralMass();
        Plato::fill(1, mTargetCenterGravity);
        Plato::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~CenterGravityCriterion()
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
     * @brief Return multiplier used to normalized mass contribution to the objective function
     * @return upper mass normalization multiplier
    **********************************************************************************/
    Plato::Scalar getMassNormalizationMultiplier() const
    {
        return (mMassNormalizationMultiplier);
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
     * @param [in] aInput 1D view with target center of gravity
    **********************************************************************************/
    void setTargetCenterGravity(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mTargetCenterGravity.size());
        Plato::copy(aInput, mTargetCenterGravity);
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
     * @brief Set cell material density
     * @param [in] aInput material density
     **********************************************************************************/
    void setCellMaterialDensity(const Plato::Scalar & aInput)
    {
        mCellMaterialDensity = aInput;
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
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aStateWS,
                       const Plato::ScalarMultiVector & aControlWS,
                       const Plato::ScalarArray3D & aConfigWS) override
    {
        this->updateLagrangeMultipliers(aStateWS, aControlWS, aConfigWS);
        this->updateAugLagPenaltyMultipliers();
    }

    /******************************************************************************//**
     * @brief Update Lagrange multipliers
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateLagrangeMultipliers(const Plato::ScalarMultiVector & aStateWS,
                                   const Plato::ScalarMultiVector & aControlWS,
                                   const Plato::ScalarArray3D & aConfigWS)
    {
        // ****** INITIALIZE TEMPORARY FUNCTORS ******
        ::SIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tTargetCenterGravity = mTargetCenterGravity;
        auto tMaterialDensity = mCellMaterialDensity;

        // ****** DEFINE CUBATURE RULE ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tCubWeight = tCubatureRule.getCubWeight();
        auto tBasisFunc = tCubatureRule.getBasisFunctions();

        // ****** COMPUTE CONSTRAINT CONTRIBUTION FROM EACH CELL ******
        const Plato::OrdinalType tNumCells = mMesh.nelems();
        Plato::ScalarMultiVector tCentroid("Centroid", tNumCells, mSpaceDim);
        Plato::ScalarMultiVector tConstraints("Constraints", tNumCells, mSpaceDim);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute mass objective contribution to augmented Lagrangian function
            Plato::Scalar tCellVolume = 0.0;
            tComputeCellVolume(aCellOrdinal, aConfigWS, tCellVolume);
            auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControlWS);
            auto tPenalizedMass = tSIMP(tCellMass);
            auto tCellPenalizedMassTimesDensity = tPenalizedMass * tCellVolume * tCubWeight * tMaterialDensity;

            // Compute constraint contribution to augmented Lagrangian function
            Plato::compute_simplex_centroid<mSpaceDim>(aCellOrdinal, aConfigWS, tCentroid);
            auto tAppxTotalMass = static_cast<Plato::Scalar>(tNumCells) * tCellMass * tCellVolume * tMaterialDensity;
            for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mSpaceDim; tDimIndex++)
            {
                auto tMyCurrentCenterGravityContribution = ( tCellPenalizedMassTimesDensity
                        * tCentroid(aCellOrdinal, tDimIndex) ) / tAppxTotalMass;
                auto tMyCenterGravityOverLimit = ( tMyCurrentCenterGravityContribution / tTargetCenterGravity(tDimIndex) );
                auto tMyCenterGravityOverLimitMinusOne = tMyCenterGravityOverLimit - static_cast<Plato::Scalar>(1.0);
                auto tMyDimConstraint = tMyCenterGravityOverLimitMinusOne * ( (tMyCenterGravityOverLimitMinusOne
                        * tMyCenterGravityOverLimitMinusOne) + static_cast<Plato::Scalar>(1.0) );
                tConstraints(aCellOrdinal, tDimIndex) += tMyDimConstraint;
            }
        },"Compute Cell Constraints");

        // ****** UPDATE LAGRANGE MULTIPLIERS ******
        auto tHostLagrangeMultipliers = Kokkos::create_mirror(mLagrangeMultipliers);
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mSpaceDim; tDimIndex++)
        {
            Plato::Scalar tMyConstraint = 0;
            auto tSubView = Kokkos::subview(tConstraints, Kokkos::ALL(), tDimIndex);
            Plato::local_sum(tSubView, tMyConstraint);
            auto tSuggestedLagrangeMultiplier = tHostLagrangeMultipliers(tDimIndex) + (mAugLagPenalty * tMyConstraint);
            tHostLagrangeMultipliers(tDimIndex) = Omega_h::max2(tSuggestedLagrangeMultiplier, static_cast<Plato::Scalar>(0.0));
        }
        Kokkos::deep_copy(mLagrangeMultipliers, tHostLagrangeMultipliers);
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
        // ****** INITIALIZE TEMPORARY FUNCTORS ******
        ::SIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tAugLagPenalty = mAugLagPenalty;
        auto tTargetCenterGravity = mTargetCenterGravity;
        auto tMaterialDensity = mCellMaterialDensity;
        auto tLagrangeMultipliers = mLagrangeMultipliers;
        auto tMassNormalizationMultiplier = mMassNormalizationMultiplier;

        // ****** INITIALIZE TEMPORARY ARRAYS ******
        const Plato::OrdinalType tNumCells = mMesh.nelems();
        Plato::ScalarVectorT<ResultT> tConstraint("constraint", tNumCells);
        Plato::ScalarMultiVectorT<ConfigT> tCentroids("Centroids", tNumCells, mSpaceDim);

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tCubWeight = tCubatureRule.getCubWeight();
        auto tBasisFunc = tCubatureRule.getBasisFunctions();
        constexpr Plato::Scalar tLagrangianMultiplier = static_cast<Plato::Scalar>(1.0/ mSpaceDim);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute mass objective contribution to augmented Lagrangian function
            ConfigT tCellVolume = 0.0;
            tComputeCellVolume(aCellOrdinal, aConfigWS, tCellVolume);
            tCellVolume *= tCubWeight;
            ControlT tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControlWS);
            ControlT tPenalizedMass = tSIMP(tCellMass);
            ResultT tCellPenalizedMassTimesDensity = tPenalizedMass * tCellVolume * tMaterialDensity;
            aResultWS(aCellOrdinal) = tCellPenalizedMassTimesDensity / tMassNormalizationMultiplier;

            // Compute constraint contribution to augmented Lagrangian function
            Plato::compute_simplex_centroid<mSpaceDim>(aCellOrdinal, aConfigWS, tCentroids);
            ResultT tAppxTotalMass = static_cast<Plato::Scalar>(tNumCells) * tCellMass * tCellVolume * tMaterialDensity;
            for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mSpaceDim; tDimIndex++)
            {
                ResultT tMyCurrentCenterGravityContribution = ( tCellPenalizedMassTimesDensity
                        * tCentroids(aCellOrdinal, tDimIndex) ) / tAppxTotalMass;
                ResultT tMyCenterGravityOverLimit = ( tMyCurrentCenterGravityContribution / tTargetCenterGravity(tDimIndex) );
                ResultT tMyCenterGravityOverLimitMinusOne = tMyCenterGravityOverLimit - static_cast<Plato::Scalar>(1.0);
                ResultT tMyConstraint = tMyCenterGravityOverLimitMinusOne * ( (tMyCenterGravityOverLimitMinusOne
                        * tMyCenterGravityOverLimitMinusOne) + static_cast<Plato::Scalar>(1.0) );
                ResultT tLagrangianTermOne = tLagrangeMultipliers(tDimIndex) * tMyConstraint;
                ResultT tLagrangianTermTwo = tAugLagPenalty * static_cast<Plato::Scalar>(0.5) * tMyConstraint * tMyConstraint;
                tConstraint(aCellOrdinal) += tLagrangianTermOne + tLagrangianTermTwo;
            }

            aResultWS(aCellOrdinal) += ( tLagrangianMultiplier * tConstraint(aCellOrdinal) );
        },"Evaluate Augmented Lagrangian Function");
    }

    /******************************************************************************//**
     * @brief Compute initial structural mass (i.e. structural mass with ersatz densities set to one)
    **********************************************************************************/
    void computeInitialStructuralMass()
    {
        auto tNumCells = mMesh.nelems();
        Plato::ScalarMultiVectorT<Plato::Scalar> tDensities("densities", tNumCells, mNumNodesPerCell);
        Kokkos::deep_copy(tDensities, 1.0);

        Plato::NodeCoordinate<mSpaceDim> tCoordinates(&mMesh);
        Plato::ScalarArray3DT<Plato::Scalar> tConfig("configuration", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::workset_config_scalar<mSpaceDim, mNumNodesPerCell>(tNumCells, tCoordinates, tConfig);

        Plato::Scalar tOutput = 0;
        Plato::StructuralMass<mSpaceDim> tComputeStructuralMass(mCellMaterialDensity);
        tComputeStructuralMass(tNumCells, tDensities, tConfig, tOutput);
        mMassNormalizationMultiplier = tOutput;
    }
};
// class CenterGravityCriterion

} // namespace Plato

#include "plato/Mechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::CenterGravityCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::CenterGravityCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::CenterGravityCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::CenterGravityCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
extern template class Plato::CenterGravityCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::CenterGravityCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::CenterGravityCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::CenterGravityCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
extern template class Plato::CenterGravityCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::CenterGravityCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::CenterGravityCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::CenterGravityCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
