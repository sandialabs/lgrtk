/*
 * AdjointStructuralDynamicsResidual.hpp
 *
 *  Created on: Apr 29, 2018
 */

#ifndef ADJOINTSTRUCTURALDYNAMICSRESIDUAL_HPP_
#define ADJOINTSTRUCTURALDYNAMICSRESIDUAL_HPP_

#include <memory>

#include <Teuchos_ParameterList.hpp>

#include "ImplicitFunctors.hpp"
#include "LinearElasticMaterial.hpp"

#include "plato/StateValues.hpp"
#include "plato/ApplyPenalty.hpp"
#include "plato/ComplexStrain.hpp"
#include "plato/InertialForces.hpp"
#include "plato/ApplyProjection.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ComplexLinearStress.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ComplexStressDivergence.hpp"
#include "plato/SimplexStructuralDynamics.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/AdjointComplexRayleighDamping.hpp"
#include "plato/StructuralDynamicsCellResidual.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, class PenaltyFunctionType, class ProjectionType>
class AdjointStructuralDynamicsResidual:
        public Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>,
        public AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    using Simplex<EvaluationType::SpatialDim>::m_numNodesPerCell;
    using Simplex<EvaluationType::SpatialDim>::m_numSpatialDims;

    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numVoigtTerms;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::mComplexSpaceDim;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numDofsPerCell;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numDofsPerNode;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

private:
    Plato::Scalar mDensity;
    Plato::Scalar mMassPropDamp;
    Plato::Scalar mStiffPropDamp;

    ProjectionType mProjectionFunction;
    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyPenalty<PenaltyFunctionType> mApplyPenalty;
    Plato::ApplyProjection<ProjectionType> mApplyProjection;

    Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms> mCellStiffness;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<m_numSpatialDims>> mCubatureRule;

public:
    /******************************************************************************//**
     *
     * @brief Constructor
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh sets data base
     * @param [in] aDataMap problem-specific data storage
     * @param [in] aProblemParams parameter list with input data
     * @param [in] aPenaltyParams parameter list with penalty model input data
     *
    **********************************************************************************/
    explicit AdjointStructuralDynamicsResidual(Omega_h::Mesh& aMesh, 
                                               Omega_h::MeshSets& aMeshSets,
                                               Plato::DataMap& aDataMap,
                                               Teuchos::ParameterList & aProblemParams,
                                               Teuchos::ParameterList& aPenaltyParams) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mDensity(aProblemParams.get<double>("Material Density", 1.0)),
            mMassPropDamp(aProblemParams.get<double>("Rayleigh Mass Damping", 0.0)),
            mStiffPropDamp(aProblemParams.get<double>("Rayleigh Stiffness Damping", 0.0)),
            mProjectionFunction(),
            mPenaltyFunction(aPenaltyParams),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<m_numSpatialDims>>())
    {
        this->initialize(aProblemParams);
    }

    /******************************************************************************//**
     *
     * @brief Constructor
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh sets data base
     * @param [in] aDataMap problem-specific data storage
     * @param [in] aProblemParams parameter list with input data
     *
    **********************************************************************************/
    explicit AdjointStructuralDynamicsResidual(Omega_h::Mesh& aMesh, 
                                               Omega_h::MeshSets& aMeshSets, 
                                               Plato::DataMap& aDataMap, 
                                               Teuchos::ParameterList & aProblemParams) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mDensity(aProblemParams.get<double>("Material Density", 1.0)),
            mMassPropDamp(aProblemParams.get<double>("Rayleigh Mass Damping", 0.0)),
            mStiffPropDamp(aProblemParams.get<double>("Rayleigh Stiffness Damping", 0.0)),
            mProjectionFunction(),
            mPenaltyFunction(),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<m_numSpatialDims>>())
    {
        this->initialize(aProblemParams);
    }

    /******************************************************************************//**
     *
     * @brief Constructor
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh sets data base
     * @param [in] aDataMap problem-specific data storage
     *
    **********************************************************************************/
    explicit AdjointStructuralDynamicsResidual(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap& aDataMap) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mDensity(1),
            mMassPropDamp(0.0),
            mStiffPropDamp(0.0),
            mProjectionFunction(),
            mPenaltyFunction(),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<m_numSpatialDims>>())
    {
        this->initialize();
    }

    /******************************************************************************//**
    *
    * @brief Destructor
    *
    **********************************************************************************/
    ~AdjointStructuralDynamicsResidual()
    {
    }

    /******************************************************************************//**
     *
     * @brief Set mass proportional damping constant
     * @param [in] aInput mass proportional damping constant
     *
    **********************************************************************************/
    void setMassPropDamping(const Plato::Scalar& aInput)
    {
        mMassPropDamp = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set stiffness proportional damping constant
     * @param [in] aInput stiffness proportional damping constant
     *
    **********************************************************************************/
    void setStiffPropDamping(const Plato::Scalar& aInput)
    {
        mStiffPropDamp = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set material density
     * @param [in] aInput material density
     *
    **********************************************************************************/
    void setMaterialDensity(const Plato::Scalar& aInput)
    {
        mDensity = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set material stiffness constants (i.e. Lame constants)
     * @param [in] aInput material stiffness constants
     *
    **********************************************************************************/
    void setMaterialStiffnessConstants(const Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms>& aInput)
    {
        mCellStiffness = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set isotropic linear elastic material constants (i.e. Lame constants)
     * @param [in] aYoungsModulus Young's modulus
     * @param [in] aPoissonsRatio Poisson's ratio
     *
    **********************************************************************************/
    void setIsotropicLinearElasticMaterial(const Plato::Scalar& aYoungsModulus, const Plato::Scalar& aPoissonsRatio)
    {
        Plato::IsotropicLinearElasticMaterial<m_numSpatialDims> tMaterialModel(aYoungsModulus, aPoissonsRatio);
        mCellStiffness = tMaterialModel.getStiffnessMatrix();
    }

    /******************************************************************************//**
     *
     * @brief Evaluate structural dynamics adjoint residual.
     *
     * The structural dynamics adjoint residual is given by:
     *
     * \f$\mathbf{R} = \left(\mathbf{K} + \mathbf{C} - \omega^2\rho\mathbf{M}\right)
     * \mathbf{u}_{n} - \frac{\partial\mathbf{f}}{\partial\mathbf{u}},
     *
     * where \f$n=1,\dots,N\f$ and \f$N\f$ is the number of frequency steps,
     * \f$\mathbf{M}$\f, \f$\mathbf{C}\f$ and \f$\mathbf{K}\f$ are the mass, damping
     * and stiffness matrices, \f$\mathbf{u}\f$ is the state vector, \f$\mathbf{f}\f$
     * is a criterion of interest (e.g. objective function), \f$\omega\f$ is the angular
     * frequency and \f$\rho\f$ is the material density.
     *
     * @param [in] aState states per cells
     * @param [in] aControl controls per cells
     * @param [in] aConfiguration coordinates per cells
     * @param [in,out] aResidual residual per cells
     * @param [in] aAngularFrequency angular frequency
     *
    **********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfiguration,
                  Plato::ScalarMultiVectorT<ResultScalarType> & aResidual,
                  Plato::Scalar aAngularFrequency = 0.0) const
    {
        // Elastic force functors
        Plato::ComputeGradientWorkset<m_numSpatialDims> tComputeGradientWorkset;
        Plato::ComplexStrain<m_numSpatialDims, m_numDofsPerNode> tComputeVoigtStrain;
        Plato::ComplexStressDivergence<m_numSpatialDims, m_numDofsPerNode> tComputeStressDivergence;
        Plato::ComplexLinearStress<m_numSpatialDims, m_numVoigtTerms> tComputeVoigtStress(mCellStiffness);
        // Damping force functors
        Plato::AdjointComplexRayleighDamping<m_numSpatialDims> tComputeDamping(mMassPropDamp, mStiffPropDamp);
        // Inertial force functors
        Plato::StateValues tComputeStateValues;
        Plato::InertialForces tComputeInertialForces(mDensity);

        using StrainScalarType =
        typename Plato::fad_type_t<Plato::SimplexStructuralDynamics<m_numSpatialDims>, StateScalarType, ConfigScalarType>;
        // Elastic forces containers
        auto tNumCells = aState.extent(0);
        Plato::ScalarMultiVectorT<ResultScalarType> tElasticForces("ElasticForces", tNumCells, m_numDofsPerCell);
        Plato::ScalarArray3DT<StrainScalarType> tCellStrain("Strain", tNumCells, mComplexSpaceDim, m_numVoigtTerms);
        Plato::ScalarArray3DT<ResultScalarType> tCellStress("Stress", tNumCells, mComplexSpaceDim, m_numVoigtTerms);
        Plato::ScalarArray3DT<ConfigScalarType> tCellGradient("Gradient", tNumCells, m_numNodesPerCell, m_numSpatialDims);
        // Damping forces containers
        Plato::ScalarMultiVectorT<ResultScalarType> tDampingForces("DampingForces", tNumCells, m_numDofsPerCell);
        // Inertial forces containers
        Plato::ScalarMultiVectorT<StateScalarType> tStateValues("StateValues", tNumCells, m_numDofsPerNode);
        Plato::ScalarMultiVectorT<ResultScalarType> tInertialForces("InertialForces", tNumCells, m_numDofsPerCell);
        // Cell volumes container
        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("Cell Volumes", tNumCells);

        // Copy data from host into device
        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;
        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        auto tOmegaTimesOmega = aAngularFrequency * aAngularFrequency;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute elastic forces
            tComputeGradientWorkset(aCellOrdinal, tCellGradient, aConfiguration, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;
            tComputeVoigtStrain(aCellOrdinal, aState, tCellGradient, tCellStrain);
            tComputeVoigtStress(aCellOrdinal, tCellStrain, tCellStress);
            tComputeStressDivergence(aCellOrdinal, tCellVolume, tCellGradient, tCellStress, tElasticForces);
            // Compute inertial forces
            tComputeStateValues(aCellOrdinal, tBasisFunctions, aState, tStateValues);
            tComputeInertialForces(aCellOrdinal, tCellVolume, tBasisFunctions, tStateValues, tInertialForces);
            // Compute damping forces
            tComputeDamping(aCellOrdinal, tElasticForces, tInertialForces, tDampingForces);
            // Apply penalty to elastic, damping and inertial forces
            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl);
            tApplyPenalty(aCellOrdinal, tCellDensity, tDampingForces);
            tApplyPenalty(aCellOrdinal, tCellDensity, tElasticForces);
            tApplyPenalty(aCellOrdinal, tCellDensity, tInertialForces);
            // Compute structural dynamics residual for the adjoint problem
            Plato::structural_dynamics_cell_residual<m_numDofsPerCell>
                (aCellOrdinal, tOmegaTimesOmega, tElasticForces, tDampingForces, tInertialForces, aResidual);
        }, "Adjoint Elastodynamcis Residual Calculation");
    }

private:
    /**************************************************************************//**
     *
     * @brief Initialize default material stiffness constants (i.e. Lame constants).
     *
    ******************************************************************************/
    void initialize()
    {
        // Create material model and get stiffness
        Teuchos::ParameterList tParamList;
        tParamList.set < Plato::Scalar > ("Poissons Ratio", 1.0);
        tParamList.set < Plato::Scalar > ("Youngs Modulus", 0.3);
        Plato::IsotropicLinearElasticMaterial<m_numSpatialDims> tDefaultMaterialModel(tParamList);
        mCellStiffness = tDefaultMaterialModel.getStiffnessMatrix();
    }

    /**************************************************************************//**
     *
     * @brief Initialize problem input parameters
     * @param [in] aParamList parameter list with input data
     *
    ******************************************************************************/
    void initialize(Teuchos::ParameterList & aProblemParams)
    {
        // Create material model and get stiffness
        Plato::ElasticModelFactory<m_numSpatialDims> tElasticModelFactory(aProblemParams);
        auto tMaterialModel = tElasticModelFactory.create();
        mCellStiffness = tMaterialModel->getStiffnessMatrix();
    }

};
// class AdjointStructuralDynamicsResidual

}// namespace Plato

#endif /* ADJOINTSTRUCTURALDYNAMICSRESIDUAL_HPP_ */
