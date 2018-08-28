/*
 * StructuralDynamicsResidual.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef STRUCTURALDYNAMICSRESIDUAL_HPP_
#define STRUCTURALDYNAMICSRESIDUAL_HPP_

#include <memory>

#include <Teuchos_ParameterList.hpp>

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "LinearElasticMaterial.hpp"

#include "plato/StateValues.hpp"
#include "plato/ApplyPenalty.hpp"
#include "plato/ComplexStrain.hpp"
#include "plato/InertialForces.hpp"
#include "plato/ApplyProjection.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ComplexLinearStress.hpp"
#include "plato/ComplexRayleighDamping.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ComplexStressDivergence.hpp"
#include "plato/SimplexStructuralDynamics.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/StructuralDynamicsCellResidual.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, class PenaltyFunctionType, class ProjectionType>
class StructuralDynamicsResidual:
        public Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>,
        public AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    using Simplex<EvaluationType::SpatialDim>::m_numSpatialDims;
    using Simplex<EvaluationType::SpatialDim>::m_numNodesPerCell;

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

    std::shared_ptr<Plato::BodyLoads<m_numSpatialDims, m_numDofsPerNode>> mBodyLoads;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<m_numSpatialDims>> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<m_numSpatialDims, m_numDofsPerNode>> mBoundaryLoads;

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
    explicit StructuralDynamicsResidual(Omega_h::Mesh& aMesh,
                                        Omega_h::MeshSets& aMeshSets,
                                        Plato::DataMap& aDataMap,
                                        Teuchos::ParameterList & aProblemParams,
                                        Teuchos::ParameterList & aPenaltyParams) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mDensity(1.0),
            mMassPropDamp(0.0),
            mStiffPropDamp(0.0),
            mProjectionFunction(),
            mPenaltyFunction(aPenaltyParams),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mBodyLoads(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<m_numSpatialDims>>()),
            mBoundaryLoads(nullptr)
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
    explicit StructuralDynamicsResidual(Omega_h::Mesh& aMesh,
                                        Omega_h::MeshSets& aMeshSets,
                                        Plato::DataMap& aDataMap,
                                        Teuchos::ParameterList & aProblemParams) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mDensity(1.0),
            mMassPropDamp(0.0),
            mStiffPropDamp(0.0),
            mProjectionFunction(),
            mPenaltyFunction(),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mBodyLoads(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<m_numSpatialDims>>()),
            mBoundaryLoads(nullptr)
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
    explicit StructuralDynamicsResidual(Omega_h::Mesh& aMesh,
                                        Omega_h::MeshSets& aMeshSets,
                                        Plato::DataMap& aDataMap) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mDensity(1.0),
            mMassPropDamp(0.0),
            mStiffPropDamp(0.0),
            mProjectionFunction(),
            mPenaltyFunction(),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mBodyLoads(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<m_numSpatialDims>>()),
            mBoundaryLoads(nullptr)
    {
        this->initialize();
    }

    /******************************************************************************//**
     *
     * @brief Destructor
     *
    **********************************************************************************/
    ~StructuralDynamicsResidual()
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
     * @brief Evaluate structural dynamics residual.
     *
     * The structural dynamics residual is given by:
     *
     * \f$\mathbf{R} = \left(\mathbf{K} + \mathbf{C} - \omega^2\rho\mathbf{M}\right)
     * \mathbf{u}_{n} - \mathbf{f},
     *
     * where \f$n=1,\dots,N\f$ and \f$N\f$ is the number of frequency steps,
     * \f$\mathbf{M}$\f, \f$\mathbf{C}\f$ and \f$\mathbf{K}\f$ are the mass, damping
     * and stiffness matrices, \f$\mathbf{u}\f$ is the state vector, \f$\mathbf{f}\f$
     * is the force vector, \f$\omega\f$ is the angular frequency and \f$\rho\f$ is
     * the material density.
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
        Plato::ComplexRayleighDamping<m_numSpatialDims> tComputeDamping(mMassPropDamp, mStiffPropDamp);
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
            // Compute structural dynamics residual
            Plato::structural_dynamics_cell_residual<m_numDofsPerCell>
                (aCellOrdinal, tOmegaTimesOmega, tElasticForces, tDampingForces, tInertialForces, aResidual);
        }, "Structural Dynamics Residual Calculation");

        // add body loads contribution
        if(mBodyLoads != nullptr)
        {
            auto tMesh = AbstractVectorFunction<EvaluationType>::getMesh();
            mBodyLoads->get(tMesh, aState, aControl, aResidual);
        }

        // add neumann loads contribution
        if( mBoundaryLoads != nullptr )
        {
            auto tMesh = AbstractVectorFunction<EvaluationType>::getMesh();
            auto tMeshSets = AbstractVectorFunction<EvaluationType>::getMeshSets();
            mBoundaryLoads->get(&tMesh, tMeshSets, aState, aControl, aResidual);
        }
    }

private:
    /**************************************************************************//**
     * 
     * \brief Checks if the material model sublist is defined in the input file.
     *
     * \param aParameterList Teuchos parameter list
     *
    ******************************************************************************/
    void isMaterialModelSublistDefined(Teuchos::ParameterList & aParameterList)
    {
        if(aParameterList.isSublist("Material Model") == false)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                << ", LINE: " << __LINE__ << "\nMESSAGE: MATERIAL MODEL SUBLIST IS NOT DEFINED IN THE INPUT FILE.\n"
                << "USER SHOULD DEFINE THE MATERIAL MODEL SUBLIST IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }
    
    /**************************************************************************//**
     * 
     * \brief Checks if the material density parameter is defined in the input file.
     *
     * \param aParameterList Teuchos parameter list
     *
    ******************************************************************************/
    void isMaterialDensityDefined(Teuchos::ParameterList & aParameterList)
    {
        auto tMaterialParamList = aParameterList.sublist("Material Model");
        if(tMaterialParamList.isParameter("Density") == false)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                << ", LINE: " << __LINE__ << "\nMESSAGE: MATERIAL DENSITY IS NOT DEFINED IN THE INPUT FILE.\n"
                << "USER SHOULD DEFINE THE MATERIAL DENSITY IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }
    
    /**************************************************************************//**
     * 
     * \brief Checks if the frequency steps sublist is defined in the input file.
     *
     * \param aParameterList Teuchos parameter list
     *
    ******************************************************************************/
    void isFrequencyStepsSublistDefined(Teuchos::ParameterList & aParameterList)
    {
        if(aParameterList.isSublist("Frequency Steps") == false)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                << ", LINE: " << __LINE__ << "\nMESSAGE: FREQUENCY STEPS SUBLIST IS NOT DEFINED IN THE INPUT FILE.\n"
                << "\nUSER SHOULD DEFINE THE FREQUENCY STEPS SUBLIST IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }

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
    void initialize(Teuchos::ParameterList & aParamList)
    {
        // Parse material density
        this->isMaterialModelSublistDefined(aParamList);
        this->isMaterialDensityDefined(aParamList);
        auto tMaterialParamList = aParamList.sublist("Material Model");
        mDensity = tMaterialParamList.get<Plato::Scalar>("Density");

        // Parse Rayleigh damping coefficients
        this->isFrequencyStepsSublistDefined(aParamList);
        mMassPropDamp = tMaterialParamList.get<Plato::Scalar>("Mass Proportional Damping", 0.0);
        mStiffPropDamp = tMaterialParamList.get<Plato::Scalar>("Stiffness Proportional Damping", 0.0);

        // Create material model and get stiffness
        Plato::ElasticModelFactory<m_numSpatialDims> tElasticModelFactory(aParamList);
        auto tMaterialModel = tElasticModelFactory.create();
        mCellStiffness = tMaterialModel->getStiffnessMatrix();

        // Parse body loads
        if(aParamList.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<m_numSpatialDims, m_numDofsPerNode>>(aParamList.sublist("Body Loads"));
        }

        // Parse Neumann loads
        if(aParamList.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<m_numSpatialDims, m_numDofsPerNode>>(aParamList.sublist("Natural Boundary Conditions"));
        }
    }
};
// class StructuralDynamicsResidual

} // namespace Plato

#endif /* STRUCTURALDYNAMICSRESIDUAL_HPP_ */
