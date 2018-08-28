/*
 * DynamicCompliance.hpp
 *
 *  Created on: Apr 25, 2018
 */

#ifndef DYNAMICCOMPLIANCE_HPP_
#define DYNAMICCOMPLIANCE_HPP_

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_matrix.hpp>

#include <Teuchos_ParameterList.hpp>

#include "ImplicitFunctors.hpp"
#include "LinearElasticMaterial.hpp"

#include "plato/StateValues.hpp"
#include "plato/ApplyPenalty.hpp"
#include "plato/ComplexStrain.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/ApplyProjection.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ComplexLinearStress.hpp"
#include "plato/ComplexElasticEnergy.hpp"
#include "plato/ComplexInertialEnergy.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/SimplexStructuralDynamics.hpp"

namespace Plato
{

template<typename EvaluationType, class PenaltyFuncType, class ProjectionFuncType>
class DynamicCompliance:
        public Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim, EvaluationType::NumControls>,
        public AbstractScalarFunction<EvaluationType>
{
private:
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numVoigtTerms;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::mComplexSpaceDim;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numDofsPerNode;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numDofsPerCell;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numNodesPerCell;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

private:
    Plato::Scalar mDensity;

    PenaltyFuncType mPenaltyFunction;
    ProjectionFuncType mProjectionFunction;
    Plato::ApplyPenalty<PenaltyFuncType> mApplyPenalty;
    Plato::ApplyProjection<ProjectionFuncType> mApplyProjection;

    Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms> mCellStiffness;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

public:
    /**************************************************************************/
    DynamicCompliance(Omega_h::Mesh& aMesh,
                      Omega_h::MeshSets& aMeshSets,
                      Plato::DataMap& aDataMap, 
                      Teuchos::ParameterList& aProblemParams,
                      Teuchos::ParameterList& aPenaltyParams) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Dynamic Energy"),
            mDensity(aProblemParams.get<double>("Material Density", 1.0)),
            mProjectionFunction(),
            mPenaltyFunction(aPenaltyParams),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        // Create material model and get stiffness
        Plato::ElasticModelFactory<EvaluationType::SpatialDim> tElasticModelFactory(aProblemParams);
        auto tMaterialModel = tElasticModelFactory.create();
        mCellStiffness = tMaterialModel->getStiffnessMatrix();
    }
    /**************************************************************************/
    DynamicCompliance(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap& aDataMap) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Dynamic Energy"),
            mDensity(1.0),
            mProjectionFunction(),
            mPenaltyFunction(),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        // Create material model and get stiffness
        Teuchos::ParameterList tParamList;
        tParamList.set < Plato::Scalar > ("Poissons Ratio", 1.0);
        tParamList.set < Plato::Scalar > ("Youngs Modulus", 0.3);
        Plato::IsotropicLinearElasticMaterial<EvaluationType::SpatialDim> tDefaultMaterialModel(tParamList);
        mCellStiffness = tDefaultMaterialModel.getStiffnessMatrix();
    }

    /**************************************************************************/
    virtual ~DynamicCompliance()
    {
    }
    /**************************************************************************/

    /*************************************************************************
     * Evaluate f(u,z)=\frac{1}{2}u^{T}(K(z) - \omega^2 M(z))u, where u denotes
     * states, z denotes controls, K denotes the stiffness matrix and M denotes
     * the mass matrix.
     **************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        using StrainScalarType =
        typename Plato::fad_type_t<Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

        // Elastic forces functors
        Plato::ComplexElasticEnergy<m_numVoigtTerms> tComputeElasticEnergy;
        Plato::ComputeGradientWorkset<EvaluationType::SpatialDim> tComputeGradientWorkset;
        Plato::ComplexStrain<EvaluationType::SpatialDim, m_numDofsPerNode> tComputeVoigtStrain;
        Plato::ComplexLinearStress<EvaluationType::SpatialDim, m_numVoigtTerms> tComputeVoigtStress(mCellStiffness);

        // Inertial forces functors
        Plato::StateValues tComputeStateValues;
        Plato::ComplexInertialEnergy<EvaluationType::SpatialDim> tComputeInertialEnergy(aTimeStep, mDensity);

        // Elastic forces containers
        auto tNumCells = aControl.extent(0);
        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("CellWeight", tNumCells);
        Plato::ScalarArray3DT<StrainScalarType> tCellStrain("CellStrain", tNumCells, mComplexSpaceDim, m_numVoigtTerms);
        Plato::ScalarArray3DT<ResultScalarType> tCellStress("CellStress", tNumCells, mComplexSpaceDim, m_numVoigtTerms);
        Plato::ScalarArray3DT<ConfigScalarType> tCellGradient("Gradient", tNumCells, m_numNodesPerCell, EvaluationType::SpatialDim);

        // Inertial forces containers
        Plato::ScalarVectorT<ResultScalarType> tElasticEnergy("ElasticEnergy", tNumCells);
        Plato::ScalarVectorT<ResultScalarType> tInertialEnergy("InertialEnergy", tNumCells);
        Plato::ScalarMultiVectorT<StateScalarType> tStateValues("StateValues", tNumCells, m_numDofsPerNode);

        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;
        auto & tPenaltyFunction = mPenaltyFunction;
        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Internal forces contribution
            tComputeGradientWorkset(aCellOrdinal, tCellGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;
            tComputeVoigtStrain(aCellOrdinal, aState, tCellGradient, tCellStrain);
            tComputeVoigtStress(aCellOrdinal, tCellStrain, tCellStress);

            // Apply penalty to internal forces
            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl);
            tApplyPenalty(aCellOrdinal, tCellDensity, tCellStress);
            tComputeElasticEnergy(aCellOrdinal, tCellStress, tCellStrain, tElasticEnergy);
            tElasticEnergy(aCellOrdinal) *= tCellVolume(aCellOrdinal);

            // Inertial forces contribution
            tComputeStateValues(aCellOrdinal, tBasisFunctions, aState, tStateValues);
            tComputeInertialEnergy(aCellOrdinal, tCellVolume, tStateValues, tInertialEnergy);
            ControlScalarType tPenaltyValue = tPenaltyFunction(tCellDensity);
            tInertialEnergy(aCellOrdinal) *= tPenaltyValue;

            // Add inertial forces contribution
            aResult(aCellOrdinal) = static_cast<Plato::Scalar>(0.5) *
                ( tElasticEnergy(aCellOrdinal) + tInertialEnergy(aCellOrdinal) );
        }, "Dynamic Compliance Calculation");
    }
};
// class DynamicCompliance

}//namespace Plato

#endif /* DYNAMICCOMPLIANCE_HPP_ */
