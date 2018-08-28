#ifndef ELASTOSTATIC_RESIDUAL_HPP
#define ELASTOSTATIC_RESIDUAL_HPP

#include <memory>

#include "plato/PlatoTypes.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/Strain.hpp"
#include "plato/LinearStress.hpp"
#include "plato/StressDivergence.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/CellForcing.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

#include "LinearElasticMaterial.hpp"
#include "NaturalBCs.hpp"
#include "BodyLoads.hpp"

namespace Plato {

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticResidual :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
        public AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    using Plato::SimplexMechanics<SpaceDim>::m_numVoigtTerms;
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerCell;

    using AbstractVectorFunction<EvaluationType>::mMesh;
    using AbstractVectorFunction<EvaluationType>::m_dataMap;
    using AbstractVectorFunction<EvaluationType>::mMeshSets;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms> m_cellStiffness;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim, m_numVoigtTerms, IndicatorFunctionType> m_applyWeighting;

    std::shared_ptr<Plato::BodyLoads<SpaceDim,m_numDofsPerNode>> m_bodyLoads;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>> m_boundaryLoads;
    std::shared_ptr<Plato::CellForcing<m_numVoigtTerms>> m_cellForcing;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    std::vector<std::string> m_plottable;

public:
    /**************************************************************************/
    ElastostaticResidual(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aProblemParams,
                         Teuchos::ParameterList& aPenaltyParams) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction),
            m_bodyLoads(nullptr),
            m_boundaryLoads(nullptr),
            m_cellForcing(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<SpaceDim> mmfactory(aProblemParams);
        auto materialModel = mmfactory.create();
        m_cellStiffness = materialModel->getStiffnessMatrix();
  

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            m_bodyLoads = std::make_shared<Plato::BodyLoads<SpaceDim,m_numDofsPerNode>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            m_boundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
        }
  
        // parse cell problem forcing
        //
        if(aProblemParams.isSublist("Cell Problem Forcing"))
        {
            int tColumnIndex = aProblemParams.sublist("Cell Problem Forcing").get<int>("Column Index");
            m_cellForcing = std::make_shared<Plato::CellForcing<m_numVoigtTerms>>(m_cellStiffness, tColumnIndex);
        }

        auto tResidualParams = aProblemParams.sublist("Elastostatics");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
          m_plottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();

    }

    /**************************************************************************/
    ElastostaticResidual(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap aDataMap) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            m_indicatorFunction(),
            m_applyWeighting(m_indicatorFunction),
            m_bodyLoads(nullptr),
            m_boundaryLoads(nullptr),
            m_cellForcing(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        // Create material model and get stiffness
        Teuchos::ParameterList tParamList;
        tParamList.set<Plato::Scalar>("Poissons Ratio", 1.0);
        tParamList.set<Plato::Scalar>("Youngs Modulus", 0.3);
        Plato::IsotropicLinearElasticMaterial<EvaluationType::SpatialDim> tDefaultMaterialModel(tParamList);
        m_cellStiffness = tDefaultMaterialModel.getStiffnessMatrix();
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & state,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & control,
                  const Plato::ScalarArray3DT<ConfigScalarType> & config,
                  Plato::ScalarMultiVectorT<ResultScalarType> & result,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        auto tNumCells = mMesh.nelems();

        using StrainScalarType =
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>,
                            StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Strain<SpaceDim>                        voigtStrain;
      LinearStress<SpaceDim>                  voigtStress(m_cellStiffness);
      StressDivergence<SpaceDim>              stressDivergence;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        strain("strain",tNumCells,m_numVoigtTerms);
    
      Plato::ScalarArray3DT<ConfigScalarType>
        gradient("gradient",tNumCells,m_numNodesPerCell,SpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType>
        stress("stress",tNumCells,m_numVoigtTerms);
    
      auto quadratureWeight = mCubatureRule->getCubWeight();
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, config, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;

        // compute strain
        //
        voigtStrain(cellOrdinal, strain, state, gradient);
    
        // compute stress
        //
        voigtStress(cellOrdinal, stress, strain);
      }, "Cauchy stress");

      if( m_cellForcing != nullptr )
      {
          m_cellForcing->add( stress );
      }

      auto& applyWeighting = m_applyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        // apply weighting
        //
        applyWeighting(cellOrdinal, stress, control);
    
        // compute stress divergence
        //
        stressDivergence(cellOrdinal, result, stress, gradient, cellVolume);
      }, "Apply weighting and compute divergence");

      if( m_bodyLoads != nullptr )
      {
          m_bodyLoads->get( mMesh, state, control, result );
      }

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, state, control, result );
      }

      if( std::count(m_plottable.begin(),m_plottable.end(),"strain") ) toMap(m_dataMap, strain, "strain");
      if( std::count(m_plottable.begin(),m_plottable.end(),"stress") ) toMap(m_dataMap, stress, "stress");

    }
};
// class ElastostaticResidual

} // namespace Plato
#endif
