#ifndef ELECTROELASTOSTATIC_RESIDUAL_HPP
#define ELECTROELASTOSTATIC_RESIDUAL_HPP

#include <memory>

#include "plato/PlatoTypes.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/SimplexElectromechanics.hpp"
#include "plato/EMKinematics.hpp"
#include "plato/EMKinetics.hpp"
#include "plato/StressDivergence.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/CellForcing.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/LinearElectroelasticMaterial.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/BodyLoads.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElectroelastostaticResidual :
        public Plato::SimplexElectromechanics<EvaluationType::SpatialDim>,
        public Plato::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    static constexpr Plato::OrdinalType NElecDims = 1;
    static constexpr Plato::OrdinalType NMechDims = SpaceDim;

    static constexpr Plato::OrdinalType EDofOffset = SpaceDim;
    static constexpr Plato::OrdinalType MDofOffset = 0;

    using Plato::SimplexElectromechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numDofsPerCell;

    using Plato::AbstractVectorFunction<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunction<EvaluationType>::m_dataMap;
    using Plato::AbstractVectorFunction<EvaluationType>::mMeshSets;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim, SpaceDim,        IndicatorFunctionType> m_applyEDispWeighting;
    ApplyWeighting<SpaceDim, m_numVoigtTerms, IndicatorFunctionType> m_applyStressWeighting;

    std::shared_ptr<Plato::BodyLoads<SpaceDim,m_numDofsPerNode>> m_bodyLoads;

    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NMechDims, m_numDofsPerNode, MDofOffset>> m_boundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NElecDims, m_numDofsPerNode, EDofOffset>> m_boundaryCharges;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpaceDim>> m_materialModel;

    std::vector<std::string> m_plottable;

public:
    /**************************************************************************/
    ElectroelastostaticResidual(Omega_h::Mesh& aMesh,
                               Omega_h::MeshSets& aMeshSets,
                               Plato::DataMap& aDataMap,
                               Teuchos::ParameterList& aProblemParams,
                               Teuchos::ParameterList& aPenaltyParams) :
            AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            m_indicatorFunction(aPenaltyParams),
            m_applyStressWeighting(m_indicatorFunction),
            m_applyEDispWeighting(m_indicatorFunction),
            m_bodyLoads(nullptr),
            m_boundaryLoads(nullptr),
            m_boundaryCharges(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        // create material model and get stiffness
        //
        Plato::ElectroelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
        m_materialModel = mmfactory.create();
  

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            m_bodyLoads = std::make_shared<Plato::BodyLoads<SpaceDim,m_numDofsPerNode>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse mechanical boundary Conditions
        // 
        if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            m_boundaryLoads =   std::make_shared<Plato::NaturalBCs<SpaceDim, NMechDims, m_numDofsPerNode, MDofOffset>>(aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
  
        // parse electrical boundary Conditions
        // 
        if(aProblemParams.isSublist("Electrical Natural Boundary Conditions"))
        {
            m_boundaryCharges = std::make_shared<Plato::NaturalBCs<SpaceDim, NElecDims, m_numDofsPerNode, EDofOffset>>(aProblemParams.sublist("Electrical Natural Boundary Conditions"));
        }
  
        auto tResidualParams = aProblemParams.sublist("Electroelastostatics");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
          m_plottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();

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

      using GradScalarType =
      typename Plato::fad_type_t<Plato::SimplexElectromechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Plato::EMKinematics<SpaceDim>                  kinematics;
      Plato::EMKinetics<SpaceDim>                    kinetics(m_materialModel);
      
      Plato::StressDivergence<SpaceDim, m_numDofsPerNode, MDofOffset> stressDivergence;
      Plato::FluxDivergence  <SpaceDim, m_numDofsPerNode, EDofOffset> edispDivergence;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",tNumCells);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient", tNumCells, m_numNodesPerCell, SpaceDim);

      Plato::ScalarMultiVectorT<GradScalarType> strain("strain", tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> efield("efield", tNumCells, SpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> edisp ("edisp" , tNumCells, SpaceDim);
    
      auto quadratureWeight = mCubatureRule->getCubWeight();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, config, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;

        // compute strain and electric field
        //
        kinematics(cellOrdinal, strain, efield, state, gradient);
    
        // compute stress and electric displacement
        //
        kinetics(cellOrdinal, stress, edisp, strain, efield);

      }, "Cauchy stress");

      auto& applyStressWeighting = m_applyStressWeighting;
      auto& applyEDispWeighting  = m_applyEDispWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        // apply weighting
        //
        applyStressWeighting(cellOrdinal, stress, control);
        applyEDispWeighting (cellOrdinal, edisp,  control);
    
        // compute divergence
        //
        stressDivergence(cellOrdinal, result, stress, gradient, cellVolume);
        edispDivergence (cellOrdinal, result, edisp,  gradient, cellVolume);
      }, "Apply weighting and compute divergence");

      if( m_bodyLoads != nullptr )
      {
          m_bodyLoads->get( mMesh, state, control, result );
      }

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, state, control, result );
      }

      if( m_boundaryCharges != nullptr )
      {
          m_boundaryCharges->get( &mMesh, mMeshSets, state, control, result );
      }

      if( std::count(m_plottable.begin(),m_plottable.end(),"strain") ) toMap(m_dataMap, strain, "strain");
      if( std::count(m_plottable.begin(),m_plottable.end(),"efield") ) toMap(m_dataMap, strain, "efield");
      if( std::count(m_plottable.begin(),m_plottable.end(),"stress") ) toMap(m_dataMap, stress, "stress");
      if( std::count(m_plottable.begin(),m_plottable.end(),"edisp" ) ) toMap(m_dataMap, stress, "edisp" );

    }
};
// class ElectroelastostaticResidual

} // namespace Plato
#endif
