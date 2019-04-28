#ifndef THERMOELASTOSTATIC_RESIDUAL_HPP
#define THERMOELASTOSTATIC_RESIDUAL_HPP

#include <memory>

#include "plato/PlatoTypes.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/SimplexThermomechanics.hpp"
#include "plato/TMKinematics.hpp"
#include "plato/TMKinetics.hpp"
#include "plato/StressDivergence.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/CellForcing.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/LinearThermoelasticMaterial.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/BodyLoads.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermoelastostaticResidual :
        public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
        public Plato::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    static constexpr int NThrmDims = 1;
    static constexpr int NMechDims = SpaceDim;

    static constexpr int TDofOffset = SpaceDim;
    static constexpr int MDofOffset = 0;

    using Plato::SimplexThermomechanics<SpaceDim>::m_numVoigtTerms;
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numDofsPerCell;

    using Plato::AbstractVectorFunction<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunction<EvaluationType>::m_dataMap;
    using Plato::AbstractVectorFunction<EvaluationType>::mMeshSets;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType m_indicatorFunction;
    Plato::ApplyWeighting<SpaceDim, SpaceDim,        IndicatorFunctionType> m_applyFluxWeighting;
    Plato::ApplyWeighting<SpaceDim, m_numVoigtTerms, IndicatorFunctionType> m_applyStressWeighting;

    std::shared_ptr<Plato::BodyLoads<SpaceDim,m_numDofsPerNode>> m_bodyLoads;

    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NMechDims, m_numDofsPerNode, MDofOffset>> m_boundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NThrmDims, m_numDofsPerNode, TDofOffset>> m_boundaryFluxes;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> m_materialModel;

    std::vector<std::string> m_plottable;

public:
    /**************************************************************************/
    ThermoelastostaticResidual(Omega_h::Mesh& aMesh,
                               Omega_h::MeshSets& aMeshSets,
                               Plato::DataMap& aDataMap,
                               Teuchos::ParameterList& aProblemParams,
                               Teuchos::ParameterList& aPenaltyParams) :
            Plato::AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            m_indicatorFunction(aPenaltyParams),
            m_applyStressWeighting(m_indicatorFunction),
            m_applyFluxWeighting(m_indicatorFunction),
            m_bodyLoads(nullptr),
            m_boundaryLoads(nullptr),
            m_boundaryFluxes(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        // create material model and get stiffness
        //
        Plato::ThermoelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
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
        if(aProblemParams.isSublist("Thermal Natural Boundary Conditions"))
        {
            m_boundaryFluxes = std::make_shared<Plato::NaturalBCs<SpaceDim, NThrmDims, m_numDofsPerNode, TDofOffset>>(aProblemParams.sublist("Thermal Natural Boundary Conditions"));
        }
  
        auto tResidualParams = aProblemParams.sublist("Thermoelastostatics");
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
      typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      TMKinematics<SpaceDim>                  kinematics;
      TMKinetics<SpaceDim>                    kinetics(m_materialModel);
      
      StressDivergence<SpaceDim, m_numDofsPerNode, MDofOffset> stressDivergence;
      FluxDivergence  <SpaceDim, m_numDofsPerNode, TDofOffset> fluxDivergence;

      Plato::InterpolateFromNodal<SpaceDim, m_numDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",tNumCells);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient", tNumCells, m_numNodesPerCell, SpaceDim);

      Plato::ScalarMultiVectorT<GradScalarType> strain("strain", tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> tgrad("tgrad", tNumCells, SpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> flux ("flux" , tNumCells, SpaceDim);
    
      Plato::ScalarVectorT<StateScalarType> temperature("Gauss point temperature", tNumCells);

      auto quadratureWeight = mCubatureRule->getCubWeight();
      auto basisFunctions = mCubatureRule->getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, config, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;

        // compute strain and electric field
        //
        kinematics(cellOrdinal, strain, tgrad, state, gradient);
    
        // compute stress and electric displacement
        //
        interpolateFromNodal(cellOrdinal, basisFunctions, state, temperature);
        kinetics(cellOrdinal, stress, flux, strain, tgrad, temperature);

      }, "Cauchy stress");

      auto& applyStressWeighting = m_applyStressWeighting;
      auto& applyFluxWeighting  = m_applyFluxWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        // apply weighting
        //
        applyStressWeighting(cellOrdinal, stress, control);
        applyFluxWeighting (cellOrdinal, flux,  control);
    
        // compute divergence
        //
        stressDivergence(cellOrdinal, result, stress, gradient, cellVolume);
        fluxDivergence (cellOrdinal, result, flux,  gradient, cellVolume);
      }, "Apply weighting and compute divergence");

      if( m_bodyLoads != nullptr )
      {
          m_bodyLoads->get( mMesh, state, control, result );
      }

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, state, control, result );
      }

      if( m_boundaryFluxes != nullptr )
      {
          m_boundaryFluxes->get( &mMesh, mMeshSets, state, control, result );
      }

      if( std::count(m_plottable.begin(),m_plottable.end(),"strain") ) toMap(m_dataMap, strain, "strain");
      if( std::count(m_plottable.begin(),m_plottable.end(),"tgrad") ) toMap(m_dataMap, strain, "tgrad");
      if( std::count(m_plottable.begin(),m_plottable.end(),"stress") ) toMap(m_dataMap, stress, "stress");
      if( std::count(m_plottable.begin(),m_plottable.end(),"flux" ) ) toMap(m_dataMap, stress, "flux" );

    }
};
// class ThermoelastostaticResidual

} // namespace Plato
#endif
