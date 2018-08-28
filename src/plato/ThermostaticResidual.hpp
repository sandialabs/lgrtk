#ifndef THERMOSTATIC_RESIDUAL_HPP
#define THERMOSTATIC_RESIDUAL_HPP

#include "plato/SimplexThermal.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/SimplexFadTypes.hpp"

#include "LinearThermalMaterial.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyWeighting.hpp"
#include "NaturalBCs.hpp"
#include "SimplexFadTypes.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermostaticResidual : 
  public SimplexThermal<EvaluationType::SpatialDim>,
  public AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using Simplex<SpaceDim>::m_numNodesPerCell;
    using SimplexThermal<SpaceDim>::m_numDofsPerCell;
    using SimplexThermal<SpaceDim>::m_numDofsPerNode;

    using AbstractVectorFunction<EvaluationType>::mMesh;
    using AbstractVectorFunction<EvaluationType>::m_dataMap;
    using AbstractVectorFunction<EvaluationType>::mMeshSets;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;



    Omega_h::Matrix< SpaceDim, SpaceDim> m_cellConductivity;
    
    Plato::Scalar m_quadratureWeight;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> m_applyWeighting;

    std::shared_ptr<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>> m_boundaryLoads;

  public:
    /**************************************************************************/
    ThermostaticResidual(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Plato::DataMap& aDataMap,
      Teuchos::ParameterList& problemParams,
      Teuchos::ParameterList& penaltyParams) :
     AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
     m_indicatorFunction(penaltyParams),
     m_applyWeighting(m_indicatorFunction),
     m_boundaryLoads(nullptr)
    /**************************************************************************/
    {
      lgr::ThermalModelFactory<SpaceDim> mmfactory(problemParams);
      auto materialModel = mmfactory.create();
      m_cellConductivity = materialModel->getConductivityMatrix();

      m_quadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType d=2; d<=SpaceDim; d++)
      { 
        m_quadratureWeight /= Plato::Scalar(d);
      }

      // parse boundary Conditions
      // 
      if(problemParams.isSublist("Natural Boundary Conditions"))
      {
          m_boundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>>(problemParams.sublist("Natural Boundary Conditions"));
      }
    
    }


    /**************************************************************************/
    void
    evaluate( const Plato::ScalarMultiVectorT<StateScalarType  > & state,
              const Plato::ScalarMultiVectorT<ControlScalarType> & control,
              const Plato::ScalarArray3DT<ConfigScalarType > & config,
              Plato::ScalarMultiVectorT<ResultScalarType > & result,
              Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<GradScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tgrad("temperature gradient",numCells,SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>
        gradient("gradient",numCells,m_numNodesPerCell,SpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tflux("thermal flux",numCells,SpaceDim);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim>  computeGradient;

      ScalarGrad<SpaceDim>            scalarGrad;
      ThermalFlux<SpaceDim>           thermalFlux(m_cellConductivity);
      FluxDivergence<SpaceDim>        fluxDivergence;
    
      auto& applyWeighting  = m_applyWeighting;
      auto quadratureWeight = m_quadratureWeight;
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
    
        computeGradient(cellOrdinal, gradient, config, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;
    
        // compute temperature gradient
        //
        scalarGrad(cellOrdinal, tgrad, state, gradient);
    
        // compute flux
        //
        thermalFlux(cellOrdinal, tflux, tgrad);
    
        // apply weighting
        //
        applyWeighting(cellOrdinal, tflux, control);
    
        // compute stress divergence
        //
        fluxDivergence(cellOrdinal, result, tflux, gradient, cellVolume);
        
      },"flux divergence");

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, state, control, result );
      }
    }
};

#endif
