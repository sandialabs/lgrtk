#ifndef HEAT_EQUATION_RESIDUAL_HPP
#define HEAT_EQUATION_RESIDUAL_HPP

#include "plato/SimplexThermal.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/ThermalContent.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/StateValues.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

#include "LinearThermalMaterial.hpp"
#include "AbstractVectorFunctionInc.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyWeighting.hpp"
#include "NaturalBCs.hpp"
#include "SimplexFadTypes.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class HeatEquationResidual : 
  public SimplexThermal<EvaluationType::SpatialDim>,
  public AbstractVectorFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using Simplex<SpaceDim>::m_numNodesPerCell;
    using SimplexThermal<SpaceDim>::m_numDofsPerCell;
    using SimplexThermal<SpaceDim>::m_numDofsPerNode;

    using AbstractVectorFunctionInc<EvaluationType>::mMesh;
    using AbstractVectorFunctionInc<EvaluationType>::m_dataMap;
    using AbstractVectorFunctionInc<EvaluationType>::mMeshSets;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;



    Omega_h::Matrix< SpaceDim, SpaceDim> m_cellConductivity;
    Plato::Scalar m_cellDensity;
    Plato::Scalar m_cellSpecificHeat;
    
    Plato::Scalar m_quadratureWeight;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> m_applyFluxWeighting;
    ApplyWeighting<SpaceDim,m_numDofsPerNode,IndicatorFunctionType> m_applyMassWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> m_cubatureRule;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>> m_boundaryLoads;

  public:
    /**************************************************************************/
    HeatEquationResidual(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Plato::DataMap& aDataMap,
      Teuchos::ParameterList& problemParams,
      Teuchos::ParameterList& penaltyParams) :
     AbstractVectorFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap),
     m_indicatorFunction(penaltyParams),
     m_applyFluxWeighting(m_indicatorFunction),
     m_applyMassWeighting(m_indicatorFunction),
     m_cubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
     m_boundaryLoads(nullptr)
    /**************************************************************************/
    {
      lgr::ThermalModelFactory<SpaceDim> mmfactory(problemParams);
      auto materialModel = mmfactory.create();
      m_cellConductivity = materialModel->getConductivityMatrix();
      m_cellDensity      = materialModel->getMassDensity();
      m_cellSpecificHeat = materialModel->getSpecificHeat();

      m_quadratureWeight = m_cubatureRule->getCubWeight();

      // parse boundary Conditions
      // 
      if(problemParams.isSublist("Natural Boundary Conditions"))
      {
          m_boundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>>(problemParams.sublist("Natural Boundary Conditions"));
      }
    
    }


    /**************************************************************************/
    void
    evaluate( const Plato::ScalarMultiVectorT< StateScalarType     > & aState,
              const Plato::ScalarMultiVectorT< PrevStateScalarType > & aPrevState,
              const Plato::ScalarMultiVectorT< ControlScalarType   > & aControl,
              const Plato::ScalarArray3DT    < ConfigScalarType    > & aConfig,
                    Plato::ScalarMultiVectorT< ResultScalarType    > & aResult,
                    Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      Kokkos::deep_copy(aResult, 0.0);

      auto numCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      using PrevGradScalarType =
        typename Plato::fad_type_t<SimplexThermal<EvaluationType::SpatialDim>, PrevStateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Plato::ScalarMultiVectorT<GradScalarType> tGrad("temperature gradient at step k",numCells,SpaceDim);
      Plato::ScalarMultiVectorT<PrevGradScalarType> tPrevGrad("temperature gradient at step k-1",numCells,SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient",numCells,m_numNodesPerCell,SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> tFlux("thermal flux at step k",numCells,SpaceDim);
      Plato::ScalarMultiVectorT<ResultScalarType> tPrevFlux("thermal flux at step k-1",numCells,SpaceDim);

      Plato::ScalarMultiVectorT<StateScalarType> tStateValues("Gauss point temperature at step k", numCells, m_numDofsPerNode);
      Plato::ScalarMultiVectorT<PrevStateScalarType> tPrevStateValues("Gauss point temperature at step k-1", numCells, m_numDofsPerNode);

      Plato::ScalarMultiVectorT<ResultScalarType> tThermalContent("Gauss point heat content at step k", numCells, m_numDofsPerNode);
      Plato::ScalarMultiVectorT<ResultScalarType> tPrevThermalContent("Gauss point heat content at step k-1", numCells, m_numDofsPerNode);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim>  computeGradient;

      ScalarGrad<SpaceDim>            scalarGrad;
      ThermalFlux<SpaceDim>           thermalFlux(m_cellConductivity);
      FluxDivergence<SpaceDim>        fluxDivergence;

      Plato::StateValues computeStateValues;
      ThermalContent thermalContent(m_cellDensity, m_cellSpecificHeat);
      Plato::ComputeProjectionWorkset projectThermalContent;
      
      auto basisFunctions = m_cubatureRule->getBasisFunctions();
    
      auto& applyFluxWeighting  = m_applyFluxWeighting;
      auto& applyMassWeighting  = m_applyMassWeighting;
      auto quadratureWeight = m_quadratureWeight;
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
    
        computeGradient(cellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;
    
        // compute temperature gradient
        //
        scalarGrad(cellOrdinal, tGrad, aState, gradient);
        scalarGrad(cellOrdinal, tPrevGrad, aPrevState, gradient);
    
        // compute flux
        //
        thermalFlux(cellOrdinal, tFlux, tGrad);
        thermalFlux(cellOrdinal, tPrevFlux, tPrevGrad);
    
        // apply weighting
        //
        applyFluxWeighting(cellOrdinal, tFlux, aControl);
        applyFluxWeighting(cellOrdinal, tPrevFlux, aControl);

        // compute stress divergence
        //
        fluxDivergence(cellOrdinal, aResult, tFlux,     gradient, cellVolume, aTimeStep/2.0);
        fluxDivergence(cellOrdinal, aResult, tPrevFlux, gradient, cellVolume, aTimeStep/2.0);


        // add capacitance terms
        
        // compute temperature at gausspoints
        //
        computeStateValues(cellOrdinal, basisFunctions, aState, tStateValues);
        computeStateValues(cellOrdinal, basisFunctions, aPrevState, tPrevStateValues);

        // compute the specific heat content (i.e., specific heat times temperature)
        //
        thermalContent(cellOrdinal, tThermalContent, tStateValues);
        thermalContent(cellOrdinal, tPrevThermalContent, tPrevStateValues);

        // apply weighting
        //
        applyMassWeighting(cellOrdinal, tThermalContent, aControl);
        applyMassWeighting(cellOrdinal, tPrevThermalContent, aControl);

        // project to nodes
        //
        projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent, aResult);
        projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tPrevThermalContent, aResult, -1.0);

      },"flux divergence");

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, aState, aControl, aResult, -aTimeStep/2.0 );
          m_boundaryLoads->get( &mMesh, mMeshSets, aPrevState, aControl, aResult, -aTimeStep/2.0 );
      }
    }
};

#endif
