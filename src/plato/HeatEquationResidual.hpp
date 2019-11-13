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
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

#include "plato/LinearThermalMaterial.hpp"
#include "plato/AbstractVectorFunctionInc.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/ProjectToNode.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/SimplexFadTypes.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class HeatEquationResidual : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractVectorFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerNode;

    using Plato::AbstractVectorFunctionInc<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunctionInc<EvaluationType>::mDataMap;
    using Plato::AbstractVectorFunctionInc<EvaluationType>::mMeshSets;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;



    Omega_h::Matrix< SpaceDim, SpaceDim> mCellConductivity;
    Plato::Scalar mCellDensity;
    Plato::Scalar mCellSpecificHeat;
    
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> mApplyFluxWeighting;
    Plato::ApplyWeighting<SpaceDim,mNumDofsPerNode,IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>> mBoundaryLoads;

  public:
    /**************************************************************************/
    HeatEquationResidual(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Plato::DataMap& aDataMap,
      Teuchos::ParameterList& problemParams,
      Teuchos::ParameterList& penaltyParams) :
     AbstractVectorFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, {"Temperature"}),
     mIndicatorFunction(penaltyParams),
     mApplyFluxWeighting(mIndicatorFunction),
     mApplyMassWeighting(mIndicatorFunction),
     mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
     mBoundaryLoads(nullptr)
    /**************************************************************************/
    {
      Plato::ThermalModelFactory<SpaceDim> mmfactory(problemParams);
      auto materialModel = mmfactory.create();
      mCellConductivity = materialModel->getConductivityMatrix();
      mCellDensity      = materialModel->getMassDensity();
      mCellSpecificHeat = materialModel->getSpecificHeat();


      // parse boundary Conditions
      // 
      if(problemParams.isSublist("Natural Boundary Conditions"))
      {
          mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>>(problemParams.sublist("Natural Boundary Conditions"));
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
      auto numCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      using PrevGradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, PrevStateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Plato::ScalarMultiVectorT<GradScalarType> tGrad("temperature gradient at step k",numCells,SpaceDim);
      Plato::ScalarMultiVectorT<PrevGradScalarType> tPrevGrad("temperature gradient at step k-1",numCells,SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient",numCells,mNumNodesPerCell,SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> tFlux("thermal flux at step k",numCells,SpaceDim);
      Plato::ScalarMultiVectorT<ResultScalarType> tPrevFlux("thermal flux at step k-1",numCells,SpaceDim);

      Plato::ScalarVectorT<StateScalarType> tTemperature("Gauss point temperature at step k", numCells);
      Plato::ScalarVectorT<PrevStateScalarType> tPrevTemperature("Gauss point temperature at step k-1", numCells);

      Plato::ScalarVectorT<ResultScalarType> tThermalContent("Gauss point heat content at step k", numCells);
      Plato::ScalarVectorT<ResultScalarType> tPrevThermalContent("Gauss point heat content at step k-1", numCells);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim>  computeGradient;

      Plato::ScalarGrad<SpaceDim>            scalarGrad;
      Plato::ThermalFlux<SpaceDim>           thermalFlux(mCellConductivity);
      Plato::FluxDivergence<SpaceDim>        fluxDivergence;

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode> interpolateFromNodal;
      Plato::ThermalContent thermalContent(mCellDensity, mCellSpecificHeat);
      Plato::ProjectToNode<SpaceDim, mNumDofsPerNode> projectThermalContent;
      
      auto basisFunctions = mCubatureRule->getBasisFunctions();
    
      auto& applyFluxWeighting  = mApplyFluxWeighting;
      auto& applyMassWeighting  = mApplyMassWeighting;
      auto quadratureWeight = mCubatureRule->getCubWeight();
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
        interpolateFromNodal(cellOrdinal, basisFunctions, aState, tTemperature);
        interpolateFromNodal(cellOrdinal, basisFunctions, aPrevState, tPrevTemperature);

        // compute the specific heat content (i.e., specific heat times temperature)
        //
        thermalContent(cellOrdinal, tThermalContent, tTemperature);
        thermalContent(cellOrdinal, tPrevThermalContent, tPrevTemperature);

        // apply weighting
        //
        applyMassWeighting(cellOrdinal, tThermalContent, aControl);
        applyMassWeighting(cellOrdinal, tPrevThermalContent, aControl);

        // project to nodes
        //
        projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent, aResult);
        projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tPrevThermalContent, aResult, -1.0);

      },"flux divergence");

      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( &mMesh, mMeshSets, aState, aControl, aConfig, aResult, -aTimeStep/2.0 );
          mBoundaryLoads->get( &mMesh, mMeshSets, aPrevState, aControl, aConfig, aResult, -aTimeStep/2.0 );
      }
    }
};
// class HeatEquationResidual

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 3)
#endif

#endif
