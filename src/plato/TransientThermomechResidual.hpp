#ifndef TRANSIENT_THERMOMECH_RESIDUAL_HPP
#define TRANSIENT_THERMOMECH_RESIDUAL_HPP

#include "plato/SimplexThermomechanics.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/ThermalContent.hpp"
#include "plato/StressDivergence.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/TMKinematics.hpp"
#include "plato/TMKinetics.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/StateValues.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

#include "plato/LinearThermoelasticMaterial.hpp"
#include "plato/AbstractVectorFunctionInc.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/ProjectToNode.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/NaturalBCs.hpp"

#include "plato/ExpInstMacros.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TransientThermomechResidual : 
  public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
  public Plato::AbstractVectorFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    static constexpr int NThrmDims = 1;
    static constexpr int NMechDims = SpaceDim;

    static constexpr int TDofOffset = SpaceDim;
    static constexpr int MDofOffset = 0;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerNode;

    using Plato::AbstractVectorFunctionInc<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunctionInc<EvaluationType>::mDataMap;
    using Plato::AbstractVectorFunctionInc<EvaluationType>::mMeshSets;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    Plato::Scalar mCellDensity;
    Plato::Scalar mCellSpecificHeat;
    
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<SpaceDim, SpaceDim,         IndicatorFunctionType> mApplyFluxWeighting;
    Plato::ApplyWeighting<SpaceDim, NThrmDims,        IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NThrmDims, mNumDofsPerNode, TDofOffset>> mBoundaryFluxes;

    Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> mMaterialModel;

  public:
    /**************************************************************************/
    TransientThermomechResidual(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Plato::DataMap& aDataMap,
      Teuchos::ParameterList& aProblemParams,
      Teuchos::ParameterList& aPenaltyParams) :
     Plato::AbstractVectorFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap,
        {"Displacement X", "Displacement Y", "Displacement Z", "Temperature"}),
     mIndicatorFunction(aPenaltyParams),
     mApplyStressWeighting(mIndicatorFunction),
     mApplyFluxWeighting(mIndicatorFunction),
     mApplyMassWeighting(mIndicatorFunction),
     mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
     mBoundaryLoads(nullptr),
     mBoundaryFluxes(nullptr)
    /**************************************************************************/
    {
      Plato::ThermoelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create();
      mCellDensity      = mMaterialModel->getMassDensity();
      mCellSpecificHeat = mMaterialModel->getSpecificHeat();


      // parse boundary Conditions
      // 
      if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
      {
          mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>>(aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
      }
      // parse thermal boundary Conditions
      // 
      if(aProblemParams.isSublist("Thermal Natural Boundary Conditions"))
      {
          mBoundaryFluxes = std::make_shared<Plato::NaturalBCs<SpaceDim, NThrmDims, mNumDofsPerNode, TDofOffset>>(aProblemParams.sublist("Thermal Natural Boundary Conditions"));
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
      auto tNumCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      using PrevGradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, PrevStateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType>     tStrain     ("strain at step k",   tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<PrevGradScalarType> tPrevStrain ("strain at step k-1", tNumCells, mNumVoigtTerms);

      Plato::ScalarMultiVectorT<GradScalarType>     tGrad     ("temperature gradient at step k",   tNumCells, SpaceDim);
      Plato::ScalarMultiVectorT<PrevGradScalarType> tPrevGrad ("temperature gradient at step k-1", tNumCells, SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient",tNumCells,mNumNodesPerCell,SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> tStress    ("stress at step k",   tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tPrevStress("stress at step k-1", tNumCells, mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType> tFlux    ("thermal flux at step k",   tNumCells, SpaceDim);
      Plato::ScalarMultiVectorT<ResultScalarType> tPrevFlux("thermal flux at step k-1", tNumCells, SpaceDim);

      Plato::ScalarVectorT<ResultScalarType> tThermalContent    ("Gauss point heat content at step k",   tNumCells);
      Plato::ScalarVectorT<ResultScalarType> tPrevThermalContent("Gauss point heat content at step k-1", tNumCells);

      Plato::ScalarVectorT<StateScalarType>     tTemperature     ("Gauss point temperature at step k",   tNumCells);
      Plato::ScalarVectorT<PrevStateScalarType> tPrevTemperature ("Gauss point temperature at step k-1", tNumCells);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim>  computeGradient;
      Plato::TMKinematics<SpaceDim>                   kinematics;
      Plato::TMKinetics<SpaceDim>                     kinetics(mMaterialModel);

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::FluxDivergence  <SpaceDim, mNumDofsPerNode, TDofOffset> fluxDivergence;
      Plato::StressDivergence<SpaceDim, mNumDofsPerNode, MDofOffset> stressDivergence;

      Plato::ThermalContent thermalContent(mCellDensity, mCellSpecificHeat);
      Plato::ProjectToNode<SpaceDim, mNumDofsPerNode, TDofOffset> projectThermalContent;
      
      auto basisFunctions = mCubatureRule->getBasisFunctions();
    
      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyFluxWeighting   = mApplyFluxWeighting;
      auto& applyMassWeighting   = mApplyMassWeighting;
      auto quadratureWeight = mCubatureRule->getCubWeight();
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
    
        computeGradient(cellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;
    
        // compute strain and temperature gradient
        //
        kinematics(cellOrdinal, tStrain,     tGrad,     aState,     gradient);
        kinematics(cellOrdinal, tPrevStrain, tPrevGrad, aPrevState, gradient);

        // compute stress and thermal flux
        //
        interpolateFromNodal(cellOrdinal, basisFunctions, aState,     tTemperature);
        interpolateFromNodal(cellOrdinal, basisFunctions, aPrevState, tPrevTemperature);

        kinetics(cellOrdinal, tStress,     tFlux,     tStrain,     tGrad,     tTemperature);
        kinetics(cellOrdinal, tPrevStress, tPrevFlux, tPrevStrain, tPrevGrad, tPrevTemperature);

        // apply weighting
        //
        applyStressWeighting(cellOrdinal, tStress,     aControl);
        applyStressWeighting(cellOrdinal, tPrevStress, aControl);

        applyFluxWeighting(cellOrdinal, tFlux,     aControl);
        applyFluxWeighting(cellOrdinal, tPrevFlux, aControl);

        // compute stress and flux divergence
        //
        stressDivergence(cellOrdinal, aResult, tStress,     gradient, cellVolume, aTimeStep/2.0);

        fluxDivergence(cellOrdinal, aResult, tFlux,     gradient, cellVolume, aTimeStep/2.0);
        fluxDivergence(cellOrdinal, aResult, tPrevFlux, gradient, cellVolume, aTimeStep/2.0);

        // add capacitance terms
        
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
        projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent,     aResult);
        projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tPrevThermalContent, aResult, -1.0);

      },"stress and flux divergence");

      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( &mMesh, mMeshSets, aState,     aControl, aConfig, aResult, -aTimeStep/2.0 );
          mBoundaryLoads->get( &mMesh, mMeshSets, aPrevState, aControl, aConfig, aResult, -aTimeStep/2.0 );
      }
      if( mBoundaryFluxes != nullptr )
      {
          mBoundaryFluxes->get( &mMesh, mMeshSets, aState,     aControl, aConfig, aResult, -aTimeStep/2.0 );
          mBoundaryFluxes->get( &mMesh, mMeshSets, aPrevState, aControl, aConfig, aResult, -aTimeStep/2.0 );
      }
    }
};

#ifdef PLATO_1D
PLATO_EXPL_DEC_INC(TransientThermomechResidual, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC_INC(TransientThermomechResidual, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC_INC(TransientThermomechResidual, Plato::SimplexThermomechanics, 3)
#endif

#endif
