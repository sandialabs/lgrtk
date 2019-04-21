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
  public AbstractVectorFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    static constexpr int NThrmDims = 1;
    static constexpr int NMechDims = SpaceDim;

    static constexpr int TDofOffset = SpaceDim;
    static constexpr int MDofOffset = 0;

    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numDofsPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numDofsPerNode;

    using AbstractVectorFunctionInc<EvaluationType>::mMesh;
    using AbstractVectorFunctionInc<EvaluationType>::m_dataMap;
    using AbstractVectorFunctionInc<EvaluationType>::mMeshSets;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    Plato::Scalar m_cellDensity;
    Plato::Scalar m_cellSpecificHeat;
    
    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim, m_numVoigtTerms,  IndicatorFunctionType> m_applyStressWeighting;
    ApplyWeighting<SpaceDim, SpaceDim,         IndicatorFunctionType> m_applyFluxWeighting;
    ApplyWeighting<SpaceDim, NThrmDims,        IndicatorFunctionType> m_applyMassWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> m_cubatureRule;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NMechDims, m_numDofsPerNode, MDofOffset>> m_boundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NThrmDims, m_numDofsPerNode, TDofOffset>> m_boundaryFluxes;

    Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> m_materialModel;

  public:
    /**************************************************************************/
    TransientThermomechResidual(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Plato::DataMap& aDataMap,
      Teuchos::ParameterList& aProblemParams,
      Teuchos::ParameterList& aPenaltyParams) :
     AbstractVectorFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap),
     m_indicatorFunction(aPenaltyParams),
     m_applyStressWeighting(m_indicatorFunction),
     m_applyFluxWeighting(m_indicatorFunction),
     m_applyMassWeighting(m_indicatorFunction),
     m_cubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
     m_boundaryLoads(nullptr),
     m_boundaryFluxes(nullptr)
    /**************************************************************************/
    {
      Plato::ThermoelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
      m_materialModel = mmfactory.create();
      m_cellDensity      = m_materialModel->getMassDensity();
      m_cellSpecificHeat = m_materialModel->getSpecificHeat();


      // parse boundary Conditions
      // 
      if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
      {
          m_boundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim, NMechDims, m_numDofsPerNode, MDofOffset>>(aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
      }
      // parse thermal boundary Conditions
      // 
      if(aProblemParams.isSublist("Thermal Natural Boundary Conditions"))
      {
          m_boundaryFluxes = std::make_shared<Plato::NaturalBCs<SpaceDim, NThrmDims, m_numDofsPerNode, TDofOffset>>(aProblemParams.sublist("Thermal Natural Boundary Conditions"));
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

      Plato::ScalarMultiVectorT<GradScalarType>     tStrain     ("strain at step k",   tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<PrevGradScalarType> tPrevStrain ("strain at step k-1", tNumCells, m_numVoigtTerms);

      Plato::ScalarMultiVectorT<GradScalarType>     tGrad     ("temperature gradient at step k",   tNumCells, SpaceDim);
      Plato::ScalarMultiVectorT<PrevGradScalarType> tPrevGrad ("temperature gradient at step k-1", tNumCells, SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient",tNumCells,m_numNodesPerCell,SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> tStress    ("stress at step k",   tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tPrevStress("stress at step k-1", tNumCells, m_numVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType> tFlux    ("thermal flux at step k",   tNumCells, SpaceDim);
      Plato::ScalarMultiVectorT<ResultScalarType> tPrevFlux("thermal flux at step k-1", tNumCells, SpaceDim);

      Plato::ScalarVectorT<ResultScalarType> tThermalContent    ("Gauss point heat content at step k",   tNumCells);
      Plato::ScalarVectorT<ResultScalarType> tPrevThermalContent("Gauss point heat content at step k-1", tNumCells);

      Plato::ScalarVectorT<StateScalarType>     tTemperature     ("Gauss point temperature at step k",   tNumCells);
      Plato::ScalarVectorT<PrevStateScalarType> tPrevTemperature ("Gauss point temperature at step k-1", tNumCells);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim>  computeGradient;
      TMKinematics<SpaceDim>                   kinematics;
      TMKinetics<SpaceDim>                     kinetics(m_materialModel);

      Plato::InterpolateFromNodal<SpaceDim, m_numDofsPerNode, TDofOffset> interpolateFromNodal;

      FluxDivergence  <SpaceDim, m_numDofsPerNode, TDofOffset> fluxDivergence;
      StressDivergence<SpaceDim, m_numDofsPerNode, MDofOffset> stressDivergence;

      ThermalContent thermalContent(m_cellDensity, m_cellSpecificHeat);
      ProjectToNode<SpaceDim, m_numDofsPerNode, TDofOffset> projectThermalContent;
      
      auto basisFunctions = m_cubatureRule->getBasisFunctions();
    
      auto& applyStressWeighting = m_applyStressWeighting;
      auto& applyFluxWeighting   = m_applyFluxWeighting;
      auto& applyMassWeighting   = m_applyMassWeighting;
      auto quadratureWeight = m_cubatureRule->getCubWeight();
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
        stressDivergence(cellOrdinal, aResult, tPrevStress, gradient, cellVolume, aTimeStep/2.0);

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

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, aState,     aControl, aResult, -aTimeStep/2.0 );
          m_boundaryLoads->get( &mMesh, mMeshSets, aPrevState, aControl, aResult, -aTimeStep/2.0 );
      }
      if( m_boundaryFluxes != nullptr )
      {
          m_boundaryFluxes->get( &mMesh, mMeshSets, aState,     aControl, aResult, -aTimeStep/2.0 );
          m_boundaryFluxes->get( &mMesh, mMeshSets, aPrevState, aControl, aResult, -aTimeStep/2.0 );
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
