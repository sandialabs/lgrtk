#ifndef INTERNAL_THERMOELASTIC_ENERGY_HPP
#define INTERNAL_THERMOELASTIC_ENERGY_HPP

#include "plato/SimplexFadTypes.hpp"
#include "plato/SimplexThermomechanics.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/TMKinematics.hpp"
#include "plato/TMKinetics.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/LinearThermoelasticMaterial.hpp"
#include "plato/ToMap.hpp"
#include "plato/ExpInstMacros.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute internal thermo-elastic energy criterion, given by
 *                  /f$ f(z) = u^{T}K_u(z)u + T^{T}K_t(z)T /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermoelasticEnergy : 
  public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
    static constexpr Plato::OrdinalType TDofOffset = mSpaceDim;
    
    using Plato::SimplexThermomechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<mSpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermomechanics<mSpaceDim>::mNumDofsPerNode;

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::LinearThermoelasticMaterial<mSpaceDim>> mMaterialModel;
    
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,        IndicatorFunctionType> mApplyFluxWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    InternalThermoelasticEnergy(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams,
                          std::string& aFunctionName ) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mIndicatorFunction(aPenaltyParams),
            mApplyStressWeighting(mIndicatorFunction),
            mApplyFluxWeighting(mIndicatorFunction),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
      Teuchos::ParameterList tProblemParams(aProblemParams);

      auto& tParams = aProblemParams.sublist(aFunctionName);
      if( tParams.get<bool>("Include Thermal Strain", true) == false )
      {
         tProblemParams.sublist("Material Model").sublist("Cubic Linear Thermoelastic").set("a11",0.0);
         tProblemParams.sublist("Material Model").sublist("Cubic Linear Thermoelastic").set("a22",0.0);
         tProblemParams.sublist("Material Model").sublist("Cubic Linear Thermoelastic").set("a33",0.0);
      }
      Plato::ThermoelasticModelFactory<mSpaceDim> mmfactory(tProblemParams);
      mMaterialModel = mmfactory.create();

      if( tProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = tProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::TMKinematics<mSpaceDim>                  kinematics;
      Plato::TMKinetics<mSpaceDim>                    kinetics(mMaterialModel);

      Plato::ScalarProduct<mNumVoigtTerms>          mechanicalScalarProduct;
      Plato::ScalarProduct<mSpaceDim>                 thermalScalarProduct;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType>   strain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType>   tgrad ("tgrad",  tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> flux  ("flux",   tNumCells, mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>   gradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      Plato::ScalarVectorT<StateScalarType> temperature("Gauss point temperature", tNumCells);

      auto quadratureWeight = mCubatureRule->getCubWeight();
      auto basisFunctions   = mCubatureRule->getBasisFunctions();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyFluxWeighting   = mApplyFluxWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute strain and temperature gradient
        //
        kinematics(aCellOrdinal, strain, tgrad, aState, gradient);

        // compute stress and thermal flux
        //
        interpolateFromNodal(aCellOrdinal, basisFunctions, aState, temperature);
        kinetics(aCellOrdinal, stress, flux, strain, tgrad, temperature);

        // apply weighting
        //
        applyStressWeighting(aCellOrdinal, stress, aControl);
        applyFluxWeighting  (aCellOrdinal, flux,   aControl);

        // compute element internal energy (inner product of strain and weighted stress)
        //
        mechanicalScalarProduct(aCellOrdinal, aResult, stress, strain, cellVolume);
        thermalScalarProduct   (aCellOrdinal, aResult, flux,   tgrad,  cellVolume);

      },"energy gradient");
    }
};

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermoelasticEnergyInc :
  public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
  public AbstractScalarFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    static constexpr int TDofOffset = SpaceDim;

    using Plato::SimplexThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerNode;

    using AbstractScalarFunctionInc<EvaluationType>::mMesh;
    using AbstractScalarFunctionInc<EvaluationType>::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> mMaterialModel;

    Plato::Scalar mQuadratureWeight;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<SpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    ApplyWeighting<SpaceDim, SpaceDim,        IndicatorFunctionType> mApplyFluxWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    InternalThermoelasticEnergyInc(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams,
                          std::string& aFunctionName ) :
            AbstractScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mIndicatorFunction(aPenaltyParams),
            mApplyStressWeighting(mIndicatorFunction),
            mApplyFluxWeighting(mIndicatorFunction),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
      Plato::ThermoelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create();

      if( aProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = aProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<PrevStateScalarType> & aPrevState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      TMKinematics<SpaceDim>                  kinematics;
      TMKinetics<SpaceDim>                    kinetics(mMaterialModel);

      ScalarProduct<mNumVoigtTerms>          mechanicalScalarProduct;
      ScalarProduct<SpaceDim>                 thermalScalarProduct;

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",numCells);

      Plato::ScalarMultiVectorT<GradScalarType>   strain("strain", numCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType>   tgrad ("tgrad",  numCells, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", numCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> flux  ("flux",   numCells, SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>   gradient("gradient",numCells,mNumNodesPerCell,SpaceDim);

      Plato::ScalarVectorT<StateScalarType> temperature("Gauss point temperature", numCells);

      auto quadratureWeight = mCubatureRule->getCubWeight();
      auto basisFunctions   = mCubatureRule->getBasisFunctions();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyFluxWeighting   = mApplyFluxWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute strain and temperature gradient
        //
        kinematics(aCellOrdinal, strain, tgrad, aState, gradient);

        // compute stress and thermal flux
        //
        interpolateFromNodal(aCellOrdinal, basisFunctions, aState, temperature);
        kinetics(aCellOrdinal, stress, flux, strain, tgrad, temperature);

        // apply weighting
        //
        applyStressWeighting(aCellOrdinal, stress, aControl);
        applyFluxWeighting  (aCellOrdinal, flux,   aControl);

        // compute element internal energy (inner product of strain and weighted stress)
        //
        mechanicalScalarProduct(aCellOrdinal, aResult, stress, strain, cellVolume);
        thermalScalarProduct   (aCellOrdinal, aResult, flux,   tgrad,  cellVolume);

      },"energy gradient");
    }
};
// class InternalThermoelasticEnergy

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 1)
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 2)
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 3)
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 3)
#endif

#endif
