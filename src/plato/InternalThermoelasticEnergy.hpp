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
    
    using Plato::SimplexThermomechanics<mSpaceDim>::m_numVoigtTerms;
    using Simplex<mSpaceDim>::m_numNodesPerCell;
    using Plato::SimplexThermomechanics<mSpaceDim>::m_numDofsPerCell;
    using Plato::SimplexThermomechanics<mSpaceDim>::m_numDofsPerNode;

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::LinearThermoelasticMaterial<mSpaceDim>> m_materialModel;
    
    IndicatorFunctionType m_indicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, m_numVoigtTerms, IndicatorFunctionType> m_applyStressWeighting;
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,        IndicatorFunctionType> m_applyFluxWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> m_CubatureRule;

    std::vector<std::string> m_plottable;

  public:
    /**************************************************************************/
    InternalThermoelasticEnergy(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams ) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Internal Thermoelastic Energy"),
            m_indicatorFunction(aPenaltyParams),
            m_applyStressWeighting(m_indicatorFunction),
            m_applyFluxWeighting(m_indicatorFunction),
            m_CubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
      Plato::ThermoelasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      m_materialModel = mmfactory.create();

      if( aProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        m_plottable = aProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
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
      Plato::TMKinetics<mSpaceDim>                    kinetics(m_materialModel);

      Plato::ScalarProduct<m_numVoigtTerms>          mechanicalScalarProduct;
      Plato::ScalarProduct<mSpaceDim>                 thermalScalarProduct;

      Plato::InterpolateFromNodal<mSpaceDim, m_numDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType>   strain("strain", tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType>   tgrad ("tgrad",  tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> flux  ("flux",   tNumCells, mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>   gradient("gradient",tNumCells,m_numNodesPerCell,mSpaceDim);

      Plato::ScalarVectorT<StateScalarType> temperature("Gauss point temperature", tNumCells);

      auto quadratureWeight = m_CubatureRule->getCubWeight();
      auto basisFunctions   = m_CubatureRule->getBasisFunctions();

      auto& applyStressWeighting = m_applyStressWeighting;
      auto& applyFluxWeighting   = m_applyFluxWeighting;
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
// class InternalThermoelasticEnergy

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 3)
#endif

#endif
