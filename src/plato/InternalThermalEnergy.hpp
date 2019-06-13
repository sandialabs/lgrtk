#ifndef INTERNAL_THERMAL_ENERGY_HPP
#define INTERNAL_THERMAL_ENERGY_HPP

#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/LinearThermalMaterial.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermalEnergy : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using Plato::Simplex<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell */
    using Plato::SimplexThermal<mSpaceDim>::m_numDofsPerCell; /*!< number of degrees of freedom per cell */

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh; /*!< mesh database */
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap; /*!< PLATO Analyze database */

    using StateScalarType   = typename EvaluationType::StateScalarType; /*!< automatic differentiation type for states */
    using ControlScalarType = typename EvaluationType::ControlScalarType; /*!< automatic differentiation type for controls */
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType; /*!< automatic differentiation type for configuration */
    using ResultScalarType  = typename EvaluationType::ResultScalarType; /*!< automatic differentiation type for results */

    Omega_h::Matrix< mSpaceDim, mSpaceDim> m_cellConductivity; /*!< conductivity coefficients */
    
    Plato::Scalar m_quadratureWeight; /*!< integration rule weight */

    IndicatorFunctionType m_indicatorFunction; /*!< penalty function */
    Plato::ApplyWeighting<mSpaceDim,mSpaceDim,IndicatorFunctionType> m_applyWeighting; /*!< applies penalty function */

  public:
    /******************************************************************************//**
     * @brief Constructor
     * @param aMesh volume mesh database
     * @param aMeshSets surface mesh database
     * @param aProblemParams input database for overall problem
     * @param aPenaltyParams input database for penalty function
    **********************************************************************************/
    InternalThermalEnergy(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams,
                          std::string& aFunctionName) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction)
    {
      Plato::ThermalModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
      auto tMaterialModel = tMaterialModelFactory.create();
      m_cellConductivity = tMaterialModel->getConductivityMatrix();

      m_quadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDim=2; tDim<=mSpaceDim; tDim++)
      { 
        m_quadratureWeight /= Plato::Scalar(tDim);
      }
    
    }

    /******************************************************************************//**
     * @brief Evaluate internal elastic energy function
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    {
      auto tNumCells = mMesh.nelems();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::ScalarGrad<mSpaceDim>                    tComputeScalarGrad;
      Plato::ScalarProduct<mSpaceDim>                 tComputeScalarProduct;
      Plato::ThermalFlux<mSpaceDim>                   tComputeThermalFlux(m_cellConductivity);

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Kokkos::View<GradScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tThermalGrad("temperature gradient",tNumCells,mSpaceDim);

      Kokkos::View<ConfigScalarType***, Kokkos::LayoutRight, Plato::MemSpace>
        tGradient("gradient",tNumCells,m_numNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tThermalFlux("thermal flux",tNumCells,mSpaceDim);

      auto tQuadratureWeight = m_quadratureWeight;
      auto tApplyWeighting  = m_applyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute temperature gradient
        //
        tComputeScalarGrad(aCellOrdinal, tThermalGrad, aState, tGradient);

        // compute flux
        //
        tComputeThermalFlux(aCellOrdinal, tThermalFlux, tThermalGrad);

        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, tThermalFlux, aControl);
    
        // compute element internal energy (inner product of tgrad and weighted tflux)
        //
        tComputeScalarProduct(aCellOrdinal, aResult, tThermalFlux, tThermalGrad, tCellVolume);

      },"energy gradient");
    }
};
// class InternalThermalEnergy

} // namespace Plato

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermalEnergyInc : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<mSpaceDim>::m_numNodesPerCell;
    using Plato::SimplexThermal<mSpaceDim>::m_numDofsPerCell;

    using Plato::AbstractScalarFunctionInc<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunctionInc<EvaluationType>::m_dataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix< mSpaceDim, mSpaceDim> m_cellConductivity;
    
    Plato::Scalar m_quadratureWeight;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<mSpaceDim,mSpaceDim,IndicatorFunctionType> m_applyWeighting;

  public:
    /**************************************************************************/
    InternalThermalEnergyInc(Omega_h::Mesh& aMesh,
                             Omega_h::MeshSets& aMeshSets,
                             Plato::DataMap& aDataMap,
                             Teuchos::ParameterList& aProblemParams,
                             Teuchos::ParameterList& aPenaltyParams,
                             std::string& aFunctionName) :
            Plato::AbstractScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction)
    /**************************************************************************/
    {
      Plato::ThermalModelFactory<mSpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      m_cellConductivity = materialModel->getConductivityMatrix();

      m_quadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (int d=2; d<=mSpaceDim; d++)
      { 
        m_quadratureWeight /= Plato::Scalar(d);
      }
    
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

      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::ScalarGrad<mSpaceDim>                    scalarGrad;
      Plato::ThermalFlux<mSpaceDim>                   thermalFlux(m_cellConductivity);
      Plato::ScalarProduct<mSpaceDim>                 scalarProduct;

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<GradScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tgrad("temperature gradient",numCells,mSpaceDim);

      Kokkos::View<ConfigScalarType***, Kokkos::LayoutRight, Plato::MemSpace>
        gradient("gradient",numCells,m_numNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tflux("thermal flux",numCells,mSpaceDim);

      auto quadratureWeight = m_quadratureWeight;
      auto applyWeighting  = m_applyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute temperature gradient
        //
        scalarGrad(aCellOrdinal, tgrad, aState, gradient);

        // compute flux
        //
        thermalFlux(aCellOrdinal, tflux, tgrad);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, tflux, aControl);
    
        // compute element internal energy (inner product of tgrad and weighted tflux)
        //
        scalarProduct(aCellOrdinal, aResult, tflux, tgrad, cellVolume);

      },"energy gradient");
    }
};
// class InternalThermalEnergyInc

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::InternalThermalEnergy, Plato::SimplexThermal, 1)
PLATO_EXPL_DEC(Plato::InternalThermalEnergyInc, Plato::SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::InternalThermalEnergy, Plato::SimplexThermal, 2)
PLATO_EXPL_DEC(Plato::InternalThermalEnergyInc, Plato::SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::InternalThermalEnergy, Plato::SimplexThermal, 3)
PLATO_EXPL_DEC(Plato::InternalThermalEnergyInc, Plato::SimplexThermal, 3)
#endif

#endif
