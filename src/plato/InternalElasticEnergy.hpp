#ifndef INTERNAL_ELASTIC_ENERGY_HPP
#define INTERNAL_ELASTIC_ENERGY_HPP

#include "plato/SimplexFadTypes.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/Strain.hpp"
#include "plato/LinearStress.hpp"
#include "plato/LinearElasticMaterial.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/ExpInstMacros.hpp"
#include "plato/ToMap.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalElasticEnergy : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    
    using Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms; /*!< number of Voigt terms */
    using Plato::Simplex<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell */
    using Plato::SimplexMechanics<mSpaceDim>::m_numDofsPerCell; /*!< number of degree of freedom per cell */

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh; /*!< mesh database */
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap; /*!< Plato Analyze database */

    using StateScalarType   = typename EvaluationType::StateScalarType; /*!< automatic differentiation type for states */
    using ControlScalarType = typename EvaluationType::ControlScalarType; /*!< automatic differentiation type for controls */
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType; /*!< automatic differentiation type for configuration */
    using ResultScalarType  = typename EvaluationType::ResultScalarType; /*!< automatic differentiation type for results */

    Omega_h::Matrix< m_numVoigtTerms, m_numVoigtTerms> m_cellStiffness; /*!< matrix with Lame constants for a cell/element */
    
    Plato::Scalar m_quadratureWeight; /*!< quadrature weight for simplex element */

    IndicatorFunctionType m_indicatorFunction; /*!< penalty function */
    Plato::ApplyWeighting<mSpaceDim,m_numVoigtTerms,IndicatorFunctionType> m_applyWeighting; /*!< apply penalty function */

    std::vector<std::string> m_plottable; /*!< database of output field names */

  public:
    /******************************************************************************//**
     * @brief Constructor
     * @param aMesh volume mesh database
     * @param aMeshSets surface mesh database
     * @param aProblemParams input database for overall problem
     * @param aPenaltyParams input database for penalty function
    **********************************************************************************/
    InternalElasticEnergy(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams,
                          std::string& aFunctionName) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction)
    {
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
        auto tMaterialModel = tMaterialModelFactory.create();
        m_cellStiffness = tMaterialModel->getStiffnessMatrix();

        m_quadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
        for(Plato::OrdinalType tDim = 2; tDim <= mSpaceDim; tDim++)
        {
            m_quadratureWeight /= Plato::Scalar(tDim);
        }

        if(aProblemParams.isType < Teuchos::Array < std::string >> ("Plottable"))
        {
            m_plottable = aProblemParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
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

        Plato::Strain<mSpaceDim> tComputeVoigtStrain;
        Plato::ScalarProduct<m_numVoigtTerms> tComputeScalarProduct;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::LinearStress<mSpaceDim> tComputeVoigtStress(m_cellStiffness);

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Kokkos::View<StrainScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tStrain("strain",tNumCells,m_numVoigtTerms);

      Kokkos::View<ConfigScalarType***, Kokkos::LayoutRight, Plato::MemSpace>
        tGradient("gradient",tNumCells,m_numNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tStress("stress",tNumCells,m_numVoigtTerms);

      auto tQuadratureWeight = m_quadratureWeight;
      auto tApplyWeighting  = m_applyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain
        //
        tComputeVoigtStrain(aCellOrdinal, tStrain, aState, tGradient);

        // compute stress
        //
        tComputeVoigtStress(aCellOrdinal, tStress, tStrain);

        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, tStress, aControl);
    
        // compute element internal energy (inner product of strain and weighted stress)
        //
        tComputeScalarProduct(aCellOrdinal, aResult, tStress, tStrain, tCellVolume);

      },"energy gradient");

      if( std::count(m_plottable.begin(),m_plottable.end(),"strain") ) toMap(m_dataMap, tStrain, "strain");
      if( std::count(m_plottable.begin(),m_plottable.end(),"stress") ) toMap(m_dataMap, tStress, "stress");

    }
};
// class InternalElasticEnergy

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::InternalElasticEnergy, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::InternalElasticEnergy, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::InternalElasticEnergy, Plato::SimplexMechanics, 3)
#endif

#endif
