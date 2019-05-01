#ifndef INTERNAL_ELECTROELASTIC_ENERGY_HPP
#define INTERNAL_ELECTROELASTIC_ENERGY_HPP

#include "plato/SimplexElectromechanics.hpp"
#include "plato/LinearElectroelasticMaterial.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/EMKinematics.hpp"
#include "plato/EMKinetics.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/Strain.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/ToMap.hpp"
#include "plato/ExpInstMacros.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute internal electro-static energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalElectroelasticEnergy : 
  public Plato::SimplexElectromechanics<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    
    using Plato::SimplexElectromechanics<mSpaceDim>::m_numVoigtTerms; /*!< number of Voigt terms */
    using Plato::Simplex<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell */
    using Plato::SimplexElectromechanics<mSpaceDim>::m_numDofsPerCell; /*!< number of degree of freedom per cell */

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh; /*!< mesh database */
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap; /*!< Plato Analyze database */

    using StateScalarType   = typename EvaluationType::StateScalarType; /*!< automatic differentiation type for states */
    using ControlScalarType = typename EvaluationType::ControlScalarType; /*!< automatic differentiation type for controls */
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType; /*!< automatic differentiation type for configuration */
    using ResultScalarType  = typename EvaluationType::ResultScalarType; /*!< automatic differentiation type for results */

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<mSpaceDim>> m_materialModel; /*!< electrostatics material model */
    
    IndicatorFunctionType m_indicatorFunction; /*!< penalty function */
    ApplyWeighting<mSpaceDim, m_numVoigtTerms, IndicatorFunctionType> m_applyStressWeighting; /*!< apply penalty function */
    ApplyWeighting<mSpaceDim, mSpaceDim, IndicatorFunctionType> m_applyEDispWeighting; /*!< apply penalty function */

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> m_CubatureRule; /*!< integration rule */

    std::vector<std::string> m_plottable; /*!< database of output field names */

  public:
    /******************************************************************************//**
     * @brief Constructor
     * @param aMesh volume mesh database
     * @param aMeshSets surface mesh database
     * @param aProblemParams input database for overall problem
     * @param aPenaltyParams input database for penalty function
    **********************************************************************************/
    InternalElectroelasticEnergy(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams ) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Internal Electroelastic Energy"),
            m_indicatorFunction(aPenaltyParams),
            m_applyStressWeighting(m_indicatorFunction),
            m_applyEDispWeighting(m_indicatorFunction),
            m_CubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
      Plato::ElectroelasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      m_materialModel = mmfactory.create();

      if( aProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        m_plottable = aProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
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

      using GradScalarType = 
        typename Plato::fad_type_t<Plato::SimplexElectromechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::EMKinematics<mSpaceDim>                  tKinematics;
      Plato::EMKinetics<mSpaceDim>                    tKinetics(m_materialModel);

      Plato::ScalarProduct<m_numVoigtTerms>          tMechanicalScalarProduct;
      Plato::ScalarProduct<mSpaceDim>                tElectricalScalarProduct;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType>   tStrain("strain", tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType>   tEfield("efield", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> tStress("stress", tNumCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tEdisp ("edisp" , tNumCells, mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>   tGradient("gradient",tNumCells,m_numNodesPerCell,mSpaceDim);

      auto tQuadratureWeight = m_CubatureRule->getCubWeight();

      auto& tApplyStressWeighting = m_applyStressWeighting;
      auto& tApplyEDispWeighting  = m_applyEDispWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain and electric field
        //
        tKinematics(aCellOrdinal, tStrain, tEfield, aState, tGradient);

        // compute stress and electric displacement
        //
        tKinetics(aCellOrdinal, tStress, tEdisp, tStrain, tEfield);

        // apply weighting
        //
        tApplyStressWeighting(aCellOrdinal, tStress, aControl);
        tApplyEDispWeighting (aCellOrdinal, tEdisp,  aControl);
    
        // compute element internal energy (inner product of strain and weighted stress)
        //
        tMechanicalScalarProduct(aCellOrdinal, aResult, tStress, tStrain, tCellVolume);
        tElectricalScalarProduct(aCellOrdinal, aResult, tEdisp,  tEfield, tCellVolume, -1.0);

      },"energy gradient");

      if( std::count(m_plottable.begin(),m_plottable.end(),"strain") ) toMap(m_dataMap, tStrain, "strain");
      if( std::count(m_plottable.begin(),m_plottable.end(),"stress") ) toMap(m_dataMap, tStress, "stress");
      if( std::count(m_plottable.begin(),m_plottable.end(),"edisp" ) ) toMap(m_dataMap, tStress, "edisp" );

    }
};
// class InternalElectroelasticEnergy

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::InternalElectroelasticEnergy, Plato::SimplexElectromechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::InternalElectroelasticEnergy, Plato::SimplexElectromechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::InternalElectroelasticEnergy, Plato::SimplexElectromechanics, 3)
#endif

#endif
