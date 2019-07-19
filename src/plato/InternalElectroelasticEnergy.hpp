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
    
    using Plato::SimplexElectromechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */
    using Plato::SimplexElectromechanics<mSpaceDim>::mNumDofsPerCell; /*!< number of degree of freedom per cell */

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh; /*!< mesh database */
    using Plato::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< Plato Analyze database */

    using StateScalarType   = typename EvaluationType::StateScalarType; /*!< automatic differentiation type for states */
    using ControlScalarType = typename EvaluationType::ControlScalarType; /*!< automatic differentiation type for controls */
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType; /*!< automatic differentiation type for configuration */
    using ResultScalarType  = typename EvaluationType::ResultScalarType; /*!< automatic differentiation type for results */

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<mSpaceDim>> mMaterialModel; /*!< electrostatics material model */
    
    IndicatorFunctionType mIndicatorFunction; /*!< penalty function */
    ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting; /*!< apply penalty function */
    ApplyWeighting<mSpaceDim, mSpaceDim, IndicatorFunctionType> mApplyEDispWeighting; /*!< apply penalty function */

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule; /*!< integration rule */

    std::vector<std::string> mPlottable; /*!< database of output field names */

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
                          Teuchos::ParameterList& aPenaltyParams,
                          std::string& aFunctionName ) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mIndicatorFunction(aPenaltyParams),
            mApplyStressWeighting(mIndicatorFunction),
            mApplyEDispWeighting(mIndicatorFunction),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
      Plato::ElectroelasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create();

      if( aProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = aProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
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
      Plato::EMKinetics<mSpaceDim>                    tKinetics(mMaterialModel);

      Plato::ScalarProduct<mNumVoigtTerms>          tMechanicalScalarProduct;
      Plato::ScalarProduct<mSpaceDim>                tElectricalScalarProduct;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType>   tStrain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType>   tEfield("efield", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> tStress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tEdisp ("edisp" , tNumCells, mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>   tGradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();

      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyEDispWeighting  = mApplyEDispWeighting;
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

      if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tStrain, "strain");
      if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tStress, "stress");
      if( std::count(mPlottable.begin(),mPlottable.end(),"edisp" ) ) toMap(mDataMap, tStress, "edisp" );

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
