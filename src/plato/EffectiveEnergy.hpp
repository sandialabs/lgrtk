#ifndef EFFECTIVE_ELASTIC_ENERGY_HPP
#define EFFECTIVE_ELASTIC_ENERGY_HPP

#include "plato/SimplexMechanics.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/Strain.hpp"
#include "plato/LinearElasticMaterial.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/HomogenizedStress.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/ExpInstMacros.hpp"
#include "plato/ToMap.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute internal effective energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class EffectiveEnergy : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<mSpaceDim>::mNumDofsPerCell;

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::mDataMap;
    using Plato::AbstractScalarFunction<EvaluationType>::mFunctionName;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim,mNumVoigtTerms,IndicatorFunctionType> mApplyWeighting;

    Omega_h::Matrix< mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;
    Omega_h::Vector<mNumVoigtTerms> mAssumedStrain;
    Plato::OrdinalType mColumnIndex;
    Plato::Scalar mQuadratureWeight;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    EffectiveEnergy(Omega_h::Mesh& aMesh,
                    Omega_h::MeshSets& aMeshSets,
                    Plato::DataMap& aDataMap,
                    Teuchos::ParameterList& aProblemParams,
                    Teuchos::ParameterList& aPenaltyParams,
                    std::string& aFunctionName) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mIndicatorFunction(aPenaltyParams),
            mApplyWeighting(mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      mCellStiffness = materialModel->getStiffnessMatrix();

      Teuchos::ParameterList& tParams = aProblemParams.sublist(aFunctionName);
      auto tAssumedStrain = tParams.get<Teuchos::Array<Plato::Scalar>>("Assumed Strain");
      assert(tAssumedStrain.size() == mNumVoigtTerms);
      for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
      {
          mAssumedStrain[iVoigt] = tAssumedStrain[iVoigt];
      }

      // parse cell problem forcing
      //
      if(aProblemParams.isSublist("Cell Problem Forcing"))
      {
          mColumnIndex = aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
      }
      else
      {
          // JR TODO: throw
      }

      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType d=2; d<=mSpaceDim; d++)
      { 
          mQuadratureWeight /= Plato::Scalar(d);
      }
    
      if( tParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = tParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      Plato::Strain<mSpaceDim> voigtStrain;
      Plato::ScalarProduct<mNumVoigtTerms> scalarProduct;
      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::HomogenizedStress < mSpaceDim > homogenizedStress(mCellStiffness, mColumnIndex);

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<StrainScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        strain("strain",numCells,mNumVoigtTerms);

      Kokkos::View<ConfigScalarType***, Kokkos::LayoutRight, Plato::MemSpace>
        gradient("gradient",numCells,mNumNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        stress("stress",numCells,mNumVoigtTerms);

      auto quadratureWeight = mQuadratureWeight;
      auto applyWeighting   = mApplyWeighting;
      auto assumedStrain    = mAssumedStrain;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute strain
        //
        voigtStrain(aCellOrdinal, strain, aState, gradient);

        // compute stress
        //
        homogenizedStress(aCellOrdinal, stress, strain);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, stress, aControl);
    
        // compute element internal energy (inner product of strain and weighted stress)
        //
        scalarProduct(aCellOrdinal, aResult, stress, assumedStrain, cellVolume);

      },"energy gradient");

      if( std::count(mPlottable.begin(),mPlottable.end(),"effective stress") ) toMap(mDataMap, stress, "effective stress");
      if( std::count(mPlottable.begin(),mPlottable.end(),"cell volume") ) toMap(mDataMap, cellVolume, "cell volume");

    }
};
// class EffectiveEnergy

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::EffectiveEnergy, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::EffectiveEnergy, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::EffectiveEnergy, Plato::SimplexMechanics, 3)
#endif

#endif
