#ifndef FLUX_P_NORM_HPPS
#define FLUX_P_NORM_HPP

#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/VectorPNorm.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/LinearThermalMaterial.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class FluxPNorm : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerCell;

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix< SpaceDim, SpaceDim> mCellConductivity;
    
    Plato::Scalar mQuadratureWeight;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> mApplyWeighting;

    Plato::OrdinalType mExponent;

  public:
    /**************************************************************************/
    FluxPNorm(Omega_h::Mesh& aMesh, 
              Omega_h::MeshSets& aMeshSets,
              Plato::DataMap& aDataMap, 
              Teuchos::ParameterList& aProblemParams, 
              Teuchos::ParameterList& aPenaltyParams,
              std::string aFunctionName) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mIndicatorFunction(aPenaltyParams),
            mApplyWeighting(mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ThermalModelFactory<SpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      mCellConductivity = materialModel->getConductivityMatrix();

      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (int d=2; d<=SpaceDim; d++)
      { 
        mQuadratureWeight /= Plato::Scalar(d);
      }

      auto params = aProblemParams.get<Teuchos::ParameterList>(aFunctionName);

      mExponent = params.get<double>("Exponent");
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

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Plato::ScalarGrad<SpaceDim>                    scalarGrad;
      Plato::ThermalFlux<SpaceDim>                   thermalFlux(mCellConductivity);
      Plato::VectorPNorm<SpaceDim>                   vectorPNorm;

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>,StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<GradScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tgrad("temperature gradient",numCells,SpaceDim);

      Kokkos::View<ConfigScalarType***, Kokkos::LayoutRight, Plato::MemSpace>
        gradient("gradient",numCells,mNumNodesPerCell,SpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tflux("thermal flux",numCells,SpaceDim);

      auto quadratureWeight = mQuadratureWeight;
      auto& applyWeighting  = mApplyWeighting;
      auto exponent         = mExponent;
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
    
        // compute vector p-norm of flux
        //
        vectorPNorm(aCellOrdinal, aResult, tflux, exponent, cellVolume);

      },"Flux P-norm");
    }

    /**************************************************************************/
    void
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      auto scale = pow(resultScalar,(1.0-mExponent)/mExponent)/mExponent;
      auto numEntries = resultVector.size();
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numEntries), LAMBDA_EXPRESSION(int entryOrdinal)
      {
        resultVector(entryOrdinal) *= scale;
      },"scale vector");
    }

    /**************************************************************************/
    void
    postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      resultValue = pow(resultValue, 1.0/mExponent);
    }
};
// class FluxPNorm

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::FluxPNorm, Plato::SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::FluxPNorm, Plato::SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::FluxPNorm, Plato::SimplexThermal, 3)
#endif

#endif
