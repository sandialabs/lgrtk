#ifndef THERMOSTATIC_RESIDUAL_HPP
#define THERMOSTATIC_RESIDUAL_HPP

#include "plato/SimplexThermal.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"
#include "plato/ToMap.hpp"

#include "plato/LinearThermalMaterial.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/SimplexFadTypes.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermostaticResidual : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<mSpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermal<mSpaceDim>::mNumDofsPerNode;

    using Plato::AbstractVectorFunction<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunction<EvaluationType>::mDataMap;
    using Plato::AbstractVectorFunction<EvaluationType>::mMeshSets;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix< mSpaceDim, mSpaceDim> mCellConductivity;
    
    Plato::Scalar mQuadratureWeight;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mSpaceDim,mSpaceDim,IndicatorFunctionType> mApplyWeighting;

    std::shared_ptr<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>> mBoundaryLoads;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    ThermostaticResidual(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aProblemParams,
                         Teuchos::ParameterList& penaltyParams) :
            Plato::AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mIndicatorFunction(penaltyParams),
            mApplyWeighting(mIndicatorFunction),
            mBoundaryLoads(nullptr)
    /**************************************************************************/
    {
      Plato::ThermalModelFactory<mSpaceDim> tMaterialFactory(aProblemParams);
      auto tMaterialModel = tMaterialFactory.create();
      mCellConductivity = tMaterialModel->getConductivityMatrix();

      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDim=2; tDim<=mSpaceDim; tDim++)
      { 
        mQuadratureWeight /= Plato::Scalar(tDim);
      }

      // parse boundary Conditions
      // 
      if(aProblemParams.isSublist("Natural Boundary Conditions"))
      {
          mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
      }

      auto tResidualParams = problemParams.sublist("Thermostatics");
      if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    
    }


    /**************************************************************************/
    void
    evaluate( const Plato::ScalarMultiVectorT<StateScalarType  > & aState,
              const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
              const Plato::ScalarArray3DT<ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT<ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      Kokkos::deep_copy(aResult, 0.0);

      auto tNumCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Kokkos::View<GradScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tGrad("temperature gradient",tNumCells,mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tFlux("thermal flux",tNumCells,mSpaceDim);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<mSpaceDim>  tComputeGradient;

      Plato::ScalarGrad<mSpaceDim>            tScalarGrad;
      Plato::ThermalFlux<mSpaceDim>           tThermalFlux(mCellConductivity);
      Plato::FluxDivergence<mSpaceDim>        tFluxDivergence;
    
      auto& tApplyWeighting  = mApplyWeighting;
      auto tQuadratureWeight = mQuadratureWeight;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
      {
    
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;
    
        // compute temperature gradient
        //
        tScalarGrad(aCellOrdinal, tGrad, aState, tGradient);
    
        // compute flux
        //
        tThermalFlux(aCellOrdinal, tFlux, tGrad);
    
        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, tFlux, aControl);
    
        // compute stress divergence
        //
        tFluxDivergence(aCellOrdinal, aResult, tFlux, tGradient, tCellVolume);
        
      },"flux divergence");

      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( &mMesh, mMeshSets, aState, aControl, aResult, -1.0 );
      }

      if( std::count(mPlottable.begin(),mPlottable.end(),"tgrad") ) toMap(mDataMap, tgrad, "tgrad");
      if( std::count(mPlottable.begin(),mPlottable.end(),"flux" ) ) toMap(mDataMap, tflux, "flux" );
    }
};
// class ThermostaticResidual

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::ThermostaticResidual, Plato::SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::ThermostaticResidual, Plato::SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::ThermostaticResidual, Plato::SimplexThermal, 3)
#endif

#endif
