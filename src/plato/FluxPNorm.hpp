#ifndef FLUX_P_NORM_HPPS
#define FLUX_P_NORM_HPP

#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/VectorPNorm.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class FluxPNorm : 
  public SimplexThermal<EvaluationType::SpatialDim>,
  public AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using SimplexThermal<SpaceDim>::m_numDofsPerCell;

    using AbstractScalarFunction<EvaluationType>::mMesh;
    using AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix< SpaceDim, SpaceDim> m_cellConductivity;
    
    Plato::Scalar m_quadratureWeight;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> m_applyWeighting;

    int m_exponent;

  public:
    /**************************************************************************/
    FluxPNorm(Omega_h::Mesh& aMesh, 
              Omega_h::MeshSets& aMeshSets,
              Plato::DataMap& aDataMap, 
              Teuchos::ParameterList& aProblemParams, 
              Teuchos::ParameterList& aPenaltyParams) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Flux P-Norm"),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction)
    /**************************************************************************/
    {
      lgr::ThermalModelFactory<SpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      m_cellConductivity = materialModel->getConductivityMatrix();

      m_quadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (int d=2; d<=SpaceDim; d++)
      { 
        m_quadratureWeight /= Plato::Scalar(d);
      }

      auto params = aProblemParams.get<Teuchos::ParameterList>("Flux P-Norm");

      m_exponent = params.get<double>("Exponent");
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
      ScalarGrad<SpaceDim>                    scalarGrad;
      ThermalFlux<SpaceDim>                   thermalFlux(m_cellConductivity);
      VectorPNorm<SpaceDim>                   vectorPNorm;

      using GradScalarType =
        typename Plato::fad_type_t<SimplexThermal<EvaluationType::SpatialDim>,StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<GradScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tgrad("temperature gradient",numCells,SpaceDim);

      Kokkos::View<ConfigScalarType***, Kokkos::LayoutRight, Plato::MemSpace>
        gradient("gradient",numCells,m_numNodesPerCell,SpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tflux("thermal flux",numCells,SpaceDim);

      auto quadratureWeight = m_quadratureWeight;
      auto& applyWeighting  = m_applyWeighting;
      auto exponent         = m_exponent;
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
      auto scale = pow(resultScalar,(1.0-m_exponent)/m_exponent)/m_exponent;
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
      resultValue = pow(resultValue, 1.0/m_exponent);
    }
};

#endif
