#ifndef VOLUME_HPP
#define VOLUME_HPP

#include "plato/ApplyWeighting.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"

/******************************************************************************/
template<typename EvaluationType, typename PenaltyFunctionType>
class Volume : public AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using AbstractScalarFunction<EvaluationType>::mMesh;
    using AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mQuadratureWeight;

    PenaltyFunctionType mPenaltyFunction;
    ApplyWeighting<SpaceDim,1,PenaltyFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    Volume(Omega_h::Mesh& aMesh, 
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap& aDataMap, 
           Teuchos::ParameterList&, 
           Teuchos::ParameterList& aPenaltyParams) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Volume"),
            mPenaltyFunction(aPenaltyParams),
            mApplyWeighting(mPenaltyFunction)
    /**************************************************************************/
    {
      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDimIndex=2; tDimIndex<=SpaceDim; tDimIndex++)
      { 
        mQuadratureWeight /= Plato::Scalar(tDimIndex);
      }
    
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> &,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;

      auto tQuadratureWeight = mQuadratureWeight;
      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tQuadratureWeight;

        aResult(aCellOrdinal) = tCellVolume;

        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, aResult, aControl);
    
      },"volume");
    }
};

#endif
