#ifndef TEMPERATURE_PNORM_HPP
#define TEMPERATURE_PNORM_HPP

#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/StateValues.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"
#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TemperatureAverageInc : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerNode;

    using Plato::AbstractScalarFunctionInc<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunctionInc<EvaluationType>::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> mCubatureRule;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim,mNumDofsPerNode,IndicatorFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    TemperatureAverageInc(Omega_h::Mesh& aMesh,
                        Omega_h::MeshSets& aMeshSets,
                        Plato::DataMap& aDataMap,
                        Teuchos::ParameterList& aProblemParams,
                        Teuchos::ParameterList& aPenaltyParams,
                        std::string& aFunctionName) :
            Plato::AbstractScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
            mIndicatorFunction(aPenaltyParams),
            mApplyWeighting(mIndicatorFunction) {}
    /**************************************************************************/

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

      Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;
      Plato::StateValues                 tComputeStateValues;

      using TScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ControlScalarType>;

      Plato::ScalarMultiVectorT<StateScalarType>  tStateValues("temperature at GPs", numCells, mNumDofsPerNode);
      Plato::ScalarMultiVectorT<TScalarType>  tWeightedStateValues("weighted temperature at GPs", numCells, mNumDofsPerNode);

      auto basisFunctions = mCubatureRule->getBasisFunctions();
      auto quadratureWeight = mCubatureRule->getCubWeight();
      auto applyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        ConfigScalarType tCellVolume(0.0);
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);

        // compute temperature at Gauss points
        //
        tComputeStateValues(aCellOrdinal, basisFunctions, aState, tStateValues);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, tStateValues, tWeightedStateValues, aControl);
    
        aResult(aCellOrdinal) = tWeightedStateValues(aCellOrdinal,0)*tCellVolume;

      },"temperature");
    }
};
// class

} // namespace Plato TemperatureAverageInc

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::TemperatureAverageInc, Plato::SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::TemperatureAverageInc, Plato::SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::TemperatureAverageInc, Plato::SimplexThermal, 3)
#endif

#endif
