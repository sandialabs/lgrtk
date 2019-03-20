#ifndef TEMPERATURE_PNORM_HPP
#define TEMPERATURE_PNORM_HPP

#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/StateValues.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TemperatureAverageInc : 
  public SimplexThermal<EvaluationType::SpatialDim>,
  public AbstractScalarFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    using Simplex<SpaceDim>::m_numNodesPerCell;
    using SimplexThermal<SpaceDim>::m_numDofsPerCell;
    using SimplexThermal<SpaceDim>::m_numDofsPerNode;

    using AbstractScalarFunctionInc<EvaluationType>::mMesh;
    using AbstractScalarFunctionInc<EvaluationType>::m_dataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> m_cubatureRule;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim,m_numDofsPerNode,IndicatorFunctionType> m_applyWeighting;

  public:
    /**************************************************************************/
    TemperatureAverageInc(Omega_h::Mesh& aMesh,
                        Omega_h::MeshSets& aMeshSets,
                        Plato::DataMap& aDataMap,
                        Teuchos::ParameterList& aProblemParams,
                        Teuchos::ParameterList& aPenaltyParams) :
            AbstractScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, "Temperature Average"),
            m_cubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction) {}
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
        typename Plato::fad_type_t<SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ControlScalarType>;

      Plato::ScalarMultiVectorT<StateScalarType>  tStateValues("temperature at GPs", numCells, m_numDofsPerNode);
      Plato::ScalarMultiVectorT<TScalarType>  tWeightedStateValues("weighted temperature at GPs", numCells, m_numDofsPerNode);

      auto basisFunctions = m_cubatureRule->getBasisFunctions();
      auto quadratureWeight = m_cubatureRule->getCubWeight();
      auto applyWeighting  = m_applyWeighting;
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
#endif
