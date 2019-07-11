#ifndef PRESSURE_GRADIENT_PROJECTION_RESIDUAL_HPP
#define PRESSURE_GRADIENT_PROJECTION_RESIDUAL_HPP

#include <memory>

#include "plato/PlatoTypes.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PressureGradient.hpp"
#include "plato/AbstractVectorFunctionVMS.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/ProjectToNode.hpp"
#include "plato/CellForcing.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class PressureGradientProjectionResidual :
        public Plato::Simplex<EvaluationType::SpatialDim>,
        public Plato::AbstractVectorFunctionVMS<EvaluationType>
/******************************************************************************/
{
private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;

    using Plato::AbstractVectorFunctionVMS<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunctionVMS<EvaluationType>::m_dataMap;
    using Plato::AbstractVectorFunctionVMS<EvaluationType>::mMeshSets;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType m_indicatorFunction;
    Plato::ApplyWeighting<SpaceDim, SpaceDim, IndicatorFunctionType> m_applyVectorWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

public:
    /**************************************************************************/
    PressureGradientProjectionResidual(Omega_h::Mesh& aMesh,
                               Omega_h::MeshSets& aMeshSets,
                               Plato::DataMap& aDataMap,
                               Teuchos::ParameterList& aProblemParams,
                               Teuchos::ParameterList& aPenaltyParams) :
            Plato::AbstractVectorFunctionVMS<EvaluationType>(aMesh, aMeshSets, aDataMap),
            m_indicatorFunction(aPenaltyParams),
            m_applyVectorWeighting(m_indicatorFunction),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType>     & aNodalPGradWS,
                  const Plato::ScalarMultiVectorT<NodeStateScalarType> & aPressureWS,
                  const Plato::ScalarMultiVectorT<ControlScalarType>   & aControlWS,
                  const Plato::ScalarArray3DT<ConfigScalarType>        & aConfigWS,
                  Plato::ScalarMultiVectorT<ResultScalarType>          & aResultWS,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      Plato::ComputeGradientWorkset <SpaceDim> computeGradient;
      Plato::PressureGradient       <SpaceDim> kinematics;

      Plato::InterpolateFromNodal   <SpaceDim, SpaceDim, 0, SpaceDim>         interpolatePGradFromNodal;
      
      Plato::ScalarVectorT      <ConfigScalarType>  tCellVolume     ("cell weight",        tNumCells);
      Plato::ScalarMultiVectorT <ResultScalarType>  tProjectedPGrad ("projected p grad",   tNumCells, SpaceDim);
      Plato::ScalarMultiVectorT <ResultScalarType>  tComputedPGrad  ("compute p grad",     tNumCells, SpaceDim);
      Plato::ScalarArray3DT     <ConfigScalarType>  tGradient       ("gradient",           tNumCells, m_numNodesPerCell, SpaceDim);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tBasisFunctions   = mCubatureRule->getBasisFunctions();

      auto& applyVectorWeighting = m_applyVectorWeighting;

      Plato::ProjectToNode<SpaceDim> projectPGradToNodal;

      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        // compute gradient operator and cell volume
        //
        computeGradient(cellOrdinal, tGradient, aConfigWS, tCellVolume);
        tCellVolume(cellOrdinal) *= tQuadratureWeight;

        // compute pressure gradient
        //
        kinematics(cellOrdinal, tComputedPGrad, aPressureWS, tGradient);

        // interpolate projected pressure gradient from nodes
        //
        interpolatePGradFromNodal (cellOrdinal, tBasisFunctions, aNodalPGradWS, tProjectedPGrad);

        // apply weighting
        //
        applyVectorWeighting (cellOrdinal, tComputedPGrad, aControlWS);
        applyVectorWeighting (cellOrdinal, tProjectedPGrad, aControlWS);

        // project pressure gradient to nodes
        //
        projectPGradToNodal (cellOrdinal, tCellVolume, tBasisFunctions, tProjectedPGrad, aResultWS);
        projectPGradToNodal (cellOrdinal, tCellVolume, tBasisFunctions, tComputedPGrad,  aResultWS, /*scale=*/-1.0);

      }, "Projected pressure gradient residual");
    }
};
// class ThermoelastostaticResidual

} // namespace Plato
#endif
