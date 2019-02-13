#ifndef THERMAL_FLUX_RATE_HPP
#define THERMAL_FLUX_RATE_HPP

#include "plato/LinearThermalMaterial.hpp"

#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"

/******************************************************************************/
template<typename EvaluationType>
class ThermalFluxRate : 
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
    using AbstractScalarFunctionInc<EvaluationType>::mMeshSets;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;
    
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>> m_boundaryLoads;

  public:
    /**************************************************************************/
    ThermalFluxRate(Omega_h::Mesh& aMesh,
                    Omega_h::MeshSets& aMeshSets,
                    Plato::DataMap& aDataMap,
                    Teuchos::ParameterList& problemParams) :
            AbstractScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, "Thermal Flux Rate"),
            m_boundaryLoads(nullptr)
    /**************************************************************************/
    {
      if(problemParams.isSublist("Natural Boundary Conditions"))
      {
          m_boundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>>(problemParams.sublist("Natural Boundary Conditions"));
      }
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<PrevStateScalarType> & aPrevState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {

      Kokkos::deep_copy(aResult, 0.0);

      auto numCells = mMesh.nelems();

      if( m_boundaryLoads != nullptr )
      {
        Plato::ScalarMultiVectorT<ResultScalarType> boundaryLoads("boundary loads", numCells, m_numDofsPerCell);
        Kokkos::deep_copy(boundaryLoads, 0.0);

        m_boundaryLoads->get( &mMesh, mMeshSets, aState, aControl, boundaryLoads, 1.0/aTimeStep );
        m_boundaryLoads->get( &mMesh, mMeshSets, aPrevState, aControl, boundaryLoads, 1.0/aTimeStep );

        Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & cellOrdinal)
        {
          for( int iNode=0; iNode<m_numNodesPerCell; iNode++) {
            aResult(cellOrdinal) += (aState(cellOrdinal, iNode) - aPrevState(cellOrdinal, iNode))*boundaryLoads(cellOrdinal,iNode);
          }
        },"scalar product");
      }
    }
};

#endif
