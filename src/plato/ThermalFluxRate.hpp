#ifndef THERMAL_FLUX_RATE_HPP
#define THERMAL_FLUX_RATE_HPP

#include "plato/LinearThermalMaterial.hpp"

#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class ThermalFluxRate : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    using Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerNode;

    using Plato::AbstractScalarFunctionInc<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunctionInc<EvaluationType>::mDataMap;
    using Plato::AbstractScalarFunctionInc<EvaluationType>::mMeshSets;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;
    
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>> mBoundaryLoads;

  public:
    /**************************************************************************/
    ThermalFluxRate(Omega_h::Mesh& aMesh,
                    Omega_h::MeshSets& aMeshSets,
                    Plato::DataMap& aDataMap,
                    Teuchos::ParameterList& problemParams) :
            Plato::AbstractScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, "Thermal Flux Rate"),
            mBoundaryLoads(nullptr)
    /**************************************************************************/
    {
      if(problemParams.isSublist("Natural Boundary Conditions"))
      {
          mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>>(problemParams.sublist("Natural Boundary Conditions"));
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

      if( mBoundaryLoads != nullptr )
      {
        Plato::ScalarMultiVectorT<ResultScalarType> boundaryLoads("boundary loads", numCells, mNumDofsPerCell);
        Kokkos::deep_copy(boundaryLoads, 0.0);

        mBoundaryLoads->get( &mMesh, mMeshSets, aState, aControl, boundaryLoads, 1.0/aTimeStep );
        mBoundaryLoads->get( &mMesh, mMeshSets, aPrevState, aControl, boundaryLoads, 1.0/aTimeStep );

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(const int & cellOrdinal)
        {
          for( int iNode=0; iNode<mNumNodesPerCell; iNode++) {
            aResult(cellOrdinal) += (aState(cellOrdinal, iNode) - aPrevState(cellOrdinal, iNode))*boundaryLoads(cellOrdinal,iNode);
          }
        },"scalar product");
      }
    }
};

} // namespace Plato

#endif
