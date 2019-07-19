#ifndef PROJECT_TO_NODE
#define PROJECT_TO_NODE

#include "plato/Simplex.hpp"

namespace Plato {

/******************************************************************************/
/*! Project to node functor.
  
    Given values at gauss points, multiply by the basis functions to project
    to the nodes.
*/
/******************************************************************************/
template<int SpaceDim, int NumDofsPerNode=SpaceDim, int DofOffset=0>
class ProjectToNode : public Plato::Simplex<SpaceDim>
{
  private:

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;

  public:
    /******************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarVectorT<Plato::Scalar> & tBasisFunctions,
                                       const Plato::ScalarMultiVectorT<GaussPointScalarType> & aStateValues,
                                       const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult, 
                                             Plato::Scalar scale = 1.0 ) const
    /******************************************************************************/
    {  
        const Plato::OrdinalType tNumDofs = aStateValues.extent(1);
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofs; tDofIndex++)
            {
                Plato::OrdinalType tMyDofIndex = (NumDofsPerNode * tNodeIndex) + tDofIndex + DofOffset;
                aResult(aCellOrdinal, tMyDofIndex) += scale * tBasisFunctions(tNodeIndex)
                        * aStateValues(aCellOrdinal, tDofIndex) * aCellVolume(aCellOrdinal);
            }
        }
    }

    /******************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarVectorT<Plato::Scalar> & tBasisFunctions,
                                       const Plato::ScalarVectorT<GaussPointScalarType> & aStateValues,
                                       const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult, 
                                             Plato::Scalar scale = 1.0 ) const
    /******************************************************************************/
    {  
        const Plato::OrdinalType tNumDofs = aStateValues.extent(1);
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tMyDofIndex = (NumDofsPerNode * tNodeIndex) + DofOffset;
            aResult(aCellOrdinal, tMyDofIndex) += scale * tBasisFunctions(tNodeIndex)
                    * aStateValues(aCellOrdinal) * aCellVolume(aCellOrdinal);
        }
    }
};

}
#endif
