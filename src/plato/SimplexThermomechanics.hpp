#ifndef SIMPLEX_TWOFIELD_THERMOMECHANICS_HPP
#define SIMPLEX_TWOFIELD_THERMOMECHANICS_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based thermomechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexThermomechanics : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::Simplex<SpaceDim>::m_numSpatialDims;

    static constexpr Plato::OrdinalType m_numVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType m_TDofOffset      = SpaceDim;
    static constexpr Plato::OrdinalType m_numDofsPerNode  = SpaceDim + 1;
    static constexpr Plato::OrdinalType m_numDofsPerCell  = m_numDofsPerNode*m_numNodesPerCell;

    static constexpr Plato::OrdinalType m_numControl = NumControls;

    static constexpr Plato::OrdinalType m_numNSPerNode = 0;

};

/******************************************************************************/
/*! Base class for simplex-based two-field thermomechanics
*/
/******************************************************************************/
template<
  Plato::OrdinalType SpaceDim,
  Plato::OrdinalType TotalDofs = SpaceDim,
  Plato::OrdinalType ProjectionDofOffset = 0,
  Plato::OrdinalType NumProjectionDof = 1,
  Plato::OrdinalType NumControls = 1>
class SimplexProjection : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::Simplex<SpaceDim>::m_numSpatialDims;

    static constexpr Plato::OrdinalType m_PDofOffset     = SpaceDim;
    static constexpr Plato::OrdinalType m_totalDofs      = TotalDofs;
    static constexpr Plato::OrdinalType m_projectionDof  = ProjectionDofOffset;
    static constexpr Plato::OrdinalType m_numDofsPerNode = SpaceDim;
    static constexpr Plato::OrdinalType m_numDofsPerCell = m_numDofsPerNode*m_numNodesPerCell;
    static constexpr Plato::OrdinalType m_numControl     = NumControls;

    // this physics can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    //
    static constexpr Plato::OrdinalType m_numNSPerNode    = NumProjectionDof;
    static constexpr Plato::OrdinalType m_numNSPerCell    = m_numNSPerNode*m_numNodesPerCell;
};


/******************************************************************************/
/*! Base class for simplex-based two-field thermomechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexStabilizedThermomechanics : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::Simplex<SpaceDim>::m_numSpatialDims;

    static constexpr Plato::OrdinalType m_numVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));

    // degree-of-freedom attributes
    //
    static constexpr Plato::OrdinalType m_PDofOffset      = SpaceDim;
    static constexpr Plato::OrdinalType m_TDofOffset      = SpaceDim + 1;
    static constexpr Plato::OrdinalType m_numDofsPerNode  = SpaceDim + 2;
    static constexpr Plato::OrdinalType m_numDofsPerCell  = m_numDofsPerNode*m_numNodesPerCell;
    static constexpr Plato::OrdinalType m_numControl      = NumControls;

    // this physics can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    //
    static constexpr Plato::OrdinalType m_numNSPerNode    = SpaceDim;
    static constexpr Plato::OrdinalType m_numNSPerCell    = m_numNSPerNode*m_numNodesPerCell;
};


} // namespace Plato

#endif
