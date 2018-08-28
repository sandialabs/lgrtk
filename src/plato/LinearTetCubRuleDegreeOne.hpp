/*
 * LinearTetCubRuleDegreeOne.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef LINEARTETCUBRULEDEGREEONE_HPP_
#define LINEARTETCUBRULEDEGREEONE_HPP_

#include <string>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class LinearTetCubRuleDegreeOne
/******************************************************************************/
{
public:
    /******************************************************************************/
    LinearTetCubRuleDegreeOne() :
            mCubWeight(1.0),
            mNumCubPoints(1),
            mBasisFunctions(),
            mCubPointsCoords()
    /******************************************************************************/
    {
        this->initialize();
    }
    /******************************************************************************/
    ~LinearTetCubRuleDegreeOne()
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION Plato::Scalar getCubWeight() const
    /******************************************************************************/
    {
        return (mCubWeight);
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION Plato::OrdinalType getNumCubPoints() const
    /******************************************************************************/
    {
        return (mNumCubPoints);
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION const Plato::ScalarVector & getBasisFunctions() const
    /******************************************************************************/
    {
        return (mBasisFunctions);
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION const Plato::ScalarVector & getCubPointsCoords() const
    /******************************************************************************/
    {
        return (mCubPointsCoords);
    }

private:
    /****************************************************************************//**/
    /*!
     * For /f$dim == 1/f$, we shift the reference cell from the /f$[-1,1]/f$ interval
     * to a /f$[0,1]/f$ interval. which is consistent with our simplex treatment in
     * higher dimensions. Therefore, the coordinates are transformed by /f$x\rightarrow
     * \frac{x + 1.0}{2.0}/f$ and the weights are cut in half.
     *
     ********************************************************************************/
    void initialize()
    {
        // set gauss weight
        for(Plato::OrdinalType tDimIndex=2; tDimIndex<=SpaceDim; tDimIndex++)
        { 
            mCubWeight /= Plato::Scalar(tDimIndex);
        }

        // initialize array with cubature points coordinates
        std::string tName = "Tet4: Degree One Cubature Rule - Coordinates View";
        mCubPointsCoords = decltype(mCubPointsCoords)(Kokkos::ViewAllocateWithoutInitializing(tName), SpaceDim);
        auto tHostCubPointsCoords = Kokkos::create_mirror(mCubPointsCoords);
        if(SpaceDim == static_cast<Plato::OrdinalType>(3))
        {
            tHostCubPointsCoords(0) = static_cast<Plato::Scalar>(0.25);
            tHostCubPointsCoords(1) = static_cast<Plato::Scalar>(0.25);
            tHostCubPointsCoords(2) = static_cast<Plato::Scalar>(0.25);
        }
        else if(SpaceDim == static_cast<Plato::OrdinalType>(2))
        {
            tHostCubPointsCoords(0) = static_cast<Plato::Scalar>(1.0/3.0);
            tHostCubPointsCoords(1) = static_cast<Plato::Scalar>(1.0/3.0);
        }
        else
        {
            tHostCubPointsCoords(0) = static_cast<Plato::Scalar>(1.0/2.0);
        }

        // initialize array with basis functions
        tName = "Tet4: Degree One Cubature Rule - Basis Functions View";
        auto tNumNodesPerCell = SpaceDim + static_cast<Plato::OrdinalType>(1);
        mBasisFunctions = decltype(mBasisFunctions)(Kokkos::ViewAllocateWithoutInitializing(tName), tNumNodesPerCell);
        auto tHostBasisFunctions = Kokkos::create_mirror(mBasisFunctions);
        if(SpaceDim == static_cast<Plato::OrdinalType>(3))
        {
            tHostBasisFunctions(0) = static_cast<Plato::Scalar>(1) - tHostCubPointsCoords(0)
                    - tHostCubPointsCoords(1) - tHostCubPointsCoords(2);
            tHostBasisFunctions(1) = tHostCubPointsCoords(0);
            tHostBasisFunctions(2) = tHostCubPointsCoords(1);
            tHostBasisFunctions(3) = tHostCubPointsCoords(2);
        }
        else if(SpaceDim == static_cast<Plato::OrdinalType>(2))
        {
            tHostBasisFunctions(0) = static_cast<Plato::Scalar>(1) - tHostCubPointsCoords(0) - tHostCubPointsCoords(1);
            tHostBasisFunctions(1) = tHostCubPointsCoords(0);
            tHostBasisFunctions(2) = tHostCubPointsCoords(1);
        }
        else
        {
            tHostBasisFunctions(0) = static_cast<Plato::Scalar>(1) - tHostCubPointsCoords(0);
            tHostBasisFunctions(1) = tHostCubPointsCoords(0);
        }

        Kokkos::deep_copy(mBasisFunctions, tHostBasisFunctions);
        Kokkos::deep_copy(mCubPointsCoords, tHostCubPointsCoords);
    }

private:
    Plato::Scalar mCubWeight;
    Plato::OrdinalType mNumCubPoints;
    Plato::ScalarVector mBasisFunctions;
    Plato::ScalarVector mCubPointsCoords;
};
// class LinearTetCubRuleDegreeOne

} // namespace Plato

#endif /* LINEARTETCUBRULEDEGREEONE_HPP_ */
