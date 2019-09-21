#ifndef ABSTRACT_VECTOR_FUNCTION_HPP
#define ABSTRACT_VECTOR_FUNCTION_HPP

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Abstract vector function (i.e. PDE) interface
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 **********************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunction
{
protected:
    Omega_h::Mesh& mMesh; /*!< volume mesh database */
    Plato::DataMap& mDataMap; /*!< PLATO Analyze database */
    Omega_h::MeshSets& mMeshSets;  /*!< surface mesh database */

public:
    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aMesh volume mesh database
     * @param [in] aMeshSets surface mesh database
     * @param [in] aDataMap PLATO Analyze database
    **********************************************************************************/
    explicit AbstractVectorFunction(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap& aDataMap) :
            mMesh(aMesh),
            mDataMap(aDataMap),
            mMeshSets(aMeshSets)
    {
    }

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    virtual ~AbstractVectorFunction()
    {
    }

    /****************************************************************************//**
    * @brief Return reference to Omega_h mesh database
    * @return volume mesh database
    ********************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /****************************************************************************//**
    * @brief Return reference to Omega_h mesh sets
    * @return surface mesh database
    ********************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /******************************************************************************//**
     * @brief Evaluate vector function
     * @param [in] aState 2D array with state variables (C,DOF)
     * @param [in] aControl 2D array with control variables (C,N)
     * @param [in] aConfig 3D array with control variables (C,N,D)
     * @param [in] aResult 1D array with control variables (C,DOF)
     * @param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> & aState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> & aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;
};
// class AbstractVectorFunction

} // namespace Plato

#endif
