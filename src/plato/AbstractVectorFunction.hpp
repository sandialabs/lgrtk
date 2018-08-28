#ifndef ABSTRACT_VECTOR_FUNCTION_HPP
#define ABSTRACT_VECTOR_FUNCTION_HPP

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunction
/******************************************************************************/
{
protected:
    Omega_h::Mesh& mMesh;
    Plato::DataMap& m_dataMap;
    Omega_h::MeshSets& mMeshSets;

public:
    /******************************************************************************/
    explicit AbstractVectorFunction(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap& aDataMap) :
    /******************************************************************************/
            mMesh(aMesh),
            m_dataMap(aDataMap),
            mMeshSets(aMeshSets)
    {
    }
    /******************************************************************************/
    virtual ~AbstractVectorFunction()
    /******************************************************************************/
    {
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh data base 
    ********************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh sets 
    ********************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /******************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> & aState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> & aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;
    /******************************************************************************/
};

#endif
