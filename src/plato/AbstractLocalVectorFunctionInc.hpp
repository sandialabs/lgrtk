#pragma once

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class AbstractLocalVectorFunctionInc
/******************************************************************************/
{
protected:
    Omega_h::Mesh& mMesh;
    Plato::DataMap& mDataMap;
    Omega_h::MeshSets& mMeshSets;
    std::vector<std::string> mDofNames;

public:
    /******************************************************************************/
    explicit 
    AbstractLocalVectorFunctionInc( Omega_h::Mesh& aMesh, 
                                    Omega_h::MeshSets& aMeshSets,
                                    Plato::DataMap& aDataMap,
                                    std::vector<std::string> aStateNames) :
    /******************************************************************************/
            mMesh(aMesh),
            mDataMap(aDataMap),
            mMeshSets(aMeshSets),
            mDofNames(aStateNames)
    {
    }
    /******************************************************************************/
    virtual ~AbstractLocalVectorFunctionInc()
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

    /****************************************************************************//**
    * \brief Return reference to state index map
    ********************************************************************************/
    decltype(mDofNames) getDofNames() const
    {
        return (mDofNames);
    }


    /****************************************************************************//**
    * \brief Evaluate the local residual equations
    ********************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType           > & aGlobalState,
             const Plato::ScalarMultiVectorT< typename EvaluationType::PrevStateScalarType       > & aGlobalStatePrev,
             const Plato::ScalarMultiVectorT< typename EvaluationType::LocalStateScalarType      > & aLocalState,
             const Plato::ScalarMultiVectorT< typename EvaluationType::PrevLocalStateScalarType  > & aLocalStatePrev,
             const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType         > & aControl,
             const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType          > & aConfig,
             const Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType          > & aResult,
                   Plato::Scalar aTimeStep = 0.0) const = 0;

    /****************************************************************************//**
    * \brief Update the local state variables
    ********************************************************************************/
    virtual void
    updateLocalState(const Plato::ScalarMultiVector & aGlobalState,
                     const Plato::ScalarMultiVector & aGlobalStatePrev,
                     const Plato::ScalarMultiVector & aLocalState,
                     const Plato::ScalarMultiVector & aLocalStatePrev,
                     const Plato::ScalarMultiVector & aControl,
                     const Plato::ScalarArray3D     & aConfig,
                           Plato::Scalar              aTimeStep = 0.0) const = 0;
};

} // namespace Plato
