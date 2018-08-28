#ifndef ABSTRACT_SCALAR_FUNCTION
#define ABSTRACT_SCALAR_FUNCTION

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
template<typename EvaluationType>
class AbstractScalarFunction
/******************************************************************************/
{
protected:
    Omega_h::Mesh& mMesh;
    Plato::DataMap& m_dataMap;
    Omega_h::MeshSets& mMeshSets;

    const std::string m_functionName;

 
public:
    AbstractScalarFunction(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap& aDataMap, std::string name) :
            mMesh(aMesh),
            m_dataMap(aDataMap),
            mMeshSets(aMeshSets),
            m_functionName(name)
    {
    }

    virtual ~AbstractScalarFunction()
    {
    }

    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> & aState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> & aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;

    virtual void postEvaluate(Plato::ScalarVector, Plato::Scalar)
    {
    }

    virtual void postEvaluate(Plato::Scalar&)
    {
    }

    const decltype(m_functionName)& getName()
    {
        return m_functionName;
    }
};

#endif
