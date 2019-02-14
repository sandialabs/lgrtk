#ifndef ABSTRACT_SCALAR_FUNCTION
#define ABSTRACT_SCALAR_FUNCTION

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************//**
 * @brief Abstract scalar function (i.e. criterion) interface
**********************************************************************************/
template<typename EvaluationType>
class AbstractScalarFunction
{
protected:
    Omega_h::Mesh& mMesh; /*!< mesh database */
    Plato::DataMap& m_dataMap; /*!< PLATO Engine and PLATO Analyze data map - enables inputs from PLATO Engine */
    Omega_h::MeshSets& mMeshSets; /*!< mesh side sets database */

    const std::string m_functionName; /*!< my abstract scalar function name */

 
public:
    /******************************************************************************//**
     * @brief Abstract scalar function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets mesh side sets database
     * @param [in] aDataMap PLATO Engine and PLATO Analyze data map
     * @param [in] aName my abstract scalar function name
    **********************************************************************************/
    AbstractScalarFunction(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap& aDataMap,
                           const std::string & aName) :
            mMesh(aMesh),
            m_dataMap(aDataMap),
            mMeshSets(aMeshSets),
            m_functionName(aName)
    {
    }

    /******************************************************************************//**
     * @brief Abstract scalar function destructor
    **********************************************************************************/
    virtual ~AbstractScalarFunction(){}

    /******************************************************************************//**
     * @brief Evaluate abstract scalar function
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> & aState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> & aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Update physics-based data in between optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarMultiVector & aState,
                               const Plato::ScalarMultiVector & aControl,
                               const Plato::ScalarArray3D & aConfig)
    { return; }

    /******************************************************************************//**
     * @brief Get abstract scalar function evaluation and total gradient
    **********************************************************************************/
    virtual void postEvaluate(Plato::ScalarVector, Plato::Scalar)
    { return; }

    /******************************************************************************//**
     * @brief Get abstract scalar function evaluation
     * @param [out] aOutput scalar function evaluation
    **********************************************************************************/
    virtual void postEvaluate(Plato::Scalar& aOutput)
    { return; }

    /******************************************************************************//**
     * @brief Return abstract scalar function name
     * @return name
    **********************************************************************************/
    const decltype(m_functionName)& getName()
    {
        return m_functionName;
    }
};

#endif
