#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "plato/ScalarFunctionBase.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Physics scalar function class 
 **********************************************************************************/
template<typename PhysicsT>
class PhysicsScalarFunction : public ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::m_numDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_numNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_numDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::m_numSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::m_numControl; /*!< number of control variables */
    using Plato::WorksetBase<PhysicsT>::m_numNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<PhysicsT>::m_numCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<PhysicsT>::m_stateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_controlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_configEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< result variables automatic differentiation type */
    using Jacobian = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian; /*!< state variables automatic differentiation type */
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX; /*!< configuration variables automatic differentiation type */
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ; /*!< control variables automatic differentiation type */

    std::shared_ptr<Plato::AbstractScalarFunction<Residual>> mScalarFunctionValue; /*!< scalar function value interface */
    std::shared_ptr<Plato::AbstractScalarFunction<Jacobian>> mScalarFunctionGradientU; /*!< scalar function value partial wrt states */
    std::shared_ptr<Plato::AbstractScalarFunction<GradientX>> mScalarFunctionGradientX; /*!< scalar function value partial wrt configuration */
    std::shared_ptr<Plato::AbstractScalarFunction<GradientZ>> mScalarFunctionGradientZ; /*!< scalar function value partial wrt controls */

    Plato::DataMap& m_dataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName;/*!< User defined function name */

	/******************************************************************************//**
     * @brief Initialization of Physics Scalar Function
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsT::FunctionFactory tFactory;

        auto tProblemDefault = aInputParams.sublist(mFunctionName);
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", ""); // Must be a hardcoded type name (e.g. Volume)

        mScalarFunctionValue =
            tFactory.template createScalarFunction<Residual>(
                aMesh, aMeshSets, m_dataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientU =
            tFactory.template createScalarFunction<Jacobian>(
                aMesh, aMeshSets, m_dataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientX =
            tFactory.template createScalarFunction<GradientX>(
                aMesh, aMeshSets, m_dataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientZ =
            tFactory.template createScalarFunction<GradientZ>(
                aMesh, aMeshSets, m_dataMap, aInputParams, tFunctionType, mFunctionName);
    }

public:
    /******************************************************************************//**
     * @brief Primary physics scalar function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(Omega_h::Mesh& aMesh,
            Omega_h::MeshSets& aMeshSets,
            Plato::DataMap & aDataMap,
            Teuchos::ParameterList& aInputParams,
            std::string& aName) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap),
            mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Secondary physics scalar function constructor, used for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    PhysicsScalarFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mScalarFunctionValue(),
            mScalarFunctionGradientU(),
            mScalarFunctionGradientX(),
            mScalarFunctionGradientZ(),
            m_dataMap(aDataMap),
            mFunctionName("Undefined Name")
    {
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateValue(const std::shared_ptr<Plato::AbstractScalarFunction<Residual>>& aInput)
    {
        mScalarFunctionValue = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the Jacobian automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientU(const std::shared_ptr<Plato::AbstractScalarFunction<Jacobian>>& aInput)
    {
        mScalarFunctionGradientU = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientZ automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientZ(const std::shared_ptr<Plato::AbstractScalarFunction<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientX automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientX(const std::shared_ptr<Plato::AbstractScalarFunction<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = aInput;
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        Plato::ScalarMultiVector tStateWS("state workset", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        Plato::ScalarMultiVector tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        Plato::ScalarArray3D tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        mScalarFunctionValue->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientU->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientZ->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientX->updateProblem(tStateWS, tControlWS, tConfigWS);
    }

    /******************************************************************************//**
     * @brief Evaluate physics scalar function
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aState,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    {
        using ConfigScalar = typename Residual::ConfigScalarType;
        using StateScalar = typename Residual::StateScalarType;
        using ControlScalar = typename Residual::ControlScalarType;
        using ResultScalar = typename Residual::ResultScalarType;

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", m_numCells);
        m_dataMap.scalarVectors[mScalarFunctionValue->getName()] = tResult;

        // evaluate function
        //
        mScalarFunctionValue->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

        // sum across elements
        //
        auto tReturnVal = Plato::local_result_sum<Plato::Scalar>(m_numCells, tResult);

        mScalarFunctionValue->postEvaluate(tReturnVal);

        return tReturnVal;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        using ConfigScalar = typename GradientX::ConfigScalarType;
        using StateScalar = typename GradientX::StateScalarType;
        using ControlScalar = typename GradientX::ControlScalarType;
        using ResultScalar = typename GradientX::ResultScalarType;

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", m_numCells);

        // evaluate function
        //
        mScalarFunctionGradientX->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientX("objective gradient configuration", m_numSpatialDims * m_numNodes);
        Plato::assemble_vector_gradient<m_numNodesPerCell, m_numSpatialDims>(m_numCells,
                                                                             m_configEntryOrdinal,
                                                                             tResult,
                                                                             tObjGradientX);

        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);
        mScalarFunctionGradientX->postEvaluate(tObjGradientX, tObjectiveValue);

        return tObjGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        using ConfigScalar = typename Jacobian::ConfigScalarType;
        using StateScalar = typename Jacobian::StateScalarType;
        using ControlScalar = typename Jacobian::ControlScalarType;
        using ResultScalar = typename Jacobian::ResultScalarType;

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("sacado-ized state", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", m_numCells);

        // evaluate function
        //
        mScalarFunctionGradientU->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", m_numDofsPerNode * m_numNodes);
        Plato::assemble_vector_gradient<m_numNodesPerCell, m_numDofsPerNode>(m_numCells,
                                                                             m_stateEntryOrdinal,
                                                                             tResult,
                                                                             tObjGradientU);
        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);
        mScalarFunctionGradientU->postEvaluate(tObjGradientU, tObjectiveValue);
        return tObjGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {        
        using ConfigScalar = typename GradientZ::ConfigScalarType;
        using StateScalar = typename GradientZ::StateScalarType;
        using ControlScalar = typename GradientZ::ControlScalarType;
        using ResultScalar = typename GradientZ::ResultScalarType;

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", m_numCells);

        // evaluate function
        //
        mScalarFunctionGradientZ->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);


        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control", m_numNodes);
        Plato::assemble_scalar_gradient<m_numNodesPerCell>(m_numCells, m_controlEntryOrdinal, tResult, tObjGradientZ);

        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);
        mScalarFunctionGradientZ->postEvaluate(tObjGradientZ, tObjectiveValue);
        return tObjGradientZ;
    }

    /******************************************************************************//**
     * @brief Set user defined function name
     * @param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * @brief Return user defined function name
     * @return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
//class PhysicsScalarFunction

}
//namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::PhysicsScalarFunction<::Plato::Thermal<1>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Mechanics<1>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Electromechanics<1>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::PhysicsScalarFunction<::Plato::Thermal<2>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Mechanics<2>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Electromechanics<2>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::PhysicsScalarFunction<::Plato::Thermal<3>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Mechanics<3>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Electromechanics<3>>;
extern template class Plato::PhysicsScalarFunction<::Plato::Thermomechanics<3>>;
#endif