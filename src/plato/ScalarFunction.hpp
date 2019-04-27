#ifndef LGR_PLATO_SCALAR_FUNCTION_HPP
#define LGR_PLATO_SCALAR_FUNCTION_HPP

#include <memory>
#include <cassert>

#include <Omega_h_mesh.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"

/******************************************************************************/
/*! objective class

 This class takes as a template argument a scalar function in the form:

 and manages the evaluation of the function and derivatives wrt state
 and control. status

 */
/******************************************************************************/
template<typename PhysicsT>
class ScalarFunction : public WorksetBase<PhysicsT>
{
private:
    using WorksetBase<PhysicsT>::m_numDofsPerCell; /*!< number of degree of freedom per cell/element */
    using WorksetBase<PhysicsT>::m_numNodesPerCell; /*!< number of nodes per cell/element */
    using WorksetBase<PhysicsT>::m_numDofsPerNode; /*!< number of degree of freedom per node */
    using WorksetBase<PhysicsT>::m_numSpatialDims; /*!< number of spatial dimensions */
    using WorksetBase<PhysicsT>::m_numControl; /*!< number of control variables */
    using WorksetBase<PhysicsT>::m_numNodes; /*!< total number of nodes in the mesh */
    using WorksetBase<PhysicsT>::m_numCells; /*!< total number of cells/elements in the mesh */

    using WorksetBase<PhysicsT>::m_stateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using WorksetBase<PhysicsT>::m_controlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using WorksetBase<PhysicsT>::m_configEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< result variables automatic differentiation type */
    using Jacobian = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian; /*!< state variables automatic differentiation type */
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX; /*!< configuration variables automatic differentiation type */
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ; /*!< control variables automatic differentiation type */

    std::shared_ptr<Plato::AbstractScalarFunction<Residual>> mScalarFunctionValue; /*!< scalar function value interface */
    std::shared_ptr<Plato::AbstractScalarFunction<Jacobian>> mScalarFunctionGradientU; /*!< scalar function value partial wrt states */
    std::shared_ptr<Plato::AbstractScalarFunction<GradientX>> mScalarFunctionGradientX; /*!< scalar function value partial wrt configuration */
    std::shared_ptr<Plato::AbstractScalarFunction<GradientZ>> mScalarFunctionGradientZ; /*!< scalar function value partial wrt controls */

    Plato::DataMap& m_dataMap; /*!< PLATO Engine and Analyze data map */

public:
    /******************************************************************************//**
     * @brief Primary scalar function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aMyName scalar function name
    **********************************************************************************/
    ScalarFunction(Omega_h::Mesh& aMesh,
                   Omega_h::MeshSets& aMeshSets,
                   Plato::DataMap & aDataMap,
                   Teuchos::ParameterList& aInputParams,
                   const std::string & aFuncName) :
            WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap)
    {
        typename PhysicsT::FunctionFactory tFactory;

        mScalarFunctionValue =
                tFactory.template createScalarFunction<Residual>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);

        mScalarFunctionGradientU =
                tFactory.template createScalarFunction<Jacobian>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);

        mScalarFunctionGradientZ =
                tFactory.template createScalarFunction<GradientZ>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);

        mScalarFunctionGradientX =
                tFactory.template createScalarFunction<GradientX>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);
    }

    /******************************************************************************//**
     * @brief Secondary scalar function constructor, used for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    ScalarFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            WorksetBase<PhysicsT>(aMesh),
            mScalarFunctionValue(nullptr),
            mScalarFunctionGradientU(nullptr),
            mScalarFunctionGradientX(nullptr),
            mScalarFunctionGradientZ(nullptr),
            m_dataMap(aDataMap)
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
        WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        Plato::ScalarMultiVector tControlWS("control workset", m_numCells, m_numNodesPerCell);
        WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        Plato::ScalarArray3D tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        mScalarFunctionValue->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientU->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientZ->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientX->updateProblem(tStateWS, tControlWS, tConfigWS);
    }

    /******************************************************************************//**
     * @brief Evaluate scalar function
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar function evaluation
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

        WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);

        WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result", m_numCells);

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
     * @brief Evaluate gradient of the scalar function with respect to (wrt) the configuration parameters
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the configuration parameters
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
        WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result", m_numCells);

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
     * @brief Evaluate gradient of the scalar function with respect to (wrt) the state variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the state variables
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
        WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result", m_numCells);

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
     * @brief Evaluate gradient of the scalar function with respect to (wrt) the control variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the control variables
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
        WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", m_numCells, m_numDofsPerCell);
        WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result
        //
        Plato::ScalarVectorT<ResultScalar> tResult("elastic energy", m_numCells);

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
};

#endif
