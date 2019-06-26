#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ScalarFunctionBaseFactory.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Weighted sum function class \f$ F(x) = \sum_{i = 1}^{n} w_i * f_i(x) \f$
 **********************************************************************************/
template<typename PhysicsT>
class WeightedSumFunction : public Plato::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
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

    std::vector<Plato::Scalar> mFunctionWeights; /*!< Vector of function weights */
    std::vector<std::shared_ptr<Plato::ScalarFunctionBase>> mScalarFunctionBaseContainer; /*!< Vector of ScalarFunctionBase objects */

    Plato::DataMap& m_dataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

	/******************************************************************************//**
     * @brief Initialization of Weighted Sum Function
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        Plato::ScalarFunctionBaseFactory<PhysicsT> tFactory;

        mScalarFunctionBaseContainer.clear();
        mFunctionWeights.clear();

        auto tProblemFunctionName = aInputParams.sublist(mFunctionName);

        auto tFunctionNamesTeuchos = tProblemFunctionName.get<Teuchos::Array<std::string>>("Functions");
        auto tFunctionWeightsTeuchos = tProblemFunctionName.get<Teuchos::Array<double>>("Weights");

        auto tFunctionNames = tFunctionNamesTeuchos.toVector();
        auto tFunctionWeights = tFunctionWeightsTeuchos.toVector();

        if (tFunctionNames.size() != tFunctionWeights.size())
        {
            const std::string tErrorString = std::string("Number of 'Functions' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Weights'";
            throw std::runtime_error(tErrorString);
        }

        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer.push_back(
                tFactory.create(
                    aMesh, aMeshSets, m_dataMap, aInputParams, tFunctionNames[tFunctionIndex]));
            mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
        }

    }

public:
    /******************************************************************************//**
     * @brief Primary weight sum function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    WeightedSumFunction(Omega_h::Mesh& aMesh,
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
     * @brief Secondary weight sum function constructor, used for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    WeightedSumFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap),
            mFunctionName("Weighted Sum")
    {
    }

    /******************************************************************************//**
     * @brief Add function weight
     * @param [in] aWeight function weight
    **********************************************************************************/
    void appendFunctionWeight(Plato::Scalar aWeight)
    {
        mFunctionWeights.push_back(aWeight);
    }

    /******************************************************************************//**
     * @brief Allocate scalar function base using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateScalarFunctionBase(const std::shared_ptr<Plato::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseContainer.push_back(aInput);
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer[tFunctionIndex]->updateProblem(aState, aControl);
        }
    }

    /******************************************************************************//**
     * @brief Evaluate weight sum function
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aState,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    {
        assert(mScalarFunctionBaseContainer.size() == mFunctionWeights.size());
        
        Plato::Scalar tResult = 0.0;
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aState, aControl, aTimeStep);
            tResult += tFunctionWeight * tFunctionValue;
        }
        // printf("%s value %f \n", mFunctionName.c_str(), tResult);
        return tResult;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the weight sum function with respect to (wrt) the configuration parameters
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        const Plato::OrdinalType tNumDofs = m_numSpatialDims * m_numNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradX = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_x(aState, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientX(tDof) += tFunctionWeight * tFunctionGradX(tDof);
            },"Weighted Sum Function Summation Grad X");
        }
        return tGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the weight sum function with respect to (wrt) the state variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        const Plato::OrdinalType tNumDofs = m_numDofsPerNode * m_numNodes;
        Plato::ScalarVector tGradientU ("gradient state", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradU = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_u(aState, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientU(tDof) += tFunctionWeight * tFunctionGradU(tDof);
            },"Weighted Sum Function Summation Grad U");
        }
        return tGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the weight sum function with respect to (wrt) the control variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        const Plato::OrdinalType tNumDofs = m_numNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradZ = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_z(aState, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientZ(tDof) += tFunctionWeight * tFunctionGradZ(tDof);
            },"Weighted Sum Function Summation Grad Z");
        }
        return tGradientZ;
    }

    /******************************************************************************//**
     * @brief Set function name
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
// class WeightedSumFunction

}
//namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::WeightedSumFunction<::Plato::Thermal<1>>;
extern template class Plato::WeightedSumFunction<::Plato::Mechanics<1>>;
extern template class Plato::WeightedSumFunction<::Plato::Electromechanics<1>>;
extern template class Plato::WeightedSumFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::WeightedSumFunction<::Plato::Thermal<2>>;
extern template class Plato::WeightedSumFunction<::Plato::Mechanics<2>>;
extern template class Plato::WeightedSumFunction<::Plato::Electromechanics<2>>;
extern template class Plato::WeightedSumFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::WeightedSumFunction<::Plato::Thermal<3>>;
extern template class Plato::WeightedSumFunction<::Plato::Mechanics<3>>;
extern template class Plato::WeightedSumFunction<::Plato::Electromechanics<3>>;
extern template class Plato::WeightedSumFunction<::Plato::Thermomechanics<3>>;
#endif