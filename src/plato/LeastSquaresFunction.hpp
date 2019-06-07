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
 * @brief Least Squares function class \f$ F(x) = \sum_{i = 1}^{n} w_i * (f_i(x) - gold_i(x))^2 \f$
 **********************************************************************************/
template<typename PhysicsT>
class LeastSquaresFunction : public Plato::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
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
    std::vector<Plato::Scalar> mFunctionGoldValues; /*!< Vector of function gold values */
    std::vector<Plato::Scalar> mFunctionNormalization; /*!< Vector of function normalization values */
    std::vector<std::shared_ptr<Plato::ScalarFunctionBase>> mScalarFunctionBaseContainer; /*!< Vector of ScalarFunctionBase objects */

    Plato::DataMap& m_dataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

    const Plato::Scalar mFunctionNormalizationCutoff = 0.1;

	/******************************************************************************//**
     * @brief Initialization of Least Squares Function
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        Plato::ScalarFunctionBaseFactory<PhysicsT> tFactory;

        mScalarFunctionBaseContainer.clear();
        mFunctionWeights.clear();
        mFunctionGoldValues.clear();
        mFunctionNormalization.clear();

        auto tProblemFunctionName = aInputParams.sublist(mFunctionName);

        auto tFunctionNamesTeuchos = tProblemFunctionName.get<Teuchos::Array<std::string>>("Functions");
        auto tFunctionWeightsTeuchos = tProblemFunctionName.get<Teuchos::Array<double>>("Weights");
        auto tFunctionGoldValuesTeuchos = tProblemFunctionName.get<Teuchos::Array<double>>("Gold Values");

        auto tFunctionNames      = tFunctionNamesTeuchos.toVector();
        auto tFunctionWeights    = tFunctionWeightsTeuchos.toVector();
        auto tFunctionGoldValues = tFunctionGoldValuesTeuchos.toVector();

        if (tFunctionNames.size() != tFunctionWeights.size())
        {
            const std::string tErrorString = std::string("Number of 'Functions' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Weights'";
            throw std::runtime_error(tErrorString);
        }

        if (tFunctionNames.size() != tFunctionGoldValues.size())
        {
            const std::string tErrorString = std::string("Number of 'Gold Values' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Functions'";
            throw std::runtime_error(tErrorString);
        }

        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer.push_back(
                tFactory.create(
                    aMesh, aMeshSets, m_dataMap, aInputParams, tFunctionNames[tFunctionIndex]));
            mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
            mFunctionGoldValues.push_back(tFunctionGoldValues[tFunctionIndex]);
            
            if (tFunctionGoldValues[tFunctionIndex] > mFunctionNormalizationCutoff)
                mFunctionNormalization.push_back(tFunctionGoldValues[tFunctionIndex]);
            else
                mFunctionNormalization.push_back(1.0);
        }

    }

public:
    /******************************************************************************//**
     * @brief Primary least squares function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    LeastSquaresFunction(Omega_h::Mesh& aMesh,
                Omega_h::MeshSets& aMeshSets,
                Plato::DataMap & aDataMap,
                Teuchos::ParameterList& aInputParams,
                const std::string aName) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap),
            mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Secondary least squares function constructor, used for unit testing / mass properties
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    LeastSquaresFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap),
            mFunctionName("Least Squares")
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
     * @brief Add function gold value
     * @param [in] aWGoldValue function gold value
    **********************************************************************************/
    void appendGoldFunctionValue(Plato::Scalar aGoldValue)
    {
        mFunctionGoldValues.push_back(aGoldValue);
        if (aGoldValue > mFunctionNormalizationCutoff)
            mFunctionNormalization.push_back(aGoldValue);
        else
            mFunctionNormalization.push_back(1.0);
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
     * @brief Evaluate least squares function
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aState,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    {
        assert(mFunctionWeights.size() == mScalarFunctionBaseContainer.size());
        assert(mFunctionGoldValues.size() == mScalarFunctionBaseContainer.size());
        assert(mFunctionNormalization.size() == mScalarFunctionBaseContainer.size());

        Plato::Scalar tResult = 0.0;
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
            const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aState, aControl, aTimeStep);
            tResult += tFunctionWeight * 
                       std::pow((tFunctionValue - tFunctionGoldValue) / tFunctionScale, 2);
        }
        return tResult;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the least squares function with respect to (wrt) the configuration parameters
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
            const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
            const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aState, aControl, aTimeStep);
            Plato::ScalarVector tFunctionGradX = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_x(aState, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientX(tDof) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                                        * tFunctionGradX(tDof) / (tFunctionScale * tFunctionScale);
            },"Least Squares Function Summation Grad X");
        }
        return tGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the least squares function with respect to (wrt) the state variables
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
            const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
            const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aState, aControl, aTimeStep);
            Plato::ScalarVector tFunctionGradU = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_u(aState, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientU(tDof) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                                        * tFunctionGradU(tDof) / (tFunctionScale * tFunctionScale);
            },"Least Squares Function Summation Grad U");
        }
        return tGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the least squares function with respect to (wrt) the control variables
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
            const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
            const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aState, aControl, aTimeStep);
            Plato::ScalarVector tFunctionGradZ = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_z(aState, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientZ(tDof) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                                        * tFunctionGradZ(tDof) / (tFunctionScale * tFunctionScale);
            },"Least Squares Function Summation Grad Z");
        }
        return tGradientZ;
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
// class LeastSquaresFunction

}
//namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::LeastSquaresFunction<::Plato::Thermal<1>>;
extern template class Plato::LeastSquaresFunction<::Plato::Mechanics<1>>;
extern template class Plato::LeastSquaresFunction<::Plato::Electromechanics<1>>;
extern template class Plato::LeastSquaresFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::LeastSquaresFunction<::Plato::Thermal<2>>;
extern template class Plato::LeastSquaresFunction<::Plato::Mechanics<2>>;
extern template class Plato::LeastSquaresFunction<::Plato::Electromechanics<2>>;
extern template class Plato::LeastSquaresFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::LeastSquaresFunction<::Plato::Thermal<3>>;
extern template class Plato::LeastSquaresFunction<::Plato::Mechanics<3>>;
extern template class Plato::LeastSquaresFunction<::Plato::Electromechanics<3>>;
extern template class Plato::LeastSquaresFunction<::Plato::Thermomechanics<3>>;
#endif