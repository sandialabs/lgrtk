#pragma once

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Scalar function base class
 **********************************************************************************/
class ScalarFunctionBase
{
public:
    virtual ~ScalarFunctionBase(){}

    /******************************************************************************//**
     * @brief Return function name
     * @return user defined function name
     **********************************************************************************/
    virtual std::string name() const = 0;

    /******************************************************************************//**
     * @brief Return function value
     * @param [in] aState state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function value
     **********************************************************************************/
    virtual Plato::Scalar value(const Plato::ScalarVector & aState,
                                const Plato::ScalarVector & aControl,
                                Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Return function gradient wrt design variables
     * @param [in] aState state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function gradient wrt design variables
     **********************************************************************************/
    virtual Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Return function gradient wrt state variables
     * @param [in] aState state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function gradient wrt state variables
     **********************************************************************************/
    virtual Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Return function gradient wrt configurtion variables
     * @param [in] aState state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function gradient wrt configurtion variables
     **********************************************************************************/
    virtual Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarVector & aState, 
                               const Plato::ScalarVector & aControl) const = 0;
};
// class ScalarFunctionBase


}
// namespace Plato