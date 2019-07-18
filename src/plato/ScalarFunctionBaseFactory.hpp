#pragma once

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ScalarFunctionBase.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Scalar function base factory
 **********************************************************************************/
template<typename PhysicsT>
class ScalarFunctionBaseFactory
{
public:
    /******************************************************************************//**
     * @brief Constructor
     **********************************************************************************/
    ScalarFunctionBaseFactory () {}

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    ~ScalarFunctionBaseFactory() {}

    /******************************************************************************//**
     * @brief Create method
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams parameter input
     * @param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::ScalarFunctionBase> 
    create(Omega_h::Mesh& aMesh,
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap & aDataMap,
           Teuchos::ParameterList& aInputParams,
           std::string& aFunctionName);
};
// class ScalarFunctionBaseFactory


}
// namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Thermal<1>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Mechanics<1>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<1>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Electromechanics<1>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Thermomechanics<1>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Thermal<2>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Mechanics<2>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<2>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Electromechanics<2>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Thermomechanics<2>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Thermal<3>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Mechanics<3>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<3>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Electromechanics<3>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::Thermomechanics<3>>;
extern template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<3>>;
#endif
