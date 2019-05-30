
#include "ScalarFunctionBaseFactory.hpp"
#include "ScalarFunctionBaseFactory_def.hpp"

template std::shared_ptr<Plato::ScalarFunctionBase> 
    Plato::ScalarFunctionBaseFactory::create<Plato::Mechanics<2>>(Omega_h::Mesh&,
                                                                   Omega_h::MeshSets& ,
                                                                   Plato::DataMap &,
                                                                   Teuchos::ParameterList&,
                                                                   const std::string);

template std::shared_ptr<Plato::ScalarFunctionBase> 
    Plato::ScalarFunctionBaseFactory::create<Plato::Mechanics<3>>(Omega_h::Mesh&,
                                                                   Omega_h::MeshSets& ,
                                                                   Plato::DataMap &,
                                                                   Teuchos::ParameterList&,
                                                                   const std::string);

template std::shared_ptr<Plato::ScalarFunctionBase> 
    Plato::ScalarFunctionBaseFactory::create<Plato::Thermomechanics<2>>(Omega_h::Mesh&,
                                                                   Omega_h::MeshSets& ,
                                                                   Plato::DataMap &,
                                                                   Teuchos::ParameterList&,
                                                                   const std::string);

template std::shared_ptr<Plato::ScalarFunctionBase> 
    Plato::ScalarFunctionBaseFactory::create<Plato::Thermomechanics<3>>(Omega_h::Mesh&,
                                                                   Omega_h::MeshSets& ,
                                                                   Plato::DataMap &,
                                                                   Teuchos::ParameterList&,
                                                                   const std::string);

template std::shared_ptr<Plato::ScalarFunctionBase> 
    Plato::ScalarFunctionBaseFactory::create<Plato::Electromechanics<2>>(Omega_h::Mesh&,
                                                                   Omega_h::MeshSets& ,
                                                                   Plato::DataMap &,
                                                                   Teuchos::ParameterList&,
                                                                   const std::string);

template std::shared_ptr<Plato::ScalarFunctionBase> 
    Plato::ScalarFunctionBaseFactory::create<Plato::Electromechanics<3>>(Omega_h::Mesh&,
                                                                   Omega_h::MeshSets& ,
                                                                   Plato::DataMap &,
                                                                   Teuchos::ParameterList&,
                                                                   const std::string);

template std::shared_ptr<Plato::ScalarFunctionBase> 
    Plato::ScalarFunctionBaseFactory::create<Plato::Thermal<2>>(Omega_h::Mesh&,
                                                                   Omega_h::MeshSets& ,
                                                                   Plato::DataMap &,
                                                                   Teuchos::ParameterList&,
                                                                   const std::string);

template std::shared_ptr<Plato::ScalarFunctionBase> 
    Plato::ScalarFunctionBaseFactory::create<Plato::Thermal<3>>(Omega_h::Mesh&,
                                                                   Omega_h::MeshSets& ,
                                                                   Plato::DataMap &,
                                                                   Teuchos::ParameterList&,
                                                                   const std::string);