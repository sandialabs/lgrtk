#ifndef COMPUTED_FIELD_HPP
#define COMPUTED_FIELD_HPP

#include <Omega_h_expr.hpp>
#include <Omega_h_mesh.hpp>

#include <Teuchos_ParameterList.hpp>

#include "ImplicitFunctors.hpp"

namespace Plato
{

/******************************************************************************/
/*!
  \brief Class for computed fields.
*/
template<int SpaceDim>
class ComputedField
/******************************************************************************/
{
  protected:
    const std::string   m_name;
    const std::string   m_funcString;
    Plato::ScalarVector m_values;
  
  public:
  
  /**************************************************************************/
  ComputedField<SpaceDim>(
    const Omega_h::Mesh& aMesh, 
    const std::string &aName, 
    const std::string &aFunc) :
    m_name(aName),
    m_funcString(aFunc),
    m_values(aName, aMesh.nverts())
  /**************************************************************************/
  {
    initialize(aMesh, aName, aFunc);
  }

  /**************************************************************************/
  void initialize(
    const Omega_h::Mesh& aMesh, 
    const std::string &aName, 
    const std::string &aFunc)
  /**************************************************************************/
  {
    auto numPoints = aMesh.nverts();
    auto x_coords = Plato::getArray_Omega_h<Plato::Scalar>("x coords", numPoints);
    auto y_coords = Plato::getArray_Omega_h<Plato::Scalar>("y coords", numPoints);
    auto z_coords = Plato::getArray_Omega_h<Plato::Scalar>("z coords", numPoints);
  
    auto coords = aMesh.coords();
    auto values = m_values;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numPoints), LAMBDA_EXPRESSION(Plato::OrdinalType aPointOrdinal)
    {
      if (SpaceDim > 0) x_coords[aPointOrdinal] = coords[aPointOrdinal*SpaceDim + 0];
      if (SpaceDim > 1) y_coords[aPointOrdinal] = coords[aPointOrdinal*SpaceDim + 1];
      if (SpaceDim > 2) z_coords[aPointOrdinal] = coords[aPointOrdinal*SpaceDim + 2];
    },"fill coords");

    Omega_h::ExprReader reader(numPoints, SpaceDim);
    if (SpaceDim > 0) reader.register_variable("x", Omega_h::any(Omega_h::Reals(x_coords)));
    if (SpaceDim > 1) reader.register_variable("y", Omega_h::any(Omega_h::Reals(y_coords)));
    if (SpaceDim > 2) reader.register_variable("z", Omega_h::any(Omega_h::Reals(z_coords)));
  
    auto result = reader.read_string(m_funcString, "Value");
    reader.repeat(result);
    Omega_h::Reals fxnValues = Omega_h::any_cast<Omega_h::Reals>(result);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numPoints), LAMBDA_EXPRESSION(Plato::OrdinalType aPointOrdinal)
    {
      values(aPointOrdinal) = fxnValues[aPointOrdinal];
    },"copy result");
  }

  ~ComputedField(){}
  
  /******************************************************************************/
  void get(Plato::ScalarVector aValues)
  /******************************************************************************/
  {
    Kokkos::deep_copy( aValues, m_values );
  }

  /******************************************************************************/
  const decltype(m_name)& name() { return m_name; }
  /******************************************************************************/

}; // end class BodyLoad


/******************************************************************************/
/*!
  \brief Owner class that contains a vector of BodyLoad objects.
*/
template<int SpaceDim>
class ComputedFields
/******************************************************************************/
{
  private:
    std::vector<std::shared_ptr<ComputedField<SpaceDim>>> CFs;

  public :

  /****************************************************************************/
  /*!
    \brief Constructor that parses and creates a vector of ComputedField objects
    based on the ParameterList.
  */
  ComputedFields(const Omega_h::Mesh& aMesh, Teuchos::ParameterList &params) : CFs()
  /****************************************************************************/
  {
    for (Teuchos::ParameterList::ConstIterator i = params.begin(); i != params.end(); ++i)
    {
        const Teuchos::ParameterEntry &entry = params.entry(i);
        const std::string             &name  = params.name(i);
  
        TEUCHOS_TEST_FOR_EXCEPTION(!entry.isList(), std::logic_error, "Parameter in Computed Fields block not valid.  Expect lists only.");
  
        std::string function = params.sublist(name).get<std::string>("Function");
        auto newCF = std::make_shared<Plato::ComputedField<SpaceDim>>(aMesh, name, function);
        CFs.push_back(newCF);
    }
  }

  /****************************************************************************/
  /*!
    \brief Get the values for the specified field.
    @param aName Name of the requested Computed Field.
    @param aValues Computed Field values.
  */
  void get(const std::string& aName, Plato::ScalarVector& aValues)
  /****************************************************************************/
  {
      for (auto& cf : CFs) {
        if( cf->name() == aName ){
          cf->get(aValues);
          return;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Requested a Computed Field that doesn't exist.");
  }
  /****************************************************************************/
  /*!
    \brief Find a Computed Field with the given name.
    @param aName Name of the requested Computed Field.
    This is a canary function.  If it doesn't find the requested Computed field a
     signal is thrown.
  */
  void find(const std::string& aName)
  /****************************************************************************/
  {
      for (auto& cf : CFs) {
        if( cf->name() == aName ){
          return;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Requested a Computed Field that doesn't exist.");
  }
};
} // end Plato namespace
#endif
