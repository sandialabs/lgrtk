#ifndef LGR_TEST_HELPERS
#define LGR_TEST_HELPERS

#include <Omega_h_assoc.hpp>
#include <Omega_h_expr.hpp>
#include <Omega_h_library.hpp>

#include "Teuchos_UnitTestHarness.hpp"
#include "Kokkos_Core.hpp"

#include "CrsMatrix.hpp"
#include <ParallelComm.hpp>

namespace lgr
{
  void initializeCommMachine(int *argc , char ***argv);
  void initializeOmegaH(int *argc , char ***argv);
  void finalizeCommMachine();
  void finalizeOmegaH();
  
  comm::Machine getCommMachine();
  Teuchos::RCP<Omega_h::Library> getLibraryOmegaH();
  
  template<class KokkosViewType>
  void evaluateNodalExpression(std::string &expressionInXYZ, int spaceDim, Omega_h::Reals nodalCoords, KokkosViewType values)
  {
    int nodeCount = nodalCoords.size() / spaceDim;
    Omega_h::Write<double> x_coords(nodeCount);
    Omega_h::Write<double> y_coords(nodeCount);
    Omega_h::Write<double> z_coords(nodeCount);
    
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,nodeCount), LAMBDA_EXPRESSION(int nodeOrdinal)
    {
      if (spaceDim > 0)
        x_coords[nodeOrdinal] = nodalCoords[nodeOrdinal * spaceDim + 0];
      if (spaceDim > 1)
        y_coords[nodeOrdinal] = nodalCoords[nodeOrdinal * spaceDim + 1];
      if (spaceDim > 2)
        z_coords[nodeOrdinal] = nodalCoords[nodeOrdinal * spaceDim + 2];
    }, "fill coords");
    
    Omega_h::ExprReader reader(nodeCount, spaceDim);
    if (spaceDim > 0) reader.register_variable("x", Teuchos::any(Omega_h::Reals(x_coords)));
    if (spaceDim > 1) reader.register_variable("y", Teuchos::any(Omega_h::Reals(y_coords)));
    if (spaceDim > 2) reader.register_variable("z", Teuchos::any(Omega_h::Reals(z_coords)));

    Teuchos::any result;
    reader.read_string(result, expressionInXYZ, "expression in x/y/z");
    reader.repeat(result);
    auto fxnValues = Teuchos::any_cast<Omega_h::Reals>(result);
    
    // assuming that values is a flat Kokkos view with entries of the same type as fxnValues, an Omega_h array,
    // copy to the values view
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,nodeCount), LAMBDA_EXPRESSION(int nodeOrdinal)
    {
      values(nodeOrdinal) = fxnValues[nodeOrdinal];
    }, "copy to Kokkos View");
  }

  bool fileExists(const std::string &filePath);
  
  template<class Scalar, class ViewType>
  void testFloatingEquality(ViewType first, ViewType second, Scalar relTol, Scalar floor, Teuchos::FancyOStream &out, bool &success)
  {
    typename ViewType::HostMirror firstHost  = Kokkos::create_mirror_view( first );
    typename ViewType::HostMirror secondHost = Kokkos::create_mirror_view( second );

    // copy to host
    Kokkos::deep_copy( firstHost, first );
    Kokkos::deep_copy( secondHost, second );

    Scalar* firstValuePtr = firstHost.data();
    Scalar* secondValuePtr = secondHost.data();

    unsigned size = first.size();
    TEST_EQUALITY( size, second.size() );

    if (size == second.size())
    {
      for (unsigned valueOrdinal=0; valueOrdinal<size; valueOrdinal++)
      {
        if ((std::abs(*firstValuePtr) > floor) || (std::abs(*secondValuePtr) > floor))
        {
          TEST_FLOATING_EQUALITY( *firstValuePtr, *secondValuePtr, relTol );
        }
        firstValuePtr++;
        secondValuePtr++;
      }
    }
  }
  
  template<class Scalar, class ViewType>
  void testFloatingEquality(ViewType first, ViewType second, Scalar tol, Teuchos::FancyOStream &out, bool &success)
  {
    // if only one tolerance is specified, use for both floor and relative tolerance
    Scalar relTol = tol;
    Scalar floor  = tol;
    testFloatingEquality(first, second, relTol, floor, out, success);
  }
  
  template<class Ordinal, class ViewType>
  void testEquality(ViewType first, ViewType second, Teuchos::FancyOStream &out, bool &success)
  {
    typename ViewType::HostMirror firstHost  = Kokkos::create_mirror_view( first );
    typename ViewType::HostMirror secondHost = Kokkos::create_mirror_view( second );
    
    // copy to host
    Kokkos::deep_copy( firstHost,  first  );
    Kokkos::deep_copy( secondHost, second );
    
    Ordinal* firstValuePtr = firstHost.data();
    Ordinal* secondValuePtr = secondHost.data();
    
    unsigned size = first.size();
    TEST_EQUALITY( size, second.size() );
    
    if (size == second.size())
    {
      for (unsigned valueOrdinal=0; valueOrdinal<size; valueOrdinal++)
      {
        TEST_EQUALITY( *firstValuePtr, *secondValuePtr );
        firstValuePtr++;
        secondValuePtr++;
      }
    }
  }
  
  template<class Ordinal, class SizeType>
  void testFloatingEquality(CrsMatrix<Ordinal, SizeType> &first,
                            CrsMatrix<Ordinal, SizeType> &second,
                            double floatTol,
                            Teuchos::FancyOStream &out, bool &success)
  {
    typedef Kokkos::View<Ordinal*, MemSpace> OrdinalVector;
    typedef Kokkos::View<Scalar* , MemSpace> ScalarVector;
    typedef Kokkos::View<SizeType*, MemSpace> SizeTypeVector;
    
    testEquality<SizeType, SizeTypeVector>( first.rowMap(),        second.rowMap(),           out, success );
    testEquality<Ordinal, OrdinalVector>(  first.columnIndices(), second.columnIndices(),     out, success );
    testFloatingEquality<Scalar, ScalarVector>(  first.entries(), second.entries(), floatTol, out, success );
  }
  
  
}

#endif
