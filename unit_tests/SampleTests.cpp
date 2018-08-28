#include "Teuchos_UnitTestHarness.hpp"

namespace {
  // here's a very basic unit test
  TEUCHOS_UNIT_TEST( Int, Assignment )
  {
    int i1 = 4;
    int i2 = i1;
    TEST_EQUALITY( i2, i1 );
  }
  // when writing tests that involve floating point calculations, use TEST_FLOATING_EQUALITY
  TEUCHOS_UNIT_TEST( Double, Addition )
  {
    const double tol = 1e-15;
    double val1 = 1.0;
    double summand = 1.0 + 1e-16;
    double sum = val1 + summand;
    TEST_FLOATING_EQUALITY(sum - val1, summand, tol);
  }
  // Sometimes, you'll want to share testing code across several tests.
  // For this, it's good to be aware that the TEUCHOS_UNIT_TEST() macro expands to a function
  // declaration that involves arguments 
  //   Teuchos::FancyOStream &out
  // and
  //   bool &success
  // These are used by the various TEUCHOS macros to communicate success alongside output to the
  // unit test harness.
  //
  // This means that you can write shared test code like this:
  void testSum(double val1, double val2, double tol, Teuchos::FancyOStream &out, bool &success)
  {
    double sum = val1 + val2;
    TEST_FLOATING_EQUALITY(sum - val2, val1, tol); // this will use "out" and "success"
    TEST_FLOATING_EQUALITY(sum - val1, val2, tol); 
  }

  // to invoke testSum from inside a unit test, just pass the out and success variables:
  TEUCHOS_UNIT_TEST( Double, Addition_BigNumbers)
  {
    testSum(1e6, 1e7, 1e-15, out, success);
  }

  TEUCHOS_UNIT_TEST( Double, Addition_SmallNumbers)
  {
    testSum(1e-6, 1e-7, 1e-13, out, success);
  }
} // namespace
