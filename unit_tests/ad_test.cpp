#include "Sacado.hpp"

#include <Kokkos_Macros.hpp>
#include <iostream>
#include <stdlib.h>

#include <P32e2.h>

#ifdef KOKKOS_HAVE_CUDA
#define LAMBDA_EXPRESSION [=]__device__
#else
#define LAMBDA_EXPRESSION [=]
#endif

int main(int argc, char* argv[]) {
  int ret = 0;

//  typedef double DataType;
//  typedef float DataType;
  typedef P32e2 DataType;

  Kokkos::initialize(argc, argv);

  auto nterms = getenv("NUM_TERMS");
  if( nterms == NULL ){
    std::cout << "'NUM_TERMS' environment variable not set." << std::endl;
  } else 
  {
 
    size_t n = atoi(nterms);

    const size_t p = 30;  // Derivative dimension

    std::cout << "Vector length: " << n << ", N derivatives: " << p << std::endl;

    typedef Sacado::Fad::SFad<DataType,p> FadType;

    using DefaultSpace = Kokkos::DefaultExecutionSpace;
    using DefaultLayout = Kokkos::DefaultExecutionSpace::array_layout;

    Kokkos::View<FadType*, DefaultLayout, DefaultSpace>  a("a",n);
    Kokkos::View<FadType*, DefaultLayout, DefaultSpace>  b("b",p);

    Kokkos::View<DataType*, DefaultLayout, DefaultSpace>  c("c",n);
    Kokkos::View<DataType*, DefaultLayout, DefaultSpace>  dcdp("c",n,p);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,p), LAMBDA_EXPRESSION(int derivOrdinal)
    {
      b(derivOrdinal) = FadType(p, derivOrdinal, 1.0);
    }, "initialize");

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,n), LAMBDA_EXPRESSION(int listOrdinal)
    {
      a(listOrdinal) = 0.0;
      for(int ip=0; ip<p; ip++)
        a(listOrdinal) += b(ip)*b(ip);

      c(listOrdinal) = a(listOrdinal).val();
      for(int ip=0; ip<p; ip++)
        dcdp(listOrdinal,ip) = a(listOrdinal).dx(ip);
    }, "compute");

    //#define AD_TEST_WRITE
    #ifdef AD_TEST_WRITE
    auto cHost = Kokkos::create_mirror_view( c );
    Kokkos::deep_copy( cHost, c );

    for(int i=0; i<n; i++)
      std::cout << "c(" << i << "): " << cHost(i) << std::endl;

    auto dcdpHost = Kokkos::create_mirror_view( dcdp );
    Kokkos::deep_copy( dcdpHost, dcdp );

    for(int i=0; i<n; i++)
      std::cout << "dcdp(" << i << "): " << dcdpHost(i,0) << ", " << dcdpHost(i,1) << ", " << dcdpHost(i,2) << std::endl;
    #endif

  }
  Kokkos::finalize();


  return ret;
}
