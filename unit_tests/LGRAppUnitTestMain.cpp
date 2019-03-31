#include "Teuchos_UnitTestRepository.hpp"
#include <mpi.h>
#include <Kokkos_Core.hpp>

int main( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  auto result = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

  Kokkos::finalize();
  MPI_Finalize();

  return result;
}
