#include "Teuchos_UnitTestRepository.hpp"
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "PlatoTestHelpers.hpp"

int main( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  PlatoUtestHelpers::initializeOmegaH(&argc, &argv);

  auto result = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

  PlatoUtestHelpers::finalizeOmegaH();
  Kokkos::finalize();
  MPI_Finalize();

  return result;
}
