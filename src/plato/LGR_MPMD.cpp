#include "LGR_App.hpp"

#ifndef NDEBUG
#include <fenv.h>
#endif

/******************************************************************************/
int main(int aArgc, char **aArgv)
/******************************************************************************/
{
#ifndef NDEBUG
//    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

    MPI_Init(&aArgc, &aArgv);
    Kokkos::initialize(aArgc, aArgv);

    Plato::Interface* tPlatoInterface = nullptr;
    try
    {
      tPlatoInterface = new Plato::Interface();
    }
    catch(...)
    {
      Kokkos::finalize_all();
      MPI_Finalize();
    }

    MPI_Comm tLocalComm;
    tPlatoInterface->getLocalComm(tLocalComm);

    MPMD_App* tMyApp = nullptr;
    try
    {
      tMyApp = new MPMD_App(aArgc, aArgv, tLocalComm);
    }
    catch(...)
    {
      Kokkos::finalize_all();
      MPI_Finalize();
    }

    try
    {
      tPlatoInterface->registerPerformer(tMyApp);
    }
    catch(...)
    {
      Kokkos::finalize_all();
      MPI_Finalize();
    }

    tPlatoInterface->perform();

    delete tMyApp;

    Kokkos::finalize_all();
    MPI_Finalize();
}
