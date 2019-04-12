#include "LGR_App.hpp"

#ifndef NDEBUG
#include <fenv.h>
#endif

void safeExit(){
  Kokkos::finalize_all();
  MPI_Finalize();
  exit(0);
}

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
      safeExit();
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
      tMyApp = nullptr;
      tPlatoInterface->Catch();
    }

    try
    {
      tPlatoInterface->registerPerformer(tMyApp);
    }
    catch(...)
    {
      safeExit();
    }

    tPlatoInterface->perform();

    if(tMyApp)
    {
      delete tMyApp;
    }
    
    if(tPlatoInterface)
    {
      delete tPlatoInterface;
    }

    safeExit();
}
