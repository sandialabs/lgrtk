/*
 * PlatoRocketAppMPMD.cpp
 *
 *  Created on: Dec 3, 2018
 */

#include <mpi.h>

#include <Plato_Interface.hpp>

#include "plato/PlatoRocketApp.hpp"

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

    Plato::RocketApp* tMyApp = nullptr;
    try
    {
      tMyApp = new Plato::RocketApp(aArgc, aArgv, tLocalComm);
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
    tMyApp->printSolution();

    delete tMyApp;

    Kokkos::finalize_all();
    MPI_Finalize();
}
