/*
 * AlgebraicRocketMain.cpp
 *
 *  Created on: Mar 27, 2019
 */

#include <mpi.h>

#include <string>
#include <memory>

#include "plato/Plato_AlgebraicRocketModel.hpp"
#include "plato/Plato_LevelSetOnExternalMesh.hpp"

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

    try
    {
        std::string tConnInputFile("./ParcMesh/tetTT.txt");
        std::string tCoordsInputFile("./ParcMesh/tetV.txt");
        auto tGeometry = std::make_shared < Plato::LevelSetOnExternalMesh > (tCoordsInputFile, tConnInputFile);
        Plato::ProblemParams tParams;
        tGeometry->initialize(tParams);

        std::string tLevelSetFile("./ParcMesh/nodalInnerDistanceField.txt");
        tGeometry->readNodalLevelSet(tLevelSetFile);
        std::string tBurnRateFile("./ParcMesh/elementalMaterialField.txt");
        tGeometry->readElementBurnRate(tBurnRateFile);

        const Plato::AlgebraicRocketInputs tRocketInputs;
        Plato::AlgebraicRocketModel tDriver(tRocketInputs, tGeometry);
        tDriver.solve();

        auto tThrustProfile = tDriver.getThrustProfile();
        std::ofstream tOutput;
        tOutput.open("thrust_profile.txt");
        for(size_t tIndex = 0; tIndex < tThrustProfile.size(); tIndex++)
        {
            tOutput << tThrustProfile[tIndex] << "\n";
        }
        tOutput.close();
    }
    catch(...)
    {
        Kokkos::finalize_all();
        MPI_Finalize();
    }

    Kokkos::finalize_all();
    MPI_Finalize();
}



