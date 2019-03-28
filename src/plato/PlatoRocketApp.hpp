/*
 * PlatoRocketApp.hpp
 *
 *  Created on: Nov 29, 2018
 */

#pragma once

#include <memory>

#include <mpi.h>

#include <Plato_Application.hpp>

#include "plato/PlatoTypes.hpp"
#include "plato/Plato_AlgebraicRocketModel.hpp"

namespace Plato
{

class RocketApp : public Plato::Application
{
public:
    /******************************************************************************//**
     * @brief Default constructor
    **********************************************************************************/
    RocketApp();

    /******************************************************************************//**
     * @brief Constructor
    **********************************************************************************/
    RocketApp(int aArgc, char **aArgv, MPI_Comm & aComm);

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    virtual ~RocketApp();

    /******************************************************************************//**
     * @brief Deallocate memory
    **********************************************************************************/
    void finalize();

    /******************************************************************************//**
     * @brief Allocate memory
    **********************************************************************************/
    void initialize();

    /******************************************************************************//**
     * @brief Perform an operation, e.g. evaluate objective function
     * @param [in] aOperationName name of operation
    **********************************************************************************/
    void compute(const std::string & aOperationName);

    /******************************************************************************//**
     * @brief Export data from user's application
     * @param [in] aArgumentName name of export data (e.g. objective gradient)
     * @param [out] aExportData container used to store output data
    **********************************************************************************/
    void exportData(const std::string & aArgumentName, Plato::SharedData & aExportData);

    /******************************************************************************//**
     * @brief Import data from Plato to user's application
     * @param [in] aArgumentName name of import data (e.g. design variables)
     * @param [in] aImportData container with import data
    **********************************************************************************/
    void importData(const std::string & aArgumentName, const Plato::SharedData & aImportData);

    /******************************************************************************//**
     * @brief Export distributed memory graph
     * @param [in] aDataLayout data layout (options: SCALAR, SCALAR_FIELD, VECTOR_FIELD,
     *                         TENSOR_FIELD, ELEMENT_FIELD, SCALAR_PARAMETER)
     * @param [out] aMyOwnedGlobalIDs my processor's global IDs
    **********************************************************************************/
    void exportDataMap(const Plato::data::layout_t & aDataLayout, std::vector<int> & aMyOwnedGlobalIDs);

    /******************************************************************************//**
     * @brief Print solution to file
    **********************************************************************************/
    void printSolution();

private:
    /******************************************************************************//**
     * @brief Set output shared data container
     * @param [in] aArgumentName export data name (e.g. objective gradient)
     * @param [out] aExportData export shared data container
    **********************************************************************************/
    void outputData(const std::string & aArgumentName, Plato::SharedData & aExportData);

    /******************************************************************************//**
     * @brief Set input shared data container
     * @param [in] aArgumentName name of import data (e.g. design variables)
     * @param [in] aImportData import shared data container
    **********************************************************************************/
    void inputData(const std::string & aArgumentName, const Plato::SharedData & aImportData);

    /******************************************************************************//**
     * @brief Perform valid application-based operation.
     * @param [in] aOperationName name of operation
    **********************************************************************************/
    void performOperation(const std::string & aOperationName);

    /******************************************************************************//**
    * @brief Update initial configuration/geometry and burn rate field.
    * @param [in] aControls optimization variables
    **********************************************************************************/
    void updateProblem(const std::vector<Plato::Scalar> & aControls);

    /******************************************************************************//**
    * @brief Compute maximum thrust from target thrust profile and set value on shared data map
    * @param [in] aThrustProfile target thrust profile
    **********************************************************************************/
    void setMaxTargetThrust(const std::vector<Plato::Scalar> & aThrustProfile);

    /******************************************************************************//**
    * @brief Compute the norm of the target thrust profile and set value on shared data map
    * @param [in] aThrustProfile target thrust profile
    **********************************************************************************/
    void setNormTargetThrustProfile(const std::vector<Plato::Scalar> & aThrustProfile);

    /******************************************************************************//**
    * @brief Set target thrust profile values on shared data map
    * @param [in] aThrustProfile target thrust profile
    **********************************************************************************/
    void setTargetThrustProfile(const std::vector<Plato::Scalar> & aThrustProfile);

    /******************************************************************************//**
     * @brief Set rocket driver - runs rocket simulation given a define geometry.
    **********************************************************************************/
    void setRocketDriver();

    /******************************************************************************//**
     * @brief Define valid application-based operations
    **********************************************************************************/
    void defineOperations();

    /******************************************************************************//**
     * @brief Define valid application-based shared data containers and layouts
    **********************************************************************************/
    void defineSharedDataMaps();

    /******************************************************************************//**
     * @brief Set default target thrust profile
    **********************************************************************************/
    void setDefaultTargetThrustProfile();

    /******************************************************************************//**
     * @brief Evaluate objective function
    **********************************************************************************/
    void evaluateObjFunc();

    /******************************************************************************//**
     * @brief evaluate objective function gradient
    **********************************************************************************/
    void evaluateObjFuncGrad();

    /******************************************************************************//**
     * @brief Set normalization constants for objective function
    **********************************************************************************/
    void setNormalizationConstants();

    /******************************************************************************//**
     * @brief Return maximum target thrust
     * @return maximum target thrust
    **********************************************************************************/
    Plato::Scalar getMaxTargetThrust() const;

    /******************************************************************************//**
     * @brief Return the norm of the target thrust profile
     * @return norm of the target thrust profile
    **********************************************************************************/
    Plato::Scalar getNormTargetThrustProfile() const;

    /******************************************************************************//**
     * @brief Evaluate thrust profile misfit given target thrust profile
     * @param [in] aControl design/optimization variables
     * @param [in] aTargetThrustProfile target thrust profile
    **********************************************************************************/
    Plato::Scalar computeObjFuncValue(const std::vector<Plato::Scalar> & aControl,
                                      const std::vector<Plato::Scalar> & aTargetProfile);

    /******************************************************************************//**
     * @brief Evaluate thrust profile misfit given target thrust profile
     * @param [in] aControls design/optimization variables
     * @param [in] aTargetThrustProfile target thrust profile
     * @param [out] aOutput objective function gradient
    **********************************************************************************/
    void computeObjFuncGrad(const std::vector<Plato::Scalar> & aControls,
                            const std::vector<Plato::Scalar> & aTargetProfile,
                            std::vector<Plato::Scalar> & aOutput);

private:
    MPI_Comm mComm; /*!< local mpi communicator */
    Plato::Scalar mLength; /*!< cylinder's length */
    Plato::Scalar mMaxRadius; /*!< cylinder's max radius */
    size_t mNumDesigVariables; /*!< import/export parameter map */

    std::shared_ptr<Plato::AlgebraicRocketModel> mRocketDriver; /*!< rocket driver */
    std::vector<std::string> mDefinedOperations; /*!< valid operations recognized by app */
    std::map<std::string, Plato::data::layout_t> mDefinedDataLayout; /*!< valid data layouts */
    std::map<std::string, std::vector<Plato::Scalar>> mSharedDataMap; /*!< import/export shared data map */

private:
    RocketApp(const Plato::RocketApp & aRhs);
    Plato::RocketApp & operator=(const Plato::RocketApp & aRhs);
};
// class RocketDesignApp

} // namespace Plato
