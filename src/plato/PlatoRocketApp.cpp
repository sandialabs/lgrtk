/*
 * PlatoRocketApp.cpp
 *
 *  Created on: Nov 29, 2018
 */

#include <numeric>

#include "plato/PlatoRocketApp.hpp"
#include "plato/Plato_LevelSetCylinderInBox.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Default constructor
**********************************************************************************/
RocketApp::RocketApp() :
        mNumDesigVariables(2),
        mDefinedOperations(),
        mSharedDataMap(),
        mDefinedDataLayout()
{
}

/******************************************************************************//**
 * @brief Constructor
**********************************************************************************/
RocketApp::RocketApp(int aArgc, char **aArgv) :
        mNumDesigVariables(2),
        mDefinedOperations(),
        mSharedDataMap(),
        mDefinedDataLayout()
{
}

/******************************************************************************//**
 * @brief Destructor
**********************************************************************************/
RocketApp::~RocketApp()
{
}

/******************************************************************************//**
 * @brief Deallocate memory
**********************************************************************************/
void RocketApp::finalize()
{
    // MEMORY MANAGEMENT AUTOMATED, NO NEED TO EXPLICITLY DEALLOCATE MEMORY
    return;
}

/******************************************************************************//**
 * @brief Allocate memory
**********************************************************************************/
void RocketApp::initialize()
{
    this->defineOperations();
    this->defineSharedDataMaps();
    this->setRocketDriver();
}

/******************************************************************************//**
 * @brief Perform an operation, e.g. evaluate objective function
 * @param [in] aOperationName name of operation
**********************************************************************************/
void RocketApp::compute(const std::string & aOperationName)
{
    try
    {
        this->performOperation(aOperationName);
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::ostringstream tFullErrorMsg;
        tFullErrorMsg << "\n\n ********\n ERROR IN FILE: " << __FILE__ << "\n FUNCTION: " << __PRETTY_FUNCTION__
                << "\n LINE: " << __LINE__ << "\n ******** \n\n";
        tFullErrorMsg << tErrorMsg.what();
        std::cout << tFullErrorMsg.str().c_str();
        throw std::invalid_argument(tFullErrorMsg.str().c_str());
    }
}

/******************************************************************************//**
 * @brief Export data from user's application
 * @param [in] aArgumentName name of export data (e.g. objective gradient)
 * @param [out] aExportData container used to store output data
**********************************************************************************/
void RocketApp::exportData(const std::string & aArgumentName, Plato::SharedData & aExportData)
{
    try
    {
        this->outputData(aArgumentName, aExportData);
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::ostringstream tFullErrorMsg;
        tFullErrorMsg << "\n\n ********\n ERROR IN FILE: " << __FILE__ << "\n FUNCTION: " << __PRETTY_FUNCTION__
                << "\n LINE: " << __LINE__ << "\n ******** \n\n";
        tFullErrorMsg << tErrorMsg.what();
        std::cout << tFullErrorMsg.str().c_str();
        throw std::invalid_argument(tFullErrorMsg.str().c_str());
    }
}

/******************************************************************************//**
 * @brief Import data from Plato to user's application
 * @param [in] aArgumentName name of import data (e.g. design variables)
 * @param [in] aImportData container with import data
**********************************************************************************/
void RocketApp::importData(const std::string & aArgumentName, const Plato::SharedData & aImportData)
{
    try
    {
        this->inputData(aArgumentName, aImportData);
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::ostringstream tFullErrorMsg;
        tFullErrorMsg << "\n\n ********\n ERROR IN FILE: " << __FILE__ << "\n FUNCTION: " << __PRETTY_FUNCTION__
                << "\n LINE: " << __LINE__ << "\n ******** \n\n";
        tFullErrorMsg << tErrorMsg.what();
        std::cout << tFullErrorMsg.str().c_str();
        throw std::invalid_argument(tFullErrorMsg.str().c_str());
    }
}

/******************************************************************************//**
 * @brief Export distributed memory graph
 * @param [in] aDataLayout data layout (options: SCALAR, SCALAR_FIELD, VECTOR_FIELD,
 *                         TENSOR_FIELD, ELEMENT_FIELD, SCALAR_PARAMETER)
 * @param [out] aMyOwnedGlobalIDs my processor's global IDs
**********************************************************************************/
void RocketApp::exportDataMap(const Plato::data::layout_t & aDataLayout, std::vector<int> & aMyOwnedGlobalIDs)
{
    // THIS IS NOT A DISTRIBUTED MEMORY EXAMPLE; HENCE, THE DISTRIBUTED MEMORY GRAPH IS NOT NEEDEDS
    return;
}

/******************************************************************************//**
 * @brief Set output shared data container
 * @param [in] aArgumentName export data name (e.g. objective gradient)
 * @param [out] aExportData export shared data container
**********************************************************************************/
void RocketApp::outputData(const std::string & aArgumentName, Plato::SharedData & aExportData)
{
    try
    {
        auto tIterator = mSharedDataMap.find(aArgumentName);
        std::vector<Plato::Scalar> & tOutputData = tIterator->second;
        aExportData.setData(tOutputData);
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        throw tErrorMsg;
    }
}

/******************************************************************************//**
 * @brief Set input shared data container
 * @param [in] aArgumentName name of import data (e.g. design variables)
 * @param [in] aImportData import shared data container
**********************************************************************************/
void RocketApp::inputData(const std::string & aArgumentName, const Plato::SharedData & aImportData)
{
    try
    {
        auto tIterator = mSharedDataMap.find(aArgumentName);
        std::vector<Plato::Scalar> & tImportData = tIterator->second;
        aImportData.getData(tImportData);
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        throw tErrorMsg;
    }
}

/******************************************************************************//**
 * @brief Set rocket driver - runs rocket simulation given a define geometry.
**********************************************************************************/
void RocketApp::setRocketDriver()
{
    const Plato::Scalar tChamberRadius = 0.075; // m
    const Plato::Scalar tChamberLength = 0.65; // m
    std::shared_ptr<Plato::LevelSetCylinderInBox<Plato::Scalar>> tGeomModel =
            std::make_shared<Plato::LevelSetCylinderInBox<Plato::Scalar>>(tChamberRadius, tChamberLength);

    Plato::AlgebraicRocketInputs<Plato::Scalar> tDefaultInputs;
    mRocketDriver = std::make_shared<Plato::AlgebraicRocketModel<Plato::Scalar>>(tDefaultInputs, tGeomModel);
}

/******************************************************************************//**
 * @brief Define valid application-based operations
**********************************************************************************/
void RocketApp::defineOperations()
{
    mDefinedOperations.push_back("ObjectiveValue");
    mDefinedOperations.push_back("ObjectiveGradient");
}

/******************************************************************************//**
 * @brief Set default target thrust profile
**********************************************************************************/
void RocketApp::setDefaultTargetThrustProfile()
{
    std::vector<Plato::Scalar> tThrustProfile =
    {   0, 1656714.377766964, 1684717.520617273, 1713123.001583093, 1741935.586049868, 1771160.083875437,
        1800801.349693849, 1830864.28322051, 1861353.829558637, 1892274.979507048, 1923632.769869272,
        1955432.283763989, 1987678.650936801, 2020377.048073344, 2053532.699113719, 2087150.875568287,
        2121236.896834771, 2155796.130516737, 2190833.992743404, 2226355.948490792, 2262367.511904243,
        2298874.246622283, 2335881.766101836, 2373395.733944806, 2411421.864226017, 2449965.921822503,
        2489033.722744186, 2528631.134465915, 2568764.076260844, 2609438.519535244, 2650660.488164633,
        2692436.058831303, 2734771.361363255, 2777672.579074459, 2821145.949106557, 2865197.762771913,
        2909834.365898075, 2955062.159173611, 3000887.598495364, 3047317.195317072, 3094357.516999425,
        3142015.18716148, 3190296.886033527, 3239209.350811319, 3288759.376011737, 3338953.813829865,
        3389799.574497465, 3441303.626642879, 3493472.997652346, 3546314.774032734, 3599836.101775718,
        3654044.186723352, 3708946.294935087, 3764549.753056224, 3820861.948687783, 3877890.330757833,
        3935642.409894215, 3994125.758798767, 4053348.012622938, 4113316.869344868, 4174040.090147917,
        4235525.499800648, 4297780.987038235, 4360814.504945371, 4424634.071340578, 4489247.76916203,
        4554663.746854796, 4620890.218759571, 4687935.465502855, 4755807.834388626, 4824515.739791448,
        4894067.663551098, 4964472.155368621, 5035737.83320389, 5107873.383674653, 5180887.562457044,
        5254789.194687578, 5329587.175366664, 5405290.469763565, 5481908.11382287, 5559449.214572486,
        5637922.950533082, 5717338.572129052, 5797705.402100981, 5879032.835919643, 5961330.342201422,
        6044607.46312535, 6128873.814851565, 6214139.087941348, 6300413.047778608, 6387705.534992979,
        6476026.465884338, 6565385.832848894, 6655793.704806847, 6747260.227631442, 6839795.624579719,
        6933410.196724654, 7028114.32338894, 7123918.462580209, 7220833.151427887};
    std::string tArgumentName = "TargetThrustProfile";
    mSharedDataMap[tArgumentName] = tThrustProfile;
    mDefinedDataLayout[tArgumentName] = Plato::data::SCALAR;

    std::vector<Plato::Scalar> tNormThrustProfile =
            {std::inner_product(tThrustProfile.begin(), tThrustProfile.end(), tThrustProfile.begin(), 0.0)};
    tArgumentName = "NormThrustProfile";
    mSharedDataMap[tArgumentName] = tNormThrustProfile;
    mDefinedDataLayout[tArgumentName] = Plato::data::SCALAR;

}

/******************************************************************************//**
 * @brief Define valid application-based shared data containers
**********************************************************************************/
void RocketApp::defineSharedDataMaps()
{
    const int tLength = 1;
    std::string tName = "ObjFuncValue";
    mSharedDataMap[tName] = std::vector<Plato::Scalar>(tLength);
    mDefinedDataLayout[tName] = Plato::data::SCALAR;

    tName = "DesignVariables";
    mSharedDataMap[tName] = std::vector<Plato::Scalar>(mNumDesigVariables);
    mDefinedDataLayout[tName] = Plato::data::SCALAR;

    tName = "ThrustMisfitObjectiveGradient";
    mSharedDataMap[tName] = std::vector<Plato::Scalar>(mNumDesigVariables);
    mDefinedDataLayout[tName] = Plato::data::SCALAR;

    this->setNormalizationConstants();
    this->setDefaultTargetThrustProfile();
}

/******************************************************************************//**
 * @brief Set normalization constants for objective function
**********************************************************************************/
void RocketApp::setNormalizationConstants()
{
    std::vector<Plato::Scalar> tValues(mNumDesigVariables);
    tValues[0] = 0.08; tValues[1] = 0.006;
    std::string tName = "NormalizationConstants";
    mSharedDataMap[tName] = tValues;
    mDefinedDataLayout[tName] = Plato::data::SCALAR;
}

/******************************************************************************//**
 * @brief Perform valid application-based operation.
 * @param [in] aOperationName name of operation
**********************************************************************************/
void RocketApp::performOperation(const std::string & aOperationName)
{
    if(aOperationName.compare("ObjectiveValue") == static_cast<int>(0))
    {
        this->evaluateObjFunc();
    }
    else if (aOperationName.compare("ObjectiveGradient") == static_cast<int>(0))
    {
        this->evaluateObjFuncGrad();
    }
}

/******************************************************************************//**
 * @brief Evaluate objective function
**********************************************************************************/
void RocketApp::evaluateObjFunc()
{
    std::string tArgumentName = "DesignVariables";
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tControl = tIterator->second;

    tArgumentName = "TargetThrustProfile";
    tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tTargetThrustProfile = tIterator->second;

    tArgumentName = "ObjFuncValue";
    tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    const Plato::OrdinalType tINDEX = 0;
    tIterator->second.operator[](tINDEX) = this->computeObjFuncValue(tControl, tTargetThrustProfile);
}

/******************************************************************************//**
 * @brief Compute objective gradient
**********************************************************************************/
void RocketApp::evaluateObjFuncGrad()
{
    std::string tArgumentName("DesignVariables");
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tControl = tIterator->second;

    tArgumentName = "TargetThrustProfile";
    tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tTargetThrustProfile = tIterator->second;

    tArgumentName = "ThrustMisfitObjectiveGradient";
    tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tGradient = tIterator->second;
    assert(tGradient.size() == mNumDesigVariables);

    this->computeObjFuncGrad(tControl, tTargetThrustProfile, tGradient);
}

/******************************************************************************//**
 * @brief Evaluate thrust profile misfit given target thrust profile
**********************************************************************************/
void RocketApp::computeObjFuncGrad(const std::vector<Plato::Scalar> & aControl,
                                   const std::vector<Plato::Scalar> & aTargetProfile,
                                   std::vector<Plato::Scalar> & aOutput)
{
    Plato::Scalar tEpsilon = 1e-4;
    const Plato::OrdinalType tNumControls = aControl.size();
    std::vector<Plato::Scalar> tControlCopy(tNumControls);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumControls; tIndex++)
    {
        // modify base value - forward
        tControlCopy[tIndex] = aControl[tIndex] + (aControl[tIndex] * tEpsilon);
        // evaluate criterion with modified value
        Plato::Scalar tForwardCriterionValue = this->computeObjFuncValue(tControlCopy, aTargetProfile);

        // modify base value - backward
        tControlCopy[tIndex] = aControl[tIndex] - (aControl[tIndex] * tEpsilon);
        // evaluate criterion with modified value
        Plato::Scalar tBackwardCriterionValue = this->computeObjFuncValue(tControlCopy, aTargetProfile);

        // reset base value
        tControlCopy[tIndex] = aControl[tIndex];
        // central difference gradient approximation
        aOutput[tIndex] = (tForwardCriterionValue - tBackwardCriterionValue) / (static_cast<Plato::Scalar>(2) * tEpsilon);
    }
}

/******************************************************************************//**
 * @brief Evaluate thrust profile misfit given target thrust profile
**********************************************************************************/
Plato::Scalar RocketApp::computeObjFuncValue(const std::vector<Plato::Scalar> & aControl,
                                             const std::vector<Plato::Scalar> & aTargetProfile)
{
    this->updateModel(aControl);

    mRocketDriver->solve();
    auto tTrialProfile = mRocketDriver->getThrustProfile();
    assert(tTrialProfile.size() == aTargetProfile.size());

    Plato::Scalar tObjFuncValue = 0;
    for(Plato::OrdinalType tIndex = 0; tIndex < aTargetProfile.size(); tIndex++)
    {
        Plato::Scalar tDeltaThrust = tTrialProfile[tIndex] - aTargetProfile[tIndex];
        tObjFuncValue += tDeltaThrust * tDeltaThrust;
    }

    std::string tArgumentName = "NormThrustProfile";
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tNormThrustProfile = tIterator->second;
    const Plato::OrdinalType tINDEX = 0;
    const Plato::Scalar tVecLength = aTargetProfile.size();
    const Plato::Scalar tDenominator = static_cast<Plato::Scalar>(2.0) * tVecLength * tNormThrustProfile[tINDEX];
    tObjFuncValue = (static_cast<Plato::Scalar>(1.0) / tDenominator) * tObjFuncValue;

    return (tObjFuncValue);
}

/******************************************************************************//**
* @brief update parameters (e.g. design variables) for simulation.
**********************************************************************************/
void RocketApp::updateModel(const std::vector<Plato::Scalar> & aControls)
{
    std::string tArgumentName = "NormalizationConstants";
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tNormalizationConstants = tIterator->second;

    Plato::OrdinalType tINDEX = 0;
    const Plato::Scalar tRadius = aControls[tINDEX] * tNormalizationConstants[0];
    std::map<std::string, Plato::Scalar> tGeomParams;
    tGeomParams.insert(std::pair<std::string, Plato::Scalar>("Radius", tRadius));
    tGeomParams.insert(std::pair<std::string, Plato::Scalar>("Configuration", Plato::Configuration::INITIAL));
    mRocketDriver->updateInitialChamberGeometry(tGeomParams);

    tINDEX = 1;
    const Plato::Scalar tRefBurnRate = aControls[tINDEX] * tNormalizationConstants[1];
    std::map<std::string, Plato::Scalar> tSimParams;
    tSimParams.insert(std::pair<std::string, Plato::Scalar>("RefBurnRate", tRefBurnRate));
    mRocketDriver->updateSimulation(tSimParams);
}

} // namespace Plato
