/*
 * PlatoRocketApp.cpp
 *
 *  Created on: Nov 29, 2018
 */

#include <numeric>
#include <fstream>

#include "plato/PlatoRocketApp.hpp"
#include "plato/Plato_RocketMocks.hpp"
#include "plato/Plato_LevelSetCylinderInBox.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Default constructor
**********************************************************************************/
RocketApp::RocketApp() :
        mComm(MPI_COMM_NULL),
        mLength(0.65),
        mMaxRadius(0.1524),
        mNumDesigVariables(2),
        mDefinedOperations(),
        mSharedDataMap(),
        mDefinedDataLayout()
{
}

/******************************************************************************//**
 * @brief Constructor
**********************************************************************************/
RocketApp::RocketApp(int aArgc, char **aArgv, MPI_Comm & aComm) :
        mComm(aComm),
        mLength(0.65),
        mMaxRadius(0.15),
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
 * @brief print solution to file
**********************************************************************************/
void RocketApp::printSolution()
{
    Plato::OrdinalType tMyProcID = 0;
    MPI_Comm_rank(mComm, &tMyProcID);
    if(tMyProcID == static_cast<Plato::OrdinalType>(0))
    {
        std::string tArgumentName = "DesignVars";
        auto tIterator = mSharedDataMap.find(tArgumentName);
        assert(tIterator != mSharedDataMap.end());
        std::vector<Plato::Scalar> & tControl = tIterator->second;

        tArgumentName = "Normalization";
        tIterator = mSharedDataMap.find(tArgumentName);
        assert(tIterator != mSharedDataMap.end());
        std::vector<Plato::Scalar> & tNormalizationConstants = tIterator->second;

        const Plato::Scalar tRadius = tControl[0] * tNormalizationConstants[0];
        const Plato::Scalar tRefBurnRate = tControl[1] * tNormalizationConstants[1];

        std::ofstream tMyFile;
        tMyFile.open ("solution.txt");
        tMyFile << "Radius = " << tRadius << "\n";
        tMyFile << "RefBurnRate = " << tRefBurnRate << "\n";
        tMyFile.close();
    }
}

/******************************************************************************//**
 * @brief Deallocate memory
**********************************************************************************/
void RocketApp::finalize()
{
    // MEMORY MANAGEMENT AUTOMATED, NO NEED TO DEALLOCATE MEMORY
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
    const Plato::Scalar tRefBurnRate = 0.005;  // meters/seconds
    const Plato::Scalar tInitialRadius = 0.075; // meters
    Plato::ProblemParams tParams =
            Plato::RocketMocks::setupConstantBurnRateCylinder(mMaxRadius /* meters */, mLength /* meters */, tInitialRadius, tRefBurnRate);

    std::shared_ptr<Plato::LevelSetCylinderInBox> tGeometry =
            std::make_shared<Plato::LevelSetCylinderInBox>(mComm);
    tGeometry->initialize(tParams);

    Plato::AlgebraicRocketInputs tDefaulRocketParams;
    mRocketDriver = std::make_shared<Plato::AlgebraicRocketModel>(tDefaulRocketParams, tGeometry);
    mRocketDriver->disableOutput();
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
    std::vector<Plato::Scalar> tThrustProfile = {0, 6811.91690423622822, 6885.76425285549794, 6952.84181118835659,
                                                 7017.10627931516956, 7083.92851111752952, 7139.96284599571663,
                                                 7207.85959512876707, 7286.36993377595445, 7369.57061836979938,
                                                 7459.08942564254812, 7543.00692218823315, 7625.02657788328361,
                                                 7696.49968238560723, 7759.77134307169581, 7823.2527826904734,
                                                 7895.65890310796476, 7979.01759389434119, 8062.93969449689394,
                                                 8151.81213485658918, 8253.89524206284477, 8343.71290927815789,
                                                 8416.29681928578793, 8481.33569105677816, 8554.10205550906358,
                                                 8634.95970754549853, 8718.61205745934421, 8801.16090885243102,
                                                 8910.19234538166893, 9016.67874341392235, 9108.66236698916327,
                                                 9175.71193584420325, 9256.98852311476548, 9336.96387801038145,
                                                 9416.95933388763297, 9502.88436091731455, 9608.42273427288819,
                                                 9727.38807251860089, 9829.52428091608272, 9913.82921799349788,
                                                 9998.3538694999479, 10082.6416999770172, 10157.9353736813082,
                                                 10251.568806531639, 10358.8430391618967, 10478.0535188534868,
                                                 10588.1879134104129, 10694.2083080393004, 10780.2779360630939,
                                                 10867.6828406155128, 10941.0225989588926, 11049.544195790344,
                                                 11155.1598565935419, 11274.1558664821805, 11398.9593453303951,
                                                 11515.8876076763408, 11604.3112275735602, 11686.0344192982993,
                                                 11780.9381728696535, 11889.2300735748886, 11995.4131687416593,
                                                 12123.6659031514719, 12267.2455406308327, 12375.9814215994884,
                                                 12461.181478405093, 12562.1121153937565, 12662.916522210362,
                                                 12767.1714721757107, 12890.1679558447431, 13040.326081441146,
                                                 13173.537266626774, 13274.989276455839, 13378.6622523041551,
                                                 13481.9180166030183, 13579.0289182385604, 13705.4482058816211,
                                                 13852.2109797835383, 13998.6223527384682, 14120.7442125255948,
                                                 14235.6426357559048, 14336.233860717286, 14434.7780273650678,
                                                 14565.7436677433197, 14709.5170657710532, 14856.3581810925607,
                                                 15008.2555186239424, 15125.1210307026959, 15226.5202296915504,
                                                 15332.4230768687303, 15471.6863473141984, 15601.3576989909761,
                                                 15771.4477240094257, 15931.7637255970021, 16050.1101398000592,
                                                 16154.536652812576, 16277.5736765839247, 16407.8902991781033,
                                                 16546.4323361623537, 16731.4999881392032, 16895.824223374424};

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
    std::string tName = "ObjFuncVal";
    mSharedDataMap[tName] = std::vector<Plato::Scalar>(tLength);
    mDefinedDataLayout[tName] = Plato::data::SCALAR;

    tName = "DesignVars";
    mSharedDataMap[tName] = std::vector<Plato::Scalar>(mNumDesigVariables);
    mDefinedDataLayout[tName] = Plato::data::SCALAR;

    tName = "ObjFuncGrad";
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
    tValues[0] = 0.1; tValues[1] = 0.01;
    std::string tName = "Normalization";
    mSharedDataMap[tName] = tValues;
    mDefinedDataLayout[tName] = Plato::data::SCALAR;
}

/******************************************************************************//**
 * @brief Perform valid application-based operation.
 * @param [in] aOperationName name of operation
**********************************************************************************/
void RocketApp::performOperation(const std::string & aOperationName)
{
    if(aOperationName.compare("ObjFuncEval") == static_cast<int>(0))
    {
        this->evaluateObjFunc();
    }
    else if (aOperationName.compare("ObjFuncGrad") == static_cast<int>(0))
    {
        this->evaluateObjFuncGrad();
    }
}

/******************************************************************************//**
 * @brief Evaluate objective function
**********************************************************************************/
void RocketApp::evaluateObjFunc()
{
    std::string tArgumentName = "DesignVars";
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tControl = tIterator->second;

    tArgumentName = "TargetThrustProfile";
    tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tTargetThrustProfile = tIterator->second;

    tArgumentName = "ObjFuncVal";
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
    std::string tArgumentName("DesignVars");
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tControl = tIterator->second;

    tArgumentName = "TargetThrustProfile";
    tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tTargetThrustProfile = tIterator->second;

    tArgumentName = "ObjFuncGrad";
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
    std::vector<Plato::Scalar> tControlCopy(aControl);
    auto tNumControls = aControl.size();
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
    this->updateProblem(aControl);
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
    //const Plato::Scalar tDenominator = tNormThrustProfile[tINDEX];
    tObjFuncValue = (static_cast<Plato::Scalar>(1.0) / tDenominator) * tObjFuncValue;

    return (tObjFuncValue);
}

/******************************************************************************//**
* @brief Update parameters (e.g. design variables) for simulation.
**********************************************************************************/
void RocketApp::updateProblem(const std::vector<Plato::Scalar> & aControls)
{
    std::string tArgumentName = "Normalization";
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tNormalizationConstants = tIterator->second;

    auto tRadius = aControls[0] * tNormalizationConstants[0]; // meters
    auto tRefBurnRate = aControls[1] * tNormalizationConstants[1]; // meters/seconds
    auto tParams = Plato::RocketMocks::setupConstantBurnRateCylinder(mMaxRadius /* meters */, mLength /* meters */, tRadius, tRefBurnRate);
    mRocketDriver->initialize(tParams);
}

} // namespace Plato
