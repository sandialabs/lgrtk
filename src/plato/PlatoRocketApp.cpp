/*
 * PlatoRocketApp.cpp
 *
 *  Created on: Nov 29, 2018
 */

#include <numeric>
#include <fstream>

#include "plato/Plato_Cylinder.hpp"
#include "plato/PlatoRocketApp.hpp"
#include "plato/Plato_RocketMocks.hpp"
#include "plato/Plato_LevelSetCylinderInBox.hpp"

namespace Plato
{

RocketApp::RocketApp() :
        mComm(MPI_COMM_NULL),
        mLength(0.65),
        mMaxRadius(0.15),
        mNumDesigVariables(2),
        mDefinedOperations(),
        mSharedDataMap(),
        mDefinedDataLayout()
{
}

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

RocketApp::~RocketApp()
{
}

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

void RocketApp::finalize()
{
    // MEMORY MANAGEMENT AUTOMATED, NO NEED TO DEALLOCATE MEMORY
    return;
}

void RocketApp::initialize()
{
    this->defineOperations();
    this->defineSharedDataMaps();
    this->setRocketDriver();
}

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

void RocketApp::exportDataMap(const Plato::data::layout_t & aDataLayout, std::vector<int> & aMyOwnedGlobalIDs)
{
    // THIS IS NOT A DISTRIBUTED MEMORY EXAMPLE; HENCE, THE DISTRIBUTED MEMORY GRAPH IS NOT NEEDEDS
    return;
}

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

void RocketApp::setRocketDriver()
{
    const Plato::Scalar tRefBurnRate = 0.005;  // meters/seconds
    const Plato::Scalar tInitialRadius = 0.075; // meters
    Plato::ProblemParams tParams =
            Plato::RocketMocks::set_constant_burn_rate_problem(mMaxRadius /* meters */, mLength /* meters */, tInitialRadius, tRefBurnRate);

    std::shared_ptr<Plato::LevelSetCylinderInBox> tGeometry =
            std::make_shared<Plato::LevelSetCylinderInBox>(mComm);
    //std::shared_ptr<Plato::Cylinder> tGeometry = std::make_shared<Plato::Cylinder>();
    tGeometry->initialize(tParams);

    Plato::AlgebraicRocketInputs tDefaulRocketParams;
    mRocketDriver = std::make_shared<Plato::AlgebraicRocketModel>(tDefaulRocketParams, tGeometry);
    mRocketDriver->disableOutput();
}

void RocketApp::defineOperations()
{
    mDefinedOperations.push_back("ObjectiveValue");
    mDefinedOperations.push_back("ObjectiveGradient");
}

void RocketApp::setDefaultTargetThrustProfile()
{
    std::vector<Plato::Scalar> tThrustProfile = {0, 6811.91690423622822, 6885.75629631850279, 6952.80491651654029,
                                                 7017.05814957706752, 7083.90155173868243, 7139.96284038090107,
                                                 7207.84031753954514, 7286.3624792566934, 7369.61968626376711,
                                                 7459.09358575368242, 7543.00663303837609, 7625.06224278821719,
                                                 7696.57732657527322, 7759.9070759998267, 7823.38092406913165,
                                                 7895.74663761342708, 7979.07885666436869, 8063.01389036155979,
                                                 8151.88822875917685, 8253.97082344292539, 8343.75291677611494,
                                                 8416.37169315791834, 8481.37346020181758, 8554.14665395097109,
                                                 8635.04996239496359, 8718.72354036365323, 8801.27848935129077,
                                                 8910.26839961372571, 9016.72491051532961, 9108.77263991399377,
                                                 9175.79386125828205, 9257.07582946822367, 9337.0534037237976,
                                                 9417.03606775631124, 9502.97355832670291, 9608.53255978156994,
                                                 9727.48452512686708, 9829.6347613188882, 9913.95207966482303,
                                                 9998.47227806756564, 10082.7715570189594, 10158.0844981374939,
                                                 10251.7522025108574, 10359.0657085234579, 10478.3022994054427,
                                                 10588.4025754307913, 10694.3858152271368, 10780.4154307675235,
                                                 10867.7878116023749, 10941.1468697820219, 11049.6619294777374,
                                                 11155.2494921476464, 11274.1735154618073, 11398.985313122872,
                                                 11515.9892578788749, 11604.3667020658631, 11686.1495558141487,
                                                 11781.0473965404399, 11889.3531766786909, 11995.4752138031854,
                                                 12123.6764025002776, 12267.2058372193351, 12375.9746187227975,
                                                 12461.2189743510062, 12562.1855840869339, 12662.9800788652028,
                                                 12767.2133765787075, 12890.2714132920992, 13040.5308747085692,
                                                 13173.6968724954404, 13275.1190948565691, 13378.7605363790335,
                                                 13481.9707481041041, 13579.0971048113624, 13705.540231900819,
                                                 13852.23315480778, 13998.6773689125039, 14120.7957099083615,
                                                 14235.6529321930357, 14336.2319927665194, 14434.8115326157258,
                                                 14565.7502529387657, 14709.5074274288963, 14856.3276478352745,
                                                 15008.1131046688461, 15125.0232340858311, 15226.3601990398201,
                                                 15332.3406496823736, 15471.593916853928, 15601.2604246407664,
                                                 15771.4234168044113, 15931.7951927888607, 16050.1193561083237,
                                                 16154.5487617903273, 16277.5638116633618, 16407.8758314797378,
                                                 16546.4189302997438, 16731.4158687006748, 16895.771919135972};

    this->setMaxTargetThrust(tThrustProfile);
    this->setTargetThrustProfile(tThrustProfile);
    this->setNormTargetThrustProfile(tThrustProfile);
}

void RocketApp::setMaxTargetThrust(const std::vector<Plato::Scalar> & aThrustProfile)
{
    std::vector<Plato::Scalar> tMaxThrust = { *std::max_element(aThrustProfile.begin(), aThrustProfile.end()) };
    std::string tArgumentName = "MaxThrust";
    mSharedDataMap[tArgumentName] = tMaxThrust;
    mDefinedDataLayout[tArgumentName] = Plato::data::SCALAR;
}

void RocketApp::setNormTargetThrustProfile(const std::vector<Plato::Scalar> & aThrustProfile)
{
    std::vector<Plato::Scalar> tNormThrustProfile =
            {std::inner_product(aThrustProfile.begin(), aThrustProfile.end(), aThrustProfile.begin(), 0.0)};
    std::string tArgumentName = "NormThrustProfile";
    mSharedDataMap[tArgumentName] = tNormThrustProfile;
    mDefinedDataLayout[tArgumentName] = Plato::data::SCALAR;
}

void RocketApp::setTargetThrustProfile(const std::vector<Plato::Scalar> & aThrustProfile)
{
    std::string tArgumentName = "TargetThrustProfile";
    mSharedDataMap[tArgumentName] = aThrustProfile;
    mDefinedDataLayout[tArgumentName] = Plato::data::SCALAR;
}

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

void RocketApp::setNormalizationConstants()
{
    std::vector<Plato::Scalar> tValues(mNumDesigVariables);
    tValues[0] = 0.1; tValues[1] = 0.01;
    std::string tName = "Normalization";
    mSharedDataMap[tName] = tValues;
    mDefinedDataLayout[tName] = Plato::data::SCALAR;
}

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

Plato::Scalar RocketApp::getMaxTargetThrust() const
{
    std::string tArgumentName = "MaxThrust";
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    return ((tIterator->second)[0]);
}

Plato::Scalar RocketApp::getNormTargetThrustProfile() const
{
    std::string tArgumentName = "NormThrustProfile";
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    return ((tIterator->second)[0]);
}

Plato::Scalar RocketApp::computeObjFuncValue(const std::vector<Plato::Scalar> & aControl,
                                             const std::vector<Plato::Scalar> & aTargetProfile)
{
    this->updateProblem(aControl);
    mRocketDriver->solve();
    auto tTrialThrustProfile = mRocketDriver->getThrustProfile();
    assert(tTrialThrustProfile.size() == aTargetProfile.size());

    Plato::Scalar tObjFuncValue = 0;
    for(Plato::OrdinalType tIndex = 0; tIndex < aTargetProfile.size(); tIndex++)
    {
        Plato::Scalar tDeltaThrust = tTrialThrustProfile[tIndex] - aTargetProfile[tIndex];
        tObjFuncValue += tDeltaThrust * tDeltaThrust;
    }

    const Plato::Scalar tNumElements = aTargetProfile.size();
    const Plato::Scalar tNormThrustProfile = this->getNormTargetThrustProfile();
    const Plato::Scalar tDenominator = static_cast<Plato::Scalar>(2.0) * tNumElements * tNormThrustProfile;
    tObjFuncValue = (static_cast<Plato::Scalar>(1.0) / tDenominator) * tObjFuncValue;

    return (tObjFuncValue);
}

void RocketApp::updateProblem(const std::vector<Plato::Scalar> & aControls)
{
    std::string tArgumentName = "Normalization";
    auto tIterator = mSharedDataMap.find(tArgumentName);
    assert(tIterator != mSharedDataMap.end());
    std::vector<Plato::Scalar> & tNormalizationConstants = tIterator->second;

    auto tRadius = aControls[0] * tNormalizationConstants[0]; // meters
    auto tRefBurnRate = aControls[1] * tNormalizationConstants[1]; // meters/seconds
    auto tParams = Plato::RocketMocks::set_constant_burn_rate_problem(mMaxRadius /* meters */, mLength /* meters */, tRadius, tRefBurnRate);
    mRocketDriver->initialize(tParams);
}

} // namespace Plato
