/*
 * PlatoRocketApp.cpp
 *
 *  Created on: Nov 29, 2018
 */

#include <numeric>
#include <fstream>

#include "plato/PlatoRocketApp.hpp"
#include "plato/Plato_LevelSetCylinderInBox.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Default constructor
**********************************************************************************/
RocketApp::RocketApp() :
        mComm(MPI_COMM_NULL),
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

void RocketApp::printSolution()
{
    Plato::OrdinalType tMyProcID = 0;
    MPI_Comm_rank(mComm, &tMyProcID);
    if(tMyProcID == static_cast<Plato::OrdinalType>(0))
    {
        std::string tArgumentName = "DesignVariables";
        auto tIterator = mSharedDataMap.find(tArgumentName);
        assert(tIterator != mSharedDataMap.end());
        std::vector<Plato::Scalar> & tControl = tIterator->second;

        tArgumentName = "NormalizationConstants";
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
    const Plato::Scalar tChamberRadius = 0.075; // m
    const Plato::Scalar tChamberLength = 0.65; // m
    std::shared_ptr<Plato::LevelSetCylinderInBox<Plato::Scalar>> tGeomModel =
            std::make_shared<Plato::LevelSetCylinderInBox<Plato::Scalar>>(tChamberRadius, tChamberLength, mComm);

    Plato::AlgebraicRocketInputs<Plato::Scalar> tDefaultInputs;
    mRocketDriver = std::make_shared<Plato::AlgebraicRocketModel<Plato::Scalar>>(tDefaultInputs, tGeomModel);
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
    std::vector<Plato::Scalar> tThrustProfile =
    {0, 1658225.96301779104, 1684599.35308251064, 1710191.72929578833,
     1735516.51746973093, 1761319.78971856111, 1788474.32299472415,
     1817051.74281903892, 1848053.60778497672, 1880941.49518394633,
     1916101.49896604964, 1952274.23182648909, 1988247.08371756272,
     2025364.49255605415, 2062680.45754474145, 2098144.24305555876,
     2133059.58457374899, 2168280.54717646865, 2202359.06706217164,
     2235786.63485889323, 2269362.61600903422, 2303606.27047893312,
     2335968.55831982987, 2366124.24693903141, 2398322.21222466091,
     2434967.60986522399, 2474766.42476609442, 2515758.39447587077,
     2555562.94606091594, 2598351.59772849502, 2643361.82917793794,
     2689612.44766329601, 2736886.53526937729, 2784774.40230861725,
     2833698.64203229547, 2882691.37882861448, 2930314.53433794295,
     2973845.2476535202, 3014411.95485122781, 3053344.35515435785,
     3093609.95011549396, 3136012.24460697407, 3179548.32224763138,
     3224104.45371416444, 3272727.42383094644, 3325044.09918975038,
     3375921.5864060251, 3426710.62721839268, 3478150.53201142279,
     3535954.48766951077, 3600047.85890252888, 3666681.63734701369,
     3731431.69178802939, 3790535.23900091508, 3845553.0628477498,
     3897589.02069075312, 3944554.81866455032, 3994468.19636945101,
     4050443.68018117454, 4106261.75979240146, 4161161.0958835152,
     4220500.30987357534, 4281020.20508432388, 4338719.01243665628,
     4398616.88256580941, 4467635.24324372411, 4546787.37910916004,
     4626455.44047849253, 4707397.23780253809, 4783792.29538778123,
     4854935.16257551499, 4919399.44213557336, 4977671.61547120474,
     5041505.24270366877, 5113173.07470646407, 5182463.6731977351,
     5246042.58420176338, 5308069.2641470544, 5378333.32595723867,
     5452794.42908227444, 5530334.15620550793, 5614074.4806811465,
     5709559.24524913542, 5807675.62193106115, 5901984.70578025375,
     5987027.20662646182, 6072012.90696318354, 6156511.17011062615,
     6237224.14467318635, 6314341.17838649545, 6394109.84751804452,
     6472279.52941741887, 6541642.18780869804, 6620624.62667534687,
     6714259.09761043079, 6809373.66328622587, 6909013.93939755578,
     7016550.60471259616, 7124790.91800781339, 7230958.52803643327};

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
    const Plato::OrdinalType tNumControls = aControl.size();
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
