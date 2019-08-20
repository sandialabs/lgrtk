#include <Omega_h_file.hpp>

#include "plato/LGR_App.hpp"
#include "plato/PlatoProblemFactory.hpp"

/******************************************************************************/
MPMD_App::MPMD_App(int aArgc, char **aArgv, MPI_Comm& aLocalComm) :
        mObjectiveValue(std::numeric_limits<Plato::Scalar>::max()),
        mConstraintValue(std::numeric_limits<Plato::Scalar>::max()),
        mLibOsh(&aArgc, &aArgv, aLocalComm),
        mMachine(aLocalComm),
        mNumSpatialDims(0),
        mMesh(&mLibOsh)
/******************************************************************************/
{
  // parse app file
  //
  const char* tInputChar = std::getenv("PLATO_APP_FILE");
  Plato::Parser* parser = new Plato::PugiParser();
  mInputData = parser->parseFile(tInputChar);

  auto tInputParams = Plato::input_file_parsing(aArgc, aArgv, mMachine);

  auto problemName = tInputParams.sublist("Runtime").get<std::string>("Input Config");
  mDefaultProblem = Teuchos::rcp(new ProblemDefinition(problemName));
  mDefaultProblem->params = tInputParams;

  this->createProblem(*mDefaultProblem);


  // parse/create the MLS PointArrays
  auto tPointArrayInputs = mInputData.getByName<Plato::InputData>("PointArray");
  for(auto tPointArrayInput=tPointArrayInputs.begin(); tPointArrayInput!=tPointArrayInputs.end(); ++tPointArrayInput)
  {
#ifdef PLATO_GEOMETRY
      auto tPointArrayName = Plato::Get::String(*tPointArrayInput,"Name");
      auto tPointArrayDims = Plato::Get::Int(*tPointArrayInput,"Dimensions");
      if( mMLS.count(tPointArrayName) != 0 )
      {
          throw Plato::ParsingException("PointArray names must be unique.");
      }
      else
      {
          if( tPointArrayDims == 1 )
              mMLS[tPointArrayName] = std::make_shared<MLSstruct>(MLSstruct({Plato::any(Plato::Geometry::MovingLeastSquares<1,double>(*tPointArrayInput)),1}));
          else
          if( tPointArrayDims == 2 )
              mMLS[tPointArrayName] = std::make_shared<MLSstruct>(MLSstruct({Plato::any(Plato::Geometry::MovingLeastSquares<2,double>(*tPointArrayInput)),2}));
          else 
          if( tPointArrayDims == 3 )
              mMLS[tPointArrayName] = std::make_shared<MLSstruct>(MLSstruct({Plato::any(Plato::Geometry::MovingLeastSquares<3,double>(*tPointArrayInput)),3}));
      }
#else
      throw Plato::ParsingException("PlatoApp was not compiled with PointArray support.  Turn on 'PLATO_GEOMETRY' option and rebuild.");
#endif // PLATO_GEOMETRY
  }
}
 
/******************************************************************************/
void
MPMD_App::
createProblem(ProblemDefinition& aDefinition){
/******************************************************************************/

  mCurrentProblemName = aDefinition.name;

  auto input_mesh = aDefinition.params.get<std::string>("Input Mesh");

  mMesh = Omega_h::read_mesh_file(input_mesh, mLibOsh.world());
  mMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

  Omega_h::Assoc tAssoc;
  if (aDefinition.params.isSublist("Associations"))
  {
    auto& tAssocParamList = aDefinition.params.sublist("Associations");
    Omega_h::update_assoc(&tAssoc, tAssocParamList);
  }
  else {
    tAssoc[Omega_h::NODE_SET] = mMesh.class_sets;
    tAssoc[Omega_h::SIDE_SET] = mMesh.class_sets;
  }
  mMeshSets = Omega_h::invert(&mMesh, tAssoc);

  mNumSpatialDims = aDefinition.params.get<int>("Spatial Dimension");

  if (mNumSpatialDims == 3)
  {
    #ifdef PLATO_3D
    Plato::ProblemFactory<3> tProblemFactory;
    mProblem = tProblemFactory.create(mMesh, mMeshSets, aDefinition.params);
    #else
    throw Plato::ParsingException("3D physics is not compiled.");
    #endif
  } else
  if (mNumSpatialDims == 2)
  {
    #ifdef PLATO_2D
    Plato::ProblemFactory<2> tProblemFactory;
    mProblem = tProblemFactory.create(mMesh, mMeshSets, aDefinition.params);
    #else
    throw Plato::ParsingException("2D physics is not compiled.");
    #endif
  } else
  if (mNumSpatialDims == 1)
  {
    #ifdef PLATO_1D
    Plato::ProblemFactory<1> tProblemFactory;
    mProblem = tProblemFactory.create(mMesh, mMeshSets, aDefinition.params);
    #else
    throw Plato::ParsingException("1D physics is not compiled.");
    #endif
  }

  mAdjoint         = mProblem->getAdjoint();
  mState           = mProblem->getState();
  mNumSolutionDofs = mProblem->getNumSolutionDofs();

  aDefinition.modified = false;
}

/******************************************************************************/
void MPMD_App::initialize()
/******************************************************************************/
{

  auto tNumLocalVals = mMesh.nverts();

  mControl    = Plato::ScalarVector("control", tNumLocalVals);

  mObjectiveGradientZ = Plato::ScalarVector("objective_gradient_z", tNumLocalVals);
  mObjectiveGradientX = Plato::ScalarVector("objective_gradient_x", mNumSpatialDims*tNumLocalVals);

  // parse problem definitions
  //
  for( auto opNode : mInputData.getByName<Plato::InputData>("Operation") ){

    std::string strProblem  = Plato::Get::String(opNode,"ProblemDefinition",mDefaultProblem->name);
    auto it = mProblemDefinitions.find(strProblem);
    if(it == mProblemDefinitions.end()){
      auto newProblem = Teuchos::rcp(new ProblemDefinition(strProblem));
      Teuchos::updateParametersFromXmlFileAndBroadcast(
         strProblem, Teuchos::Ptr<Teuchos::ParameterList>(&(newProblem->params)), *(mMachine.teuchosComm));
      mProblemDefinitions[strProblem] = newProblem;
    }
  }
  

  // parse Operation definition
  //
  for( auto tOperationNode : mInputData.getByName<Plato::InputData>("Operation") ){

    std::string tStrFunction = Plato::Get::String(tOperationNode,"Function");
    std::string tStrName     = Plato::Get::String(tOperationNode,"Name");
    std::string tStrProblem  = Plato::Get::String(tOperationNode,"ProblemDefinition",mDefaultProblem->name);

    auto opDef = mProblemDefinitions[tStrProblem];
  
    if(tStrFunction == "ComputeSolution"){
      mOperationMap[tStrName] = new ComputeSolution(this, tOperationNode, opDef);
    } else 

    if(tStrFunction == "Reinitialize"){
      mOperationMap[tStrName] = new Reinitialize(this, tOperationNode, opDef);
    } else 

    if(tStrFunction == "UpdateProblem"){
      mOperationMap[tStrName] = new UpdateProblem(this, tOperationNode, opDef);
    } else

    if(tStrFunction == "ComputeObjective"){
      mOperationMap[tStrName] = new ComputeObjective(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeObjectiveX"){
      mOperationMap[tStrName] = new ComputeObjectiveX(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeObjectiveValue"){
      mOperationMap[tStrName] = new ComputeObjectiveValue(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeObjectiveGradient"){
      mOperationMap[tStrName] = new ComputeObjectiveGradient(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeObjectiveGradientX"){
      mOperationMap[tStrName] = new ComputeObjectiveGradientX(this, tOperationNode, opDef);
    } else 

    if(tStrFunction == "ComputeConstraint"){
      mOperationMap[tStrName] = new ComputeConstraint(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeConstraintX"){
      mOperationMap[tStrName] = new ComputeConstraintX(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeConstraintValue"){
      mOperationMap[tStrName] = new ComputeConstraintValue(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeConstraintGradient"){
      mOperationMap[tStrName] = new ComputeConstraintGradient(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeConstraintGradientX"){
      mOperationMap[tStrName] = new ComputeConstraintGradientX(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "WriteOutput"){
      mOperationMap[tStrName] = new WriteOutput(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeFiniteDifference"){
      mOperationMap[tStrName] = new ComputeFiniteDifference(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeMLSField"){
  #ifdef PLATO_GEOMETRY
      auto tMLSName = Plato::Get::String(tOperationNode,"MLSName");
      if( mMLS.count(tMLSName) == 0 )
      { throw Plato::ParsingException("MPMD_App::ComputeMLSField: Requested a PointArray that isn't defined."); }

      if( mCoords.extent(0) == 0 )
      {
        mCoords = getCoords();
      }

      auto tMLS = mMLS[tMLSName];
      if( tMLS->dimension == 3 ) { mOperationMap[tStrName] = new ComputeMLSField<3>(this, tOperationNode, opDef); }
      else
      if( tMLS->dimension == 2 ) { mOperationMap[tStrName] = new ComputeMLSField<2>(this, tOperationNode, opDef); }
      else
      if( tMLS->dimension == 1 ) { mOperationMap[tStrName] = new ComputeMLSField<1>(this, tOperationNode, opDef); }
  #else
      throw Plato::ParsingException("MPMD_App was not compiled with ComputeMLSField enabled.  Turn on 'PLATO_GEOMETRY' option and rebuild.");
  #endif // PLATO_GEOMETRY
    } else 
    if(tStrFunction == "ComputePerturbedMLSField"){
  #ifdef PLATO_GEOMETRY
      auto tMLSName = Plato::Get::String(tOperationNode,"MLSName");
      if( mMLS.count(tMLSName) == 0 )
      { throw Plato::ParsingException("MPMD_App::ComputePerturbedMLSField: Requested a PointArray that isn't defined."); }

      if( mCoords.extent(0) == 0 )
      {
        mCoords = getCoords();
      }

      auto tMLS = mMLS[tMLSName];
      if( tMLS->dimension == 3 ) { mOperationMap[tStrName] = new ComputePerturbedMLSField<3>(this, tOperationNode, opDef); }
      else
      if( tMLS->dimension == 2 ) { mOperationMap[tStrName] = new ComputePerturbedMLSField<2>(this, tOperationNode, opDef); }
      else
      if( tMLS->dimension == 1 ) { mOperationMap[tStrName] = new ComputePerturbedMLSField<1>(this, tOperationNode, opDef); }
  #else
      throw Plato::ParsingException("MPMD_App was not compiled with ComputePerturbedMLSField enabled.  Turn on 'PLATO_GEOMETRY' option and rebuild.");
  #endif // PLATO_GEOMETRY
    }
  }
}

/******************************************************************************/
MPMD_App::LocalOp*
MPMD_App::getOperation(const std::string & aOperationName)
/******************************************************************************/
{
  auto tIterator = mOperationMap.find(aOperationName);
  if(tIterator == mOperationMap.end()){
    std::stringstream tErrorMsg;
    tErrorMsg << "Request for operation ('" << aOperationName << "') that doesn't exist.";
    throw Plato::LogicException(tErrorMsg.str());
  }
  return tIterator->second;
}

/******************************************************************************/
void
MPMD_App::compute(const std::string & aOperationName)
/******************************************************************************/
{
  LocalOp *tOperation = this->getOperation(aOperationName);

  // if a different problem definition is needed, create it
  //
  auto tProblemDefinition = tOperation->getProblemDefinition();
  if( tProblemDefinition->name != mCurrentProblemName || tProblemDefinition->modified  )
  {
    this->createProblem(*tProblemDefinition);
  }

  // call the operation
  //
  (*tOperation)();
}

/******************************************************************************/
MPMD_App::LocalOp::
LocalOp(MPMD_App* aMyApp, Plato::InputData& aOperationNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        mMyApp(aMyApp),
        mDef(aOpDef)
/******************************************************************************/
{
  // parse parameters
  for( auto &pNode : aOperationNode.getByName<Plato::InputData>("Parameter") )
  {
    auto tName   = Plato::Get::String(pNode, "ArgumentName");
    auto tTarget = Plato::Get::String(pNode, "Target");
    auto tValue  = Plato::Get::Double(pNode, "InitialValue");
    
    if( mParameters.count(tName) )
    {
      Plato::ParsingException pe("ArgumentNames must be unique.");
      throw pe;
    }

    mParameters[tName] = Teuchos::rcp(new Parameter(tName, tTarget, tValue));
  }
}

/******************************************************************************/
void
MPMD_App::LocalOp::
updateParameters(std::string aName, Plato::Scalar aValue)
/******************************************************************************/
{
  if( mParameters.count(aName) == 0 )
  {
    std::stringstream ss;
    ss << "Attempted to update a parameter ('" << aName << "') that wasn't defined for this operation";
    Plato::ParsingException pe(ss.str());
    throw pe;
  } 
  else
  {
    auto it = mParameters.find(aName);
    auto pm = it->second;
    pm->mValue = aValue;

    // if a target is given, update the problem definition
    if ( pm->mTarget.empty() == false )
    {
        parseInline(mDef->params, pm->mTarget, pm->mValue);
        mDef->modified = true;
    }
  }
}

/******************************************************************************/
MPMD_App::ComputeObjective::
ComputeObjective(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjective::operator()()
/******************************************************************************/
{
  mMyApp->mState = mMyApp->mProblem->solution(mMyApp->mControl);

  mMyApp->mObjectiveValue      = mMyApp->mProblem->objectiveValue(mMyApp->mControl, mMyApp->mState);
  mMyApp->mObjectiveGradientZ = mMyApp->mProblem->objectiveGradient(mMyApp->mControl, mMyApp->mState);
}

/******************************************************************************/
MPMD_App::ComputeObjectiveX::
ComputeObjectiveX(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjectiveX::operator()()
/******************************************************************************/
{
  mMyApp->mState = mMyApp->mProblem->solution(mMyApp->mControl);

  mMyApp->mObjectiveValue      = mMyApp->mProblem->objectiveValue(mMyApp->mControl, mMyApp->mState);
  mMyApp->mObjectiveGradientX = mMyApp->mProblem->objectiveGradientX(mMyApp->mControl, mMyApp->mState);
}


/******************************************************************************/
MPMD_App::ComputeObjectiveValue::
ComputeObjectiveValue(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjectiveValue::operator()()
/******************************************************************************/
{
  mMyApp->mState = mMyApp->mProblem->solution(mMyApp->mControl);
  mMyApp->mObjectiveValue = mMyApp->mProblem->objectiveValue(mMyApp->mControl,mMyApp->mState);
}

/******************************************************************************/
MPMD_App::ComputeObjectiveGradient::
ComputeObjectiveGradient(MPMD_App* aMyApp, Plato::InputData& aOpNode,  Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}

/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjectiveGradient::operator()()
/******************************************************************************/
{
  mMyApp->mObjectiveGradientZ = mMyApp->mProblem->objectiveGradient(mMyApp->mControl, mMyApp->mState);
}

/******************************************************************************/
MPMD_App::ComputeObjectiveGradientX::
ComputeObjectiveGradientX(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjectiveGradientX::operator()()
/******************************************************************************/
{
  mMyApp->mObjectiveGradientX = mMyApp->mProblem->objectiveGradientX(mMyApp->mControl, mMyApp->mState);
}

/******************************************************************************/
MPMD_App::ComputeConstraint::
ComputeConstraint(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraint::operator()()
/******************************************************************************/
{
  mMyApp->mConstraintValue      = mMyApp->mProblem->constraintValue(mMyApp->mControl, mMyApp->mState);
  mMyApp->mConstraintGradientZ = mMyApp->mProblem->constraintGradient(mMyApp->mControl, mMyApp->mState);

  std::cout << "Plato:: Constraint value = " << mMyApp->mConstraintValue << std::endl;
}

/******************************************************************************/
MPMD_App::ComputeConstraintX::
ComputeConstraintX(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                  Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraintX::operator()()
/******************************************************************************/
{
  mMyApp->mConstraintValue      = mMyApp->mProblem->constraintValue(mMyApp->mControl, mMyApp->mState);
  mMyApp->mConstraintGradientX = mMyApp->mProblem->constraintGradientX(mMyApp->mControl, mMyApp->mState);

  std::cout << "Plato:: Constraint value = " << mMyApp->mConstraintValue << std::endl;
}


/******************************************************************************/
MPMD_App::ComputeConstraintValue::
ComputeConstraintValue(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraintValue::operator()()
/******************************************************************************/
{
  mMyApp->mConstraintValue = mMyApp->mProblem->constraintValue(mMyApp->mControl,mMyApp->mState);

  std::cout << "Plato:: Constraint value = " << mMyApp->mConstraintValue << std::endl;
}

/******************************************************************************/
MPMD_App::ComputeConstraintGradient::
ComputeConstraintGradient(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraintGradient::operator()()
/******************************************************************************/
{
  mMyApp->mConstraintGradientZ = mMyApp->mProblem->constraintGradient(mMyApp->mControl, mMyApp->mState);
}

/******************************************************************************/
MPMD_App::ComputeConstraintGradientX::
ComputeConstraintGradientX(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraintGradientX::operator()()
/******************************************************************************/
{
  mMyApp->mConstraintGradientX = mMyApp->mProblem->constraintGradientX(mMyApp->mControl, mMyApp->mState);
}

/******************************************************************************/
MPMD_App::ComputeSolution::
ComputeSolution(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeSolution::operator()()
/******************************************************************************/
{
  mMyApp->mState = mMyApp->mProblem->solution(mMyApp->mControl);
}

/******************************************************************************/
MPMD_App::Reinitialize::
Reinitialize(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::Reinitialize::operator()()
/******************************************************************************/
{
  auto def = mMyApp->mProblemDefinitions[mMyApp->mCurrentProblemName];
  mMyApp->createProblem(*def);
}


/******************************************************************************/
MPMD_App::UpdateProblem::
UpdateProblem(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}

/******************************************************************************/

/******************************************************************************/
void MPMD_App::UpdateProblem::operator()()
/******************************************************************************/
{
    mMyApp->mProblem->updateProblem(mMyApp->mControl, mMyApp->mState);
}

/******************************************************************************/
MPMD_App::WriteOutput::
WriteOutput(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::WriteOutput::operator()() { }
/******************************************************************************/

/******************************************************************************/
MPMD_App::ComputeFiniteDifference::
ComputeFiniteDifference(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef),
        mStrInitialValue("Initial Value"),
        mStrPerturbedValue("Perturbed Value"),
        mStrGradient("Gradient")
/******************************************************************************/
{ 
    aMyApp->mValuesMap[mStrInitialValue] = std::vector<Plato::Scalar>();
    aMyApp->mValuesMap[mStrPerturbedValue] = std::vector<Plato::Scalar>();

    int tNumTerms = std::round(mParameters["Vector Length"]->mValue);
    aMyApp->mValuesMap[mStrGradient] = std::vector<Plato::Scalar>(tNumTerms);

    mDelta = Plato::Get::Double(aOpNode,"Delta");
}

/******************************************************************************/
void MPMD_App::ComputeFiniteDifference::operator()() 
/******************************************************************************/
{ 
    int tIndex=0;
    Plato::Scalar tDelta=0.0;
    if( mParameters.count("Perturbed Index") )
    {
        tIndex = std::round(mParameters["Perturbed Index"]->mValue);
        tDelta = mDelta;
    }

    auto& outVector = mMyApp->mValuesMap[mStrGradient];
    auto tPerturbedValue = mMyApp->mValuesMap[mStrPerturbedValue][0];
    auto tInitialValue = mMyApp->mValuesMap[mStrInitialValue][0];
    outVector[tIndex] = (tPerturbedValue - tInitialValue) / tDelta;
}

/******************************************************************************/
void MPMD_App::finalize() { }
/******************************************************************************/


/******************************************************************************/
void MPMD_App::importData(const std::string& aName, const Plato::SharedData& aSharedField)
/******************************************************************************/
{
    this->importDataT(aName, aSharedField);
}


/******************************************************************************/
void MPMD_App::exportData(const std::string& aName, Plato::SharedData& aSharedField)
/******************************************************************************/
{
    this->exportDataT(aName, aSharedField);
}

/******************************************************************************/
void MPMD_App::exportDataMap(const Plato::data::layout_t & aDataLayout, std::vector<int> & aMyOwnedGlobalIDs)
/******************************************************************************/
{
    if(aDataLayout == Plato::data::layout_t::SCALAR_FIELD)
    {
        Plato::OrdinalType tNumLocalVals = mMesh.nverts();
        aMyOwnedGlobalIDs.resize(tNumLocalVals);
        for(Plato::OrdinalType tLocalID = 0; tLocalID < tNumLocalVals; tLocalID++)
        {
            aMyOwnedGlobalIDs[tLocalID] = tLocalID + 1;
        }
    }
    else if(aDataLayout == Plato::data::layout_t::ELEMENT_FIELD)
    {
        Plato::OrdinalType tNumLocalVals = mMesh.nelems();
        aMyOwnedGlobalIDs.resize(tNumLocalVals);
        for(Plato::OrdinalType tLocalID = 0; tLocalID < tNumLocalVals; tLocalID++)
        {
            aMyOwnedGlobalIDs[tLocalID] = tLocalID + 1;
        }
    }
    else
    {
        Plato::ParsingException tParsingException("lgrMPMD currently only supports SCALAR_FIELD and ELEMENT_FIELD data layout");
        throw tParsingException;
    }
}

/******************************************************************************/
Plato::ScalarVector
getVectorComponent(Plato::ScalarVector aFrom, int aComponent, int aStride)
/******************************************************************************/
{
  int tNumLocalVals = aFrom.size()/aStride;
  Plato::ScalarVector tRetVal("vector component", tNumLocalVals);
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumLocalVals), LAMBDA_EXPRESSION(const int & aNodeOrdinal) {
    tRetVal(aNodeOrdinal) = aFrom(aStride*aNodeOrdinal+aComponent);
  },"copy component from vector");
  return tRetVal;
}

/******************************************************************************/
Plato::ScalarVector
setVectorComponent(Plato::ScalarVector aFrom, int aComponent, int aStride)
/******************************************************************************/
{
  int tNumLocalVals = aFrom.size()/aStride;
  Plato::ScalarVector tRetVal("vector component", tNumLocalVals);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumLocalVals), LAMBDA_EXPRESSION(const int & aNodeOrdinal) {
    tRetVal(aNodeOrdinal) = aFrom(aStride*aNodeOrdinal+aComponent);
  },"copy component from vector");
  return tRetVal;
}


/******************************************************************************/
MPMD_App::~MPMD_App()
/******************************************************************************/
{
}

/******************************************************************************/
std::vector<std::string>
split( const std::string& aInputString, const char aDelimiter )
/******************************************************************************/
{
  // break aInputString apart by 'aDelimiter' below //
  // produces a vector of strings: tTokens   //
  std::vector<std::string> tTokens;
  {
    std::istringstream tStream(aInputString);
    std::string tToken;
    while (std::getline(tStream, tToken, aDelimiter))
    {
      tTokens.push_back(tToken);
    }
  }
  return tTokens;
}
/******************************************************************************/
void 
parseInline( Teuchos::ParameterList& params, 
             const std::string& target, 
             Plato::Scalar value )
/******************************************************************************/
{
  std::vector<std::string> tokens = split(target,':');

  Teuchos::ParameterList& innerList = getInnerList(params, tokens);
  setParameterValue(innerList, tokens, value);

}

/******************************************************************************/
Teuchos::ParameterList&
getInnerList( Teuchos::ParameterList& params, 
              std::vector<std::string>& tokens)
/******************************************************************************/
{
    auto& token = tokens[0];
    if( token.front() == '[' && token.back()  == ']' )
    {
      // listName = token with '[' and ']' removed.
      std::string listName = token.substr(1,token.size()-2);
      tokens.erase(tokens.begin());
      return getInnerList( params.sublist(listName, /*must exist=*/true), tokens );
    } 
    else 
    {
      return params;
    }
}
/******************************************************************************/
void
setParameterValue( Teuchos::ParameterList& params, 
                   std::vector<std::string> tokens, Plato::Scalar value)
/******************************************************************************/
{
  // if '(int)' then
  auto& token = tokens[0];
  auto p1 = token.find("(");
  auto p2 = token.find(")");
  if( p1 != string::npos && p2 != string::npos )
  {
      std::string vecName = token.substr(0,p1);
      auto vec = params.get<Teuchos::Array<double>>(vecName);

      std::string strVecEntry = token.substr(p1+1,p2-p1-1);
      int vecEntry = std::stoi(strVecEntry);
      vec[vecEntry] = value;

      params.set(vecName,vec);
  }
  else
  {
      params.set<double>(token,value);
  }
}

/******************************************************************************/
Plato::ScalarMultiVector MPMD_App::getCoords()
/******************************************************************************/
{
    auto tCoords = mMesh.coords();
    auto tNumVerts = mMesh.nverts();
    auto tNumDims = mMesh.dim();
    Plato::ScalarMultiVector retval("coords", tNumVerts, tNumDims);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumVerts), LAMBDA_EXPRESSION(const Plato::OrdinalType & tVertOrdinal){
        for (int iDim=0; iDim<tNumDims; iDim++){
            retval(tVertOrdinal,iDim) = tCoords[tVertOrdinal*tNumDims+iDim];
        }
    }, "get coordinates");

    return retval;
}

