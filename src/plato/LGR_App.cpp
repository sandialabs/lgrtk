#include <Omega_h_file.hpp>

#include "plato/LGR_App.hpp"
#include "plato/PlatoProblemFactory.hpp"

/******************************************************************************/
MPMD_App::MPMD_App(int aArgc, char **aArgv, MPI_Comm& aLocalComm) :
        m_objective_value(std::numeric_limits<Plato::Scalar>::max()),
        m_constraint_value(std::numeric_limits<Plato::Scalar>::max()),
        m_lib_osh(&aArgc, &aArgv, aLocalComm),
        m_machine(aLocalComm),
        m_numSpatialDims(0),
        mMesh(&m_lib_osh)
/******************************************************************************/
{
  // parse app file
  //
  const char* tInputChar = std::getenv("PLATO_APP_FILE");
  Plato::Parser* parser = new Plato::PugiParser();
  m_inputData = parser->parseFile(tInputChar);

  auto tInputParams = lgr::input_file_parsing(aArgc, aArgv, m_machine);

  auto problemName = tInputParams.sublist("Runtime").get<std::string>("Input Config");
  m_defaultProblem = Teuchos::rcp(new ProblemDefinition(problemName));
  m_defaultProblem->params = tInputParams;

  createProblem(*m_defaultProblem);
}
 
/******************************************************************************/
void
MPMD_App::
createProblem(ProblemDefinition& definition){
/******************************************************************************/

  m_currentProblemName = definition.name;

  auto input_mesh = definition.params.get<std::string>("Input Mesh");
  auto& assoc_pl  = definition.params.sublist("Associations");

  // Read input mesh
  mMesh = Omega_h::binary::read(input_mesh, &m_lib_osh);
  mMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

  // Compute mesh sets - hold side sets information
  Omega_h::update_assoc(&mAssoc, assoc_pl);
  mMeshSets = Omega_h::invert(&mMesh, mAssoc);

  m_numSpatialDims = definition.params.get<int>("Spatial Dimension");

  if (m_numSpatialDims == 3)
  {
    Plato::ProblemFactory<3> tProblemFactory;
    m_problem = tProblemFactory.create(mMesh, mMeshSets, definition.params);
    m_adjoint = m_problem->getAdjoint();
    m_state = m_problem->getState();
  } else
  if (m_numSpatialDims == 2)
  {
    Plato::ProblemFactory<2> tProblemFactory;
    m_problem = tProblemFactory.create(mMesh, mMeshSets, definition.params);
    m_adjoint = m_problem->getAdjoint();
    m_state = m_problem->getState();
  } else
  if (m_numSpatialDims == 1)
  {
    Plato::ProblemFactory<1> tProblemFactory;
    m_problem = tProblemFactory.create(mMesh, mMeshSets, definition.params);
    m_adjoint = m_problem->getAdjoint();
    m_state = m_problem->getState();
  }

  definition.modified = false;
}

/******************************************************************************/
void MPMD_App::initialize()
/******************************************************************************/
{

  auto tNumLocalVals = mMesh.nverts();

  m_control    = Plato::ScalarVector("control", tNumLocalVals);

  m_objective_gradient_z = Plato::ScalarVector("objective_gradient_z", tNumLocalVals);
  m_objective_gradient_x = Plato::ScalarVector("objective_gradient_x", m_numSpatialDims*tNumLocalVals);

  // parse problem definitions
  //
  for( auto opNode : m_inputData.getByName<Plato::InputData>("Operation") ){

    std::string strProblem  = Plato::Get::String(opNode,"ProblemDefinition",m_defaultProblem->name);
    auto it = m_problemDefinitions.find(strProblem);
    if(it == m_problemDefinitions.end()){
      auto newProblem = Teuchos::rcp(new ProblemDefinition(strProblem));
      Teuchos::updateParametersFromXmlFileAndBroadcast(
         strProblem, Teuchos::Ptr<Teuchos::ParameterList>(&(newProblem->params)), *(m_machine.teuchosComm));
      m_problemDefinitions[strProblem] = newProblem;
    }
  }
  

  // parse Operation definition
  //
  for( auto tOperationNode : m_inputData.getByName<Plato::InputData>("Operation") ){

    std::string tStrFunction = Plato::Get::String(tOperationNode,"Function");
    std::string tStrName     = Plato::Get::String(tOperationNode,"Name");
    std::string tStrProblem  = Plato::Get::String(tOperationNode,"ProblemDefinition",m_defaultProblem->name);

    auto opDef = m_problemDefinitions[tStrProblem];
  
    if(tStrFunction == "ComputeSolution"){
      m_operationMap[tStrName] = new ComputeSolution(this, tOperationNode, opDef);
    } else 

    if(tStrFunction == "Reinitialize"){
      m_operationMap[tStrName] = new Reinitialize(this, tOperationNode, opDef);
    } else 

    if(tStrFunction == "ComputeObjective"){
      m_operationMap[tStrName] = new ComputeObjective(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeObjectiveX"){
      m_operationMap[tStrName] = new ComputeObjectiveX(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeObjectiveValue"){
      m_operationMap[tStrName] = new ComputeObjectiveValue(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeObjectiveGradient"){
      m_operationMap[tStrName] = new ComputeObjectiveGradient(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeObjectiveGradientX"){
      m_operationMap[tStrName] = new ComputeObjectiveGradientX(this, tOperationNode, opDef);
    } else 

    if(tStrFunction == "ComputeConstraint"){
      m_operationMap[tStrName] = new ComputeConstraint(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeConstraintX"){
      m_operationMap[tStrName] = new ComputeConstraintX(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeConstraintValue"){
      m_operationMap[tStrName] = new ComputeConstraintValue(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeConstraintGradient"){
      m_operationMap[tStrName] = new ComputeConstraintGradient(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "ComputeConstraintGradientX"){
      m_operationMap[tStrName] = new ComputeConstraintGradientX(this, tOperationNode, opDef);
    } else 
    if(tStrFunction == "WriteOutput"){
      m_operationMap[tStrName] = new WriteOutput(this, tOperationNode, opDef);
    }
  }
}

/******************************************************************************/
MPMD_App::LocalOp*
MPMD_App::getOperation(const std::string & aOperationName)
/******************************************************************************/
{
  auto tIterator = m_operationMap.find(aOperationName);
  if(tIterator == m_operationMap.end()){
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
  LocalOp *op = getOperation(aOperationName);

  // if a different problem definition is needed, create it
  //
  auto def = op->getProblemDefinition();
  if( def->name != m_currentProblemName || def->modified  )
  {
    createProblem(*def);
  }

  // call the operation
  //
  (*op)();
}

/******************************************************************************/
MPMD_App::LocalOp::
LocalOp(MPMD_App* aMyApp, Plato::InputData& aOperationNode, 
        Teuchos::RCP<ProblemDefinition> aOpDef) : mMyApp(aMyApp), m_def(aOpDef)
/******************************************************************************/
{
  // parse parameters
  for( auto &pNode : aOperationNode.getByName<Plato::InputData>("Parameter") )
  {
    auto tName   = Plato::Get::String(pNode, "ArgumentName");
    auto tTarget = Plato::Get::String(pNode, "Target");
    auto tValue  = Plato::Get::Double(pNode, "InitialValue");
    
    if( m_parameters.count(tName) )
    {
      Plato::ParsingException pe("ArgumentNames must be unique.");
      throw pe;
    }

    m_parameters[tName] = Teuchos::rcp(new Parameter(tName, tTarget, tValue));
  }
}

/******************************************************************************/
void
MPMD_App::LocalOp::
updateParameters(std::string aName, Plato::Scalar aValue)
/******************************************************************************/
{
  if( m_parameters.count(aName) == 0 )
  {
    std::stringstream ss;
    ss << "Attempted to update a parameter ('" << aName << "') that wasn't defined for this operation";
    Plato::ParsingException pe(ss.str());
    throw pe;
  } 
  else
  {
    auto it = m_parameters.find(aName);
    auto pm = it->second;
    parseInline(m_def->params, pm->m_target, aValue);
    m_def->modified = true;
  }
}


/******************************************************************************/
MPMD_App::ComputeObjective::
ComputeObjective(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                 Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef){}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjective::operator()()
/******************************************************************************/
{
  mMyApp->m_state = mMyApp->m_problem->solution(mMyApp->m_control);

  mMyApp->m_objective_value      = mMyApp->m_problem->objectiveValue(mMyApp->m_control, mMyApp->m_state);
  mMyApp->m_objective_gradient_z = mMyApp->m_problem->objectiveGradient(mMyApp->m_control, mMyApp->m_state);
}

/******************************************************************************/
MPMD_App::ComputeObjectiveX::
ComputeObjectiveX(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                  Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef){}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjectiveX::operator()()
/******************************************************************************/
{
  mMyApp->m_state = mMyApp->m_problem->solution(mMyApp->m_control);

  mMyApp->m_objective_value      = mMyApp->m_problem->objectiveValue(mMyApp->m_control, mMyApp->m_state);
  mMyApp->m_objective_gradient_x = mMyApp->m_problem->objectiveGradientX(mMyApp->m_control, mMyApp->m_state);
}


/******************************************************************************/
MPMD_App::ComputeObjectiveValue::
ComputeObjectiveValue(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                      Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjectiveValue::operator()()
/******************************************************************************/
{
  mMyApp->m_state = mMyApp->m_problem->solution(mMyApp->m_control);
  mMyApp->m_objective_value = mMyApp->m_problem->objectiveValue(mMyApp->m_control,mMyApp->m_state);
}

/******************************************************************************/
MPMD_App::ComputeObjectiveGradient::
ComputeObjectiveGradient(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                         Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjectiveGradient::operator()()
/******************************************************************************/
{
  mMyApp->m_objective_gradient_z = mMyApp->m_problem->objectiveGradient(mMyApp->m_control, mMyApp->m_state);
}

/******************************************************************************/
MPMD_App::ComputeObjectiveGradientX::
ComputeObjectiveGradientX(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                          Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeObjectiveGradientX::operator()()
/******************************************************************************/
{
  mMyApp->m_objective_gradient_x = mMyApp->m_problem->objectiveGradientX(mMyApp->m_control, mMyApp->m_state);
}

/******************************************************************************/
MPMD_App::ComputeConstraint::
ComputeConstraint(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                  Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraint::operator()()
/******************************************************************************/
{
  mMyApp->m_constraint_value      = mMyApp->m_problem->constraintValue(mMyApp->m_control, mMyApp->m_state);
  mMyApp->m_constraint_gradient_z = mMyApp->m_problem->constraintGradient(mMyApp->m_control, mMyApp->m_state);

  std::cout << "Plato:: Constraint value = " << mMyApp->m_constraint_value << std::endl;
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
  mMyApp->m_constraint_value      = mMyApp->m_problem->constraintValue(mMyApp->m_control, mMyApp->m_state);
  mMyApp->m_constraint_gradient_x = mMyApp->m_problem->constraintGradientX(mMyApp->m_control, mMyApp->m_state);

  std::cout << "Plato:: Constraint value = " << mMyApp->m_constraint_value << std::endl;
}


/******************************************************************************/
MPMD_App::ComputeConstraintValue::
ComputeConstraintValue(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                       Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraintValue::operator()()
/******************************************************************************/
{
  mMyApp->m_constraint_value = mMyApp->m_problem->constraintValue(mMyApp->m_control,mMyApp->m_state);

  std::cout << "Plato:: Constraint value = " << mMyApp->m_constraint_value << std::endl;
}

/******************************************************************************/
MPMD_App::ComputeConstraintGradient::
ComputeConstraintGradient(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                          Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraintGradient::operator()()
/******************************************************************************/
{
  mMyApp->m_constraint_gradient_z = mMyApp->m_problem->constraintGradient(mMyApp->m_control, mMyApp->m_state);
}

/******************************************************************************/
MPMD_App::ComputeConstraintGradientX::
ComputeConstraintGradientX(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                          Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeConstraintGradientX::operator()()
/******************************************************************************/
{
  mMyApp->m_constraint_gradient_x = mMyApp->m_problem->constraintGradientX(mMyApp->m_control, mMyApp->m_state);
}

/******************************************************************************/
MPMD_App::ComputeSolution::
ComputeSolution(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
                Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeSolution::operator()()
/******************************************************************************/
{
  mMyApp->m_state = mMyApp->m_problem->solution(mMyApp->m_control);
}

/******************************************************************************/
MPMD_App::Reinitialize::
Reinitialize(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
             Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::Reinitialize::operator()()
/******************************************************************************/
{
  auto def = mMyApp->m_problemDefinitions[mMyApp->m_currentProblemName];
  mMyApp->createProblem(*def);
}


/******************************************************************************/
MPMD_App::WriteOutput::
WriteOutput(MPMD_App* aMyApp, Plato::InputData& aOpNode, 
            Teuchos::RCP<ProblemDefinition> aOpDef) : LocalOp(aMyApp, aOpNode, aOpDef) { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::WriteOutput::operator()() { }
/******************************************************************************/

/******************************************************************************/
void MPMD_App::finalize() { }
/******************************************************************************/


/******************************************************************************/
void MPMD_App::importData(const std::string& aName, const Plato::SharedData& aSharedField)
/******************************************************************************/
{
  importDataT(aName, aSharedField);
}


/******************************************************************************/
void MPMD_App::exportData(const std::string& aName, Plato::SharedData& aSharedField)
/******************************************************************************/
{
  exportDataT(aName, aSharedField);
}


/******************************************************************************/
void MPMD_App::exportDataMap(const Plato::data::layout_t & aDataLayout, 
                             std::vector<int> & aMyOwnedGlobalIDs)
/******************************************************************************/
{
    if(aDataLayout == Plato::data::layout_t::SCALAR_FIELD)
    {
      int tNumLocalVals = mMesh.nverts();
      aMyOwnedGlobalIDs.resize(tNumLocalVals);
      for(int tLocalID=0; tLocalID<tNumLocalVals; tLocalID++){
        aMyOwnedGlobalIDs[tLocalID] = tLocalID+1;
      }
    } else
    if(aDataLayout == Plato::data::layout_t::ELEMENT_FIELD)
    {
      int tNumLocalVals = mMesh.nelems();
      aMyOwnedGlobalIDs.resize(tNumLocalVals);
      for(int tLocalID=0; tLocalID<tNumLocalVals; tLocalID++){
        aMyOwnedGlobalIDs[tLocalID] = tLocalID+1;
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
