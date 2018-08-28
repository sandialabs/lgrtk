#include <Python.h>
#include <structmember.h>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include "plato/lgr_App.hpp"

std::vector<double> double_vector_from_list(PyObject* list);
PyObject* list_from_double_vector(std::vector<double> inVector);

namespace PlatoPython
{

class SharedData {
  public:
    SharedData(Plato::data::layout_t layout, int size, double initVal=0.0 ) :
      m_data(size,initVal), m_layout(layout){}

    void setData(const std::vector<double> & aData)
    {
      m_data = aData;
    }
    void getData(std::vector<double> & aData) const
    {
      aData = m_data;
    }
    int size() const
    {
      return m_data.size();
    }

    std::string myContext() const {return m_context;}
    void setContext(std::string context) {m_context = context;}

    Plato::data::layout_t myLayout() const
    {
      return m_layout;
    }

    double operator[](int index){ return m_data[index]; }

  protected:
    std::vector<double> m_data;
    Plato::data::layout_t m_layout;
    std::string m_context;
};
class NodeField : public SharedData
{
  public:
    NodeField(int size, double initVal=0.0) :
       SharedData(Plato::data::layout_t::SCALAR_FIELD, size, initVal){}
};

class ElementField : public SharedData
{
  public:
    ElementField(int size, double initVal=0.0) :
       SharedData(Plato::data::layout_t::ELEMENT_FIELD, size, initVal){}
};

class ScalarParameter : public SharedData
{
  public:
    ScalarParameter(std::string context, double initVal=0.0) :
       SharedData(Plato::data::layout_t::SCALAR_PARAMETER, 1, initVal)
         {m_context = context;}
};

class SharedValue : public SharedData
{
  public:
    SharedValue(int size, double initVal=0.0) :
       SharedData(Plato::data::layout_t::SCALAR, size, initVal){}
};

} // end namespace 


struct Analyze {
    PyObject_HEAD
    std::string m_inputfileName;
    std::string m_appfileName;
    std::string m_instanceName;
    std::shared_ptr<MPMD_App> m_MPMD_App;
    std::vector<int> m_localNodeIDs;
    std::vector<int> m_localElemIDs;
    static int m_numInstances;
};

int Analyze::m_numInstances=0;

static PyObject *
Analyze_initialize(Analyze* self)
{
    self->m_MPMD_App->initialize();

    // get the local node IDs.  These aren't used for distributed computing.
    //
    self->m_MPMD_App->exportDataMap(Plato::data::layout_t::SCALAR_FIELD, self->m_localNodeIDs);
    self->m_MPMD_App->exportDataMap(Plato::data::layout_t::ELEMENT_FIELD, self->m_localElemIDs);

    return Py_BuildValue("i", 1);
}

static PyObject *
Analyze_importData(Analyze *self, PyObject *args, PyObject *kwds)
{
    // parse incoming arguments 
    //
    char *inputDataName;
    char *inputDataType;
    PyObject *inputData;

    if (! PyArg_ParseTuple(args, "ssO", &inputDataName, &inputDataType, &inputData) )
    {
        return Py_BuildValue("i", -1);
    }
    std::cout << "Importing " << inputDataName << std::endl;

    std::string inName(inputDataName);
    std::string inType(inputDataType);

    if( inType == "SCALAR_FIELD" )
    {
        PlatoPython::NodeField inData(self->m_localNodeIDs.size());
        auto vecData = double_vector_from_list(inputData);
        inData.setData(vecData);
        self->m_MPMD_App->importDataT(inName, inData);
    } else
    if( inType == "ELEMENT_FIELD" )
    {
        PlatoPython::ElementField inData(self->m_localElemIDs.size());
        auto vecData = double_vector_from_list(inputData);
        inData.setData(vecData);
        self->m_MPMD_App->importDataT(inName, inData);
    } else
    if( inType == "SCALAR_PARAMETER" )
    {
        std::vector<std::string> tokens = split(inName,':');
        auto context = tokens[0];
        auto parameter = tokens[1];
        PlatoPython::ScalarParameter inData(context);
        std::vector<double> vecData(1, PyFloat_AsDouble(inputData));
        inData.setData(vecData);
        self->m_MPMD_App->importDataT(parameter, inData);
    } else
    if( inType == "SCALAR" )
    {
        PlatoPython::SharedValue inData(1);
        std::vector<double> vecData(1, PyFloat_AsDouble(inputData));
        inData.setData(vecData);
        self->m_MPMD_App->importDataT(inName, inData);
    }

    return Py_BuildValue("i", 1);
}

static PyObject *
Analyze_compute(Analyze *self, PyObject *args, PyObject *kwds)
{
    // parse incoming arguments 
    //
    char *operationName;

    if (! PyArg_ParseTuple(args, "s", &operationName) )
    {
        return Py_BuildValue("i", -1);
    }
    std::cout << "Computing " << operationName << std::endl;

    std::string opName(operationName);
    self->m_MPMD_App->compute(opName);

    return Py_BuildValue("i", 1);
}

static PyObject *
Analyze_exportData(Analyze *self, PyObject *args, PyObject *kwds)
{
    // parse incoming arguments 
    //
    char *outputDataName;
    char *outputDataType;

    if (! PyArg_ParseTuple(args, "ss", &outputDataName, &outputDataType) )
    {
        return Py_BuildValue("i", -1);
    }
    std::cout << "Exporting " << outputDataName << std::endl;

    std::string outName(outputDataName);
    std::string outType(outputDataType);

    if( outType == "SCALAR_FIELD" )
    {
        PlatoPython::NodeField outData(self->m_localNodeIDs.size());
        self->m_MPMD_App->exportDataT(outName, outData);
        std::vector<double> vecData(self->m_localNodeIDs.size());
        outData.getData(vecData);
        return list_from_double_vector(vecData);
    } else
    if( outType == "ELEMENT_FIELD" )
    {
        PlatoPython::ElementField outData(self->m_localElemIDs.size());
        self->m_MPMD_App->exportDataT(outName, outData);
        std::vector<double> vecData(self->m_localElemIDs.size());
        outData.getData(vecData);
        return list_from_double_vector(vecData);
    } else
    if( outType == "SCALAR" )
    {
        PlatoPython::SharedValue outData(1);
        self->m_MPMD_App->exportDataT(outName, outData);
        std::vector<double> vecData(1);
        outData.getData(vecData);
        return PyFloat_FromDouble(vecData[0]);
    }
    return Py_BuildValue("i", 1);
}

static PyObject *
Analyze_finalize(Analyze* self)
{
    self->m_MPMD_App->finalize();
    return Py_BuildValue("i", 1);
}

static PyMethodDef Plato_methods[] = {
    {NULL}  /* Sentinel */
};

static void
Analyze_dealloc(Analyze* self)
{
    self->m_numInstances--;
    if(self->m_numInstances == 0)
    {
        Kokkos::finalize();
        int isFinalized;
        MPI_Finalized(&isFinalized);
        if( !isFinalized ) MPI_Finalize();
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
Analyze_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Analyze *self;

    self = (Analyze *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}


static int
Analyze_init(Analyze *self, PyObject *args, PyObject *kwds)
{

    // parse incoming arguments 
    //
    char *inputfileName;
    char *appfileName;
    char *instanceName;

    if (! PyArg_ParseTuple(args, "sss", &inputfileName, &appfileName, &instanceName))
    {
        return -1;
    }

    self->m_inputfileName = std::string(inputfileName);
    self->m_appfileName   = std::string(appfileName);
    self->m_instanceName  = std::string(instanceName);


    // construct artificial argc and argv for initializing mpi, kokkos, and the MPMD_App
    //
    int argc = 2;
    char** argv = (char**)malloc((argc+1)*sizeof(char*));
    char exeName[] = "exeName";
    char* arg0 = strdup(exeName);
    argv[0] = arg0;
    std::stringstream inArgs;
    inArgs << "--input-config=" << self->m_inputfileName;
    char* arg1 = strdup(inArgs.str().c_str());
    argv[1] = arg1;
    argv[argc] = NULL;

    int mpiIsInitialized;
    MPI_Initialized( &mpiIsInitialized );
    if( !mpiIsInitialized )
    {
        MPI_Init(&argc, &argv);
        Kokkos::initialize(argc, argv);
    }

    // construct the MPMD_App
    //
    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    setenv("PLATO_APP_FILE", self->m_appfileName.c_str(), true);
    self->m_MPMD_App = std::make_shared<MPMD_App>(argc, argv, myComm);

    free(arg0); free(arg1); free(argv);

    

    // increment the instance counter.  This is used to finalize mpi and kokkos
    // when the last instance is deleted.  This will conflict with other modules 
    // if they're using mpi and/or kokkos.
    //
    self->m_numInstances++;

    return 0;
}

static PyMemberDef Analyze_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject *
Analyze_name(Analyze* self)
{
    PyObject *result = Py_BuildValue("s", self->m_instanceName.c_str());

    return result;
}

static PyMethodDef Analyze_methods[] = {
    {"name",       (PyCFunction)Analyze_name,       METH_NOARGS,   "Return the instance name" },
    {"initialize", (PyCFunction)Analyze_initialize, METH_NOARGS,   "Plato::Application::initialize()" },
    {"importData", (PyCFunction)Analyze_importData, METH_VARARGS,  "Plato::Application::importData()" },
    {"compute",    (PyCFunction)Analyze_compute,    METH_VARARGS,  "Plato::Application::compute()" },
    {"exportData", (PyCFunction)Analyze_exportData, METH_VARARGS,  "Plato::Application::exportData()" },
    {"finalize",   (PyCFunction)Analyze_finalize,   METH_NOARGS,   "Plato::Application::finalize()" },
    {NULL}  /* Sentinel */
};

static PyTypeObject AnalyzeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Plato.Analyze",           /* tp_name */
    sizeof(Analyze),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)Analyze_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "Analyze objects",         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    Analyze_methods,           /* tp_methods */
    Analyze_members,           /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Analyze_init,    /* tp_init */
    0,                         /* tp_alloc */
    Analyze_new,               /* tp_new */
};


#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initPlato(void) 
{
    PyObject* m;

    AnalyzeType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AnalyzeType) < 0)
        return;

    m = Py_InitModule3("Plato", Plato_methods,
                       "Example module that creates an extension type.");

    Py_INCREF(&AnalyzeType);
    PyModule_AddObject(m, "Analyze", (PyObject *)&AnalyzeType);
}

/*****************************************************************************/
// create a double vector from a Python list
/*****************************************************************************/
std::vector<double> double_vector_from_list(PyObject* inList)
{
  int length = PyList_Size(inList);
  std::vector<double> outVector(length);
  for(int i = 0; i < length; i++) {
    PyObject *v = PyList_GetItem(inList,i);
    if(!PyFloat_Check(v)) {
      PyErr_SetString(PyExc_TypeError, "list must contain only reals");
      outVector[i] = 0.0;
    }
    outVector[i] = PyFloat_AsDouble(v);
  }
  return outVector;
}

/*****************************************************************************/
// create a python list from double vector
/*****************************************************************************/
PyObject* list_from_double_vector(std::vector<double> inVector)
{
  int array_length = inVector.size();
  PyObject *newlist = PyList_New(array_length);

  for(int i=0; i<array_length; i++)
    PyList_SetItem(newlist, i, PyFloat_FromDouble(inVector[i]));

  return newlist;
}


