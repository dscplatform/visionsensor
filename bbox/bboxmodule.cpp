#include <iostream>
#include "Python.h"
#include "numpy/arrayobject.h"


static PyObject*
extract (PyObject *self, PyObject *args)
{
  PyObject *arg1 = NULL, *out = NULL;
  PyArrayObject *tensor = NULL, *outarr = NULL;

  if (!PyArg_ParseTuple(args, "O!", &arg1, &PyArray_Type, &out))
  {
    return NULL;
  }

  npy_intp *shape = PyArray_DIMS(tensor);







}
