
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "main.hpp"

namespace py = pybind11;
using namespace xtest;

PYBIND11_MODULE(xtest, m)
{

    m.doc() = "XTEST.";

    py::class_<XTest, shared_ptr<XTest>>(m, "XTest")
        .def(py::init<int, int>())
        .def_readwrite("a", &XTest::a)
        .def_readwrite("b", &XTest::b)
        .def(py::self + py::self);

}
