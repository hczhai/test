
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "main.hpp"
#include "matrix_functions.hpp"
#include <iostream>

namespace py = pybind11;
using namespace xtest;
using namespace block2;

PYBIND11_MODULE(xtest, m)
{

    m.doc() = "XTEST.";

    py::class_<XTest, shared_ptr<XTest>>(m, "XTest")
        .def(py::init<int, int>())
        .def_readwrite("a", &XTest::a)
        .def_readwrite("b", &XTest::b)
        .def("test", [](XTest *self) {
            vector<double> x(25);
            block2::Random::rand_seed(1234);
            block2::Random::fill_rand_double(x.data(), x.size());
            cout << x[0] << endl;
            MatrixFunctions::inverse(MatrixRef(x.data(), 5, 5));
            cout << x[0] << endl;
        })
        .def(py::self + py::self);

    py::class_<ParallelCommunicator<void>, shared_ptr<ParallelCommunicator<void>>>(
        m, "ParallelCommunicator")
        .def(py::init<>())
        .def(py::init<int, int, int>())
        .def_readwrite("size", &ParallelCommunicator<void>::size)
        .def_readwrite("rank", &ParallelCommunicator<void>::rank)
        .def_readwrite("root", &ParallelCommunicator<void>::root);

#ifdef _HAS_MPI
    py::class_<MPICommunicator<void>, shared_ptr<MPICommunicator<void>>,
               ParallelCommunicator<void>>(m, "MPICommunicator")
        .def(py::init<>())
        .def(py::init<int>());
#endif
}
