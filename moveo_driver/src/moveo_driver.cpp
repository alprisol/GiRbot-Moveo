#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "moveo.h"

namespace py = pybind11;

PYBIND11_MODULE(moveo_driver, m) {
    py::class_<Moveo>(m, "Moveo")
        .def(py::init<>())
        .def("connect", &Moveo::connect)
        .def("disconnect", &Moveo::disconnect)
        .def("setMotorParams", &Moveo::setMotorParams)
        .def("setMotorRanges", &Moveo::setMotorRanges)
        .def("enable", &Moveo::enable)
        .def("disable", &Moveo::disable)
        .def("stop", &Moveo::stop)
        .def("openGripper", &Moveo::openGripper)
        .def("closeGripper", &Moveo::closeGripper)
        .def("setTargetPosition", &Moveo::setTargetPosition)
        .def("getCurrentPosition", &Moveo::getCurrentPosition)
        .def("moveToPosition", &Moveo::moveToPosition)
        .def("trajToPosition", &Moveo::trajToPosition);
}