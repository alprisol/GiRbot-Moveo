from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "moveo_driver",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=["include", pybind11.get_include()],
    )
]

setup(
    name="moveo_driver",
    description='Driver for controlling a Moveo robot arm with serial communication and COBS protocol',
    author="Patryk Cieslak & Albert Prieto",
    author_email='patryk.cieslak@udg.edu , albert.prieto@udg.edu',
    version='1.0',
    setup_requires=['pybind11'],  # Ensure pybind11 is included
    install_requires=['pybind11'],  # In case pybind11 is required at runtime
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)