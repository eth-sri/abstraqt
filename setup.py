import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(exclude=["tests.*", "tests"])
tests_require = [
    'pytest-xdist>=2.2.1',
    'pytest>=6.2.3',
    'matplotlib>=3.4',
    'tqdm>=4.60',
    'galois==0.0.14',
    'sh>=1.14',
    'GitPython>=3.1.24',
    # 'pyzx==0.7.3'
]
setuptools.setup(
    name="abstraqt",
    version="0.0.1",
    description="Detect invariants in quantum circuits using abstract interpretation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=packages,
    python_requires='>=3.8',
    install_requires=[
        'numpy==1.20',
        'scipy==1.7',
        'numba==0.54',
        'appdirs>=1.4',
        'click>=8.0',  # appdirs raises a warning without this
        'pybind11>=2.6.2',
        'cppimport>=21.3',
        'qiskit==0.40.0',
        'cachier>=1.5.0',
        'pandas>=1.2',
        'argparse>=1.4',
    ],
    tests_require=tests_require,
    extras_require={
        'test': tests_require
    }
)
