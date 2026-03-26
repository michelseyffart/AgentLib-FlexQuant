import setuptools

requirements = [
    "agentlib[full] @ git+https://github.com/RWTH-EBC/AgentLib.git@1ed7b4d",
    "agentlib_mpc[fmu, interactive] @ git+https://github.com/RWTH-EBC/AgentLib-MPC.git@quickfix-custom-objectives",
    "pycombina @ git+https://github.com/adbuerger/pycombina.git",
    "pathlib",
    "astor==0.8.1",
    "black",
    "pre-commit",
    # Building the docs
    "sphinx>=6.1.3",
    "m2r2",
    "myst-parser",
    "autodoc_pydantic>=1.8.0",
    "sphinx-material",
]


setuptools.setup(
    name="agentlib_flexquant",
    version="0.2.0",
    author="",
    author_email="",
    description="Flexibility quantification setup based on agentlib_mpc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
