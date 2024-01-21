from setuptools import setup, find_packages

VERSION = '1.0.2'
DESCRIPTION = 'Gradient Boosting Decision Trees for Random Utility Models'
LONG_DESCRIPTION = 'A package that allows to estimate some random utility models through gradient boosting decision trees'

# Setting up
setup(
    name="rumboost",
    version=VERSION,
    author="Nicolas Salvadé, Tim Hillel",
    author_email="<nicolas.salvade.22@ucl.ac.uk>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    url="https://github.com/NicoSlvd/rumboost",
    install_requires=[
        "biogeme==3.2.11",
        "hyperopt==0.2.7",
        "lightgbm==3.3.5",
        "matplotlib==3.7.1",
        "numpy==1.24.3",
        "pandas==2.0.2",
        "scikit-learn==1.2.2",
        "scipy==1.10.1",
        "seaborn==0.12.2"
    ],
    keywords=['python', 'gbdt', 'rum', 'ml'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)