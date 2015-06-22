from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='RiskPredictor',
    version='0.0.1',
    description='A library to predict risks',
    long_description=long_description,
    url='https://github.com/TheHonestGene/riskpredictor',
    author=['Bjarni Vilhjalmsson','Uemit Seren'],
    author_email='bjarni.vilhjalmsson@gmai.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='Risk Prediction',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'riskpredictor=riskpredictor:main'
        ],
    },
)

