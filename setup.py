"""
Copyright (C) 2019 Cognizant Digital Business, Cognizant Artificial Intelligence Practice.
All rights reserved.

Installer for the RIO SDK
"""
import os
import sys

from setuptools import setup

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of Rio requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)


def _read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as file:
        return file.read()


setup(
    name='rio-sdk',
    version='2.0.3',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    packages=['rio_sdk', 'rio_sdk.generated', 'rio_sdk.protos'],
    package_data={
        'rio_sdk.protos': ['rio.proto']
    },
    install_requires=[
        "gpflow==1.5.1",
        "grpcio-tools==1.29.0",
        "grpcio==1.29.0",
        "protobuf==3.12.4",
    ],
    description='This is the SDK for accessing RIO as a service.',
    long_description=_read('README.md'),
    author='Darren Sargent',
    url='https://github.com/leaf-ai/rio-sdk/'
)
