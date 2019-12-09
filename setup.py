"""
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
    version='1.0.0',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    packages=['rio_sdk', 'rio_sdk.generated'],
    package_data={
        'rio_sdk.proto': ['rio.proto']
    },
    install_requires=[
        "gpflow==1.3.0",
        "grpcio-tools==1.16.0",
        "grpcio==1.16.0",
        "protobuf==3.9.1",
    ],
    description='This is the SDK for accessing RIO as a service.',
    long_description=_read('README.md'),
    author='Darren Sargent',
    url='https://github.com/leaf-ai/rio-sdk/'
)
