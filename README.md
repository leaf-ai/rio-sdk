# RIO SDK
This library contains modules for accessing RIO services. It is intended to be used in the case where you wish to access
a RIO gRPC service running either on your local machine, or in the Cloud. The library should be `pip install`ed into
your local environment or "venv" (virtual environment).

The gRPC stubs can be found under `rio_sdk/generated` in the files `rio_pbs2.py` and `rio_pbs_grpc.py`.

For documentation on the interface, see `rio.proto` which describes the gRPC interface used to communicate with the
RIO server.

For more information on the technology behind RIO, see the [whitepaper](https://arxiv.org/abs/1906.00588).
