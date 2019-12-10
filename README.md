# RIO SDK

## Overview
This library contains modules for accessing RIO services. It is intended to be used in the case where you wish to access
a RIO gRPC service running either on your local machine, or in the Cloud. The library should be `pip install`ed into
your local environment or "venv" (virtual environment).

The gRPC stubs can be found under `rio_sdk/generated` in the files `rio_pbs2.py` and `rio_pbs_grpc.py`.

For documentation on the interface, see `rio.proto` which describes the gRPC interface used to communicate with the
RIO server.

For more information on the technology behind RIO, see the [whitepaper](https://arxiv.org/abs/1906.00588).

## Steps for using RIO
Here is some pseudocode showing how you might interact with RIO. It is not intended to be a full, runnable code sample
and only shows an outline of how you might use RIO.

```python
server = connect_to_rpc_server()

train_request = create_train_request()
train_response = server.Train(train_request)

predict_request = create_predict_request()
predict_request.model = train_response.model
predict_response = server.Predict(predict_request)

my_test_labels = get_test_labels()
evaluate_performance(predict_response, my_test_labels)
```

