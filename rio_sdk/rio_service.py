"""
Copyright (C) 2019 Cognizant Digital Business, Cognizant Artifical Intelligence Practice.
All rights reserved.
"""
import grpc
import numpy as np

from rio_sdk.generated.rio_pb2 import TrainRequest, PredictRequest, FrameworkVariant, KernelType
from rio_sdk.generated.rio_pb2_grpc import RioServiceStub


class RioService:
    """
    Class to interact with the RIO service
    """

    def __init__(self, rio_host, rio_port):
        """
        Instantiates a class that can interact with the RIO service
        """
        target = f"{rio_host}:{rio_port}"
        print(f"Opening gRPC connection to {target}")
        channel = grpc.insecure_channel(target)
        self.rio_service = RioServiceStub(channel)

    def train(self,
              train_predictions_df,
              train_x_df,
              train_y_df,
              framework_variant=FrameworkVariant.GP_CORRECTED,
              kernel_type=KernelType.RBF_PLUS_RBF,
              num_svgp_inducing_points=50,
              max_iterations_optimizer=1000,
              use_ard=True):
        """
        Trains a RIO model to estimate the uncertainty of a trained model and reduce its prediction errors.
        Based on Stochastic Variational Gaussian Processes (SVGP).
        :param train_predictions_df: a Pandas DataFrame containing the predictions made by the trained model
        on the training data
        :param train_x_df: a Pandas DataFrame containing the the features used to train the model
        :param train_y_df: a Pandas DataFrame containing the labels used to train the model
        :param framework_variant: Gaussian process type to use to train the SVGP model
        :param kernel_type: kernel type to use to train the SVGP model
        :param num_svgp_inducing_points: number of inducing points for the SVGP model
        :param max_iterations_optimizer: number of maximum iterations for optimizer
        :param use_ard: boolean to turn on/off Automatic Relevance Determination (ARD)
        :return: the bytes of a RIO model that can be used to enhance predictions
        """
        train_predictions_csv = train_predictions_df.to_csv(header=False, index=False, line_terminator=",")
        train_x_csv = train_x_df.to_csv(index=False)
        train_y_csv = train_y_df.to_csv(header=False, index=False)
        train_request = TrainRequest(
            framework_variant=framework_variant,
            kernel_type=kernel_type,
            normed_train_data=train_x_csv,
            train_labels=train_y_csv,
            train_predictions=train_predictions_csv,
            num_svgp_inducing_points=num_svgp_inducing_points,
            max_iterations_optimizer=max_iterations_optimizer,
            use_ard=use_ard)
        train_response = self.rio_service.Train(train_request)
        return train_response.model

    def predict(self, x_df, predictions_df, rio_model):
        """
        Enhances the model's predictions on the passed features using the corresponding trained RIO model.
        :param x_df: a Pandas DataFrame containing the features for which we want RIO's enhanced predictions
        :param predictions_df: a Pandas DataFrame containing the initial predictions on the passed x_df features
        :param rio_model: a previously trained RIO model to use to reduce the predictions error
        :return: means and variances of the RIO corrected predictions, as numpy arrays
        """
        predictions_csv = predictions_df.to_csv(header=False, index=False, line_terminator=",")
        predict_request = PredictRequest(
            model=rio_model,
            normed_test_data=x_df.to_csv(index=False),
            test_predictions=predictions_csv)
        predict_response = self.rio_service.Predict(predict_request)
        means = np.fromstring(predict_response.mean, dtype=float, sep=',')
        variances = np.fromstring(predict_response.var, dtype=float, sep=',')
        return means, variances

    @staticmethod
    def save(rio_model, filepath):
        """
        Saves the passed rio model to a file, as bytes
        :param rio_model: the rio model to save to a file
        :param filepath: the name of the file to save to, e.g. 'models/rio_model.bytes'
        :return: nothing
        """
        with open(filepath, 'wb') as rio_model_file:
            rio_model_file.write(rio_model)

    @staticmethod
    def load(filepath):
        """
        Loads a RIO model from the passed file name
        :param filepath: the name of the file that contains a RIO model
        :return: the bytes of a RIO model
        """
        with open(filepath, 'rb') as rio_model_file:
            rio_model = rio_model_file.read()
        return rio_model
