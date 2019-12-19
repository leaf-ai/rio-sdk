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
              original_model,
              encoded_train_x_df,
              train_y_df,
              framework_variant=FrameworkVariant.GP_CORRECTED,
              kernel_type=KernelType.RBF_PLUS_RBF,
              num_svgp_inducing_points=50,
              max_iterations_optimizer=1000):
        """
        Trains a RIO model to estimate the uncertainty of the passed model and reduce the prediction errors.
        Based on Stochastic Variational Gaussian Processes (SVGP).
        :param  original_model: the model to evaluate with RIO.
        :param encoded_train_x_df: the features used to train the original model
        :param train_y_df: the labels used to train the original model
        :param framework_variant: Gaussian process type to use to train the SVGP model
        :param kernel_type: kernel type to use to train the SVGP model
        :param num_svgp_inducing_points: number of inducing points for the SVGP model
        :param max_iterations_optimizer: number of maximum iterations for optimizer
        :return: the bytes of a RIO model that can be used to enhance predictions
        """
        outcome_train_predictions = original_model.predict(encoded_train_x_df)
        outcome_train_predictions_csv = ','.join((map(str, list(outcome_train_predictions))))
        outcome_train_request = TrainRequest(
            framework_variant=framework_variant,
            kernel_type=kernel_type,
            normed_train_data=encoded_train_x_df.to_csv(index=False),
            train_labels=train_y_df.to_csv(header=False, index=False),
            train_predictions=outcome_train_predictions_csv,
            num_svgp_inducing_points=num_svgp_inducing_points,
            max_iterations_optimizer=max_iterations_optimizer)
        train_response = self.rio_service.Train(outcome_train_request)
        return train_response.model

    def predict(self, original_model, rio_model, encoded_x_df):
        """
        Enhances the original model's predictions using the corresponding trained RIO model.
        :param original_model: the model to use to make predictions
        :param rio_model: the trained RIO model to use to reduce the predictions error
        :param encoded_x_df: the DataFrame containing the features for which to make predictions
        :return: means and variances of the RIO corrected predictions, as numpy arrays
        """
        outcome_test_predictions = original_model.predict(encoded_x_df)
        outcome_test_predictions_csv = ','.join((map(str, list(outcome_test_predictions))))
        outcome_predict_request = PredictRequest(
            model=rio_model,
            normed_test_data=encoded_x_df.to_csv(index=False),
            test_predictions=outcome_test_predictions_csv)
        predict_response = self.rio_service.Predict(outcome_predict_request)
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
