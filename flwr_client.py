import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))


from collections import OrderedDict
from model import cnn_model
import flwr as fl



class FlClient(fl.client.Client):
    def __init__(self, client_id, gamma, K, seed):
        super (FlClient, self).__init__()
        torch.manual_seed(seed)

        self.model = cnn_model().double()


        '''
        if init_norm:
            params = init_params(dataset, model_choice, zeros=False)
            self.net.set_params(params)
        '''
        X_train = np.load("data/train_data_client_" + str(client_id) + ".npy" )
        y_train = np.load("data/train_labels_client_" + str(client_id) + ".npy" )
        
        self.X_train = torch.as_tensor(X_train, dtype = torch.double).reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        self.y_train = torch.as_tensor(y_train, dtype = torch.int64)
        
        
        #self.X_test = np.load("data/train_data_client_" + str(client_id) + ".npy" )
        #self.y_test = np.load("data/train_data_client_" + str(client_id) + ".npy" )

        self.gamma = gamma
        self.K = K
        
    def get_weights(self) -> fl.common.NDArrays:
            """Get model weights as a list of NumPy ndarrays."""
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays.

        Parameters
        ----------
        weights: fl.common.NDArrays
            Weights received by the server and set to local model


        Returns
        -------

        """
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config) -> fl.common.GetParametersRes:
        """Encapsulates the weight into Flower Parameters """
        weights: fl.common.NDArrays = self.get_weights()
        parameters = fl.common.ndarrays_to_parameters(weights)
        return fl.common.GetParametersRes(status = fl.common.Status(code = fl.common.Code(0), message = ""), parameters=parameters)
    
    def fit(self, ins):

        weights: fl.common.NDArrays = fl.common.parameters_to_ndarrays(ins.parameters)

        self.set_weights(weights)

        opt = torch.optim.SGD(self.model.parameters(), lr=self.gamma)
        crit = nn.CrossEntropyLoss()

        loss = self.model.train(self.X_train, self.y_train, opt, crit, self.K)
        
        return fl.common.FitRes(
            status = fl.common.Status(code = fl.common.Code(0), message = ""),
            parameters = fl.common.ndarrays_to_parameters(self.get_weights()),
            num_examples = self.X_train.shape[0],
            metrics = {"training loss" : loss.detach().numpy()}
        )
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        results =  self.model.test()
        #loss = 1-accuracy
        #loss = results["accuracy"]
        loss = results
        return float(loss), self.X_test.shape[0], {"accuracy": float(results)}