import flwr as fl
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar


class TrainLossStrategy(fl.server.strategy.FedAvg):


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)


        losses = np.array([results[i][1].metrics["training loss"] for i in range(len(results))])

        print(losses)
        '''
        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)
        '''
        return aggregated_parameters, aggregated_metrics