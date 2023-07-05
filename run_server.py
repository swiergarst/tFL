import flwr as fl
import argparse
from flwr_client import FlClient

from strategies import TrainLossStrategy
import logging

nr = 10

strategy = TrainLossStrategy(
    fraction_fit = 1,
    fraction_evaluate = 0
)

parser = argparse.ArgumentParser(description="run script for flower client")
parser.add_argument("--nc", type = int, default = 2, help = "number of clients to use")
parser.add_argument("--gamma", type = float, default = 1, help = "local learning rate")
parser.add_argument("--K", type = int, default = 1)

args = parser.parse_args()


def client_fn(cid) -> FlClient:
    FLOWER_LOGGER = logging.getLogger("flwr")
    FLOWER_LOGGER.setLevel(logging.ERROR)

    gamma = args.gamma
    K = args.K
    seed = 42
    return(FlClient(cid, gamma, K, seed))

'''
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}
'''

#logging.setLevel(logging.ERROR)
FLOWER_LOGGER = logging.getLogger("flwr")
FLOWER_LOGGER.setLevel(logging.ERROR)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients =  args.nc,
    config = fl.server.ServerConfig(num_rounds = 3),
    strategy = strategy
)

#fl.server.start_server(config = fl.server.ServerConfig(num_rounds = nr))