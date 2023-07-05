import flwr as fl
import argparse
from flwr_client import FlClient

parser = argparse.ArgumentParser(description="run script for flower client")
parser.add_argument("--cid", type=int, default=None,help="client id")
parser.add_argument("--K", type =int, default = 1)
parser.add_argument("--gamma", type = float, default = 1, help="local learning rate")
parser.add_argument("--seed", type = int, default = 42, help="random seed")

args = parser.parse_args()


fl.client.start_numpy_client(server_address = "[::]:8080", client = FlClient(args.cid, args.gamma, args.K, args.seed))