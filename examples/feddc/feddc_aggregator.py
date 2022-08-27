import os
import sys

from fedscale.core.aggregation.aggregator import Aggregator
from fedscale.core.logger.aggragation import *


class FedDC_Aggregator(Aggregator):
    """Feed aggregator using tensorflow models"""
    def __init__(self, args):
        super().__init__(args)

if __name__ == "__main__":
    aggregator = FedDC_Aggregator(args)
    aggregator.run()