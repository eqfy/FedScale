# -*- coding: utf-8 -*-

import os
import sys

from feddc_client import FedDC_Client

from fedscale.core.execution.executor import Executor
from fedscale.core.logger.execution import args

"""In this example, we only need to change the Client Component we need to import"""

class FedDC_Executor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super(FedDC_Executor, self).__init__(args)

    def get_client_trainer(self, conf):
        return FedDC_Client(conf)

if __name__ == "__main__":
    executor = FedDC_Executor(args)
    executor.run()