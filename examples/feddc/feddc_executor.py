# -*- coding: utf-8 -*-

import logging
import os
import pickle
import sys
import time

import fedscale.core.channels.job_api_pb2 as job_api_pb2
from feddc_client import FedDC_Client
from fedscale.core import fllibs
from fedscale.core import commons
from fedscale.core.execution.executor import Executor
from fedscale.core.execution.rlclient import RLClient
from fedscale.core.logger import execution
from fedscale.core.logger.execution import args
from fedscale.dataloaders.divide_data import select_dataset

"""A customized executor for FedDC"""
class FedDC_Executor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super(FedDC_Executor, self).__init__(args)

        self.temp_mask_path = os.path.join(
            execution.logDir, 'mask_'+str(args.this_rank)+'.pth.tar')
        self.mask = []
        self.epoch = 0

    def get_client_trainer(self, conf):
        return FedDC_Client(conf)

    def UpdateMask(self, configs):
        """Receive the broadcasted global mask for current round

        Args:
            config (PyTorch mask): The broadcasted global mask config
        
        """
        self.update_mask_handler(mask=configs)

    def update_mask_handler(self, mask):
        self.mask = mask
        
        # Dump latest mask to disk
        with open(self.temp_mask_path, 'wb') as mask_out:
            pickle.dump(self.mask, mask_out)

    def load_shared_mask(self):
        with open(self.temp_mask_path, 'rb') as mask_in:
            mask = pickle.load(mask_in)
        return mask

    def training_handler(self, clientId, conf, model=None):
        """Train model given client id
        
        Args:
            clientId (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result
        
        """
        # load last global model and mask
        client_model = self.load_global_model() if model is None else model
        mask_model = self.load_shared_mask()

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = fllibs.tokenizer
        if self.args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets,
                                         batch_size=conf.batch_size, args=self.args,
                                         collate_fn=self.collate_fn)

            client = self.get_client_trainer(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf, mask_model=mask_model, epochNo=self.epoch)

        return train_res

    def event_monitor(self):
        """Activate event handler once receiving new message
        """
        logging.info("Start monitoring events ...")
        self.client_register()

        while self.received_stop_request == False:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == commons.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(client_id=str(client_id), executor_id=self.executor_id,
                                                    event=commons.UPLOAD_MODEL, status=True, msg=None,
                                                    meta_result=None, data_result=self.serialize_response(train_res)
                                                    ))
                    future_call.add_done_callback(lambda _response: self.dispatch_worker_events(_response.result()))

                elif current_event == commons.MODEL_TEST:
                    self.Test(self.deserialize_response(request.meta))

                elif current_event == commons.UPDATE_MODEL:
                    broadcast_config = self.deserialize_response(request.data)
                    self.UpdateModel(broadcast_config)

                elif current_event == commons.UPDATE_MASK:
                    broadcast_config = self.deserialize_response(request.data)
                    self.UpdateMask(broadcast_config)

                elif current_event == commons.SHUT_DOWN:
                    self.Stop()

                elif current_event == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                self.client_ping()

if __name__ == "__main__":
    executor = FedDC_Executor(args)
    executor.run()