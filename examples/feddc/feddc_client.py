import logging
import math
import os
import pickle
import sys

import numpy as np
import torch
from torch.autograd import Variable

from fedscale.core.execution.client import Client
from fedscale.core.logger.execution import logDir
from fedscale.core.config_parser import args
from fedscale.utils.compressor.topk import TopKCompressor

"""A customized client for FedDC"""
class FedDC_Client(Client):
    """Basic client component in Federated Learning"""

    def load_compensation(self, temp_path):
        # load last global model
        with open(temp_path, 'rb') as model_in:
            model = pickle.load(model_in)
        return model
    
    def save_compensation(self, model, temp_path):
        # serialized_data = pickle.dumps(model)
        # model = pickle.loads(serialized_data)
        model = [i.to(device='cpu') for i in model]
        with open(temp_path, 'wb') as model_out:
            pickle.dump(model, model_out)

    def load_global_grad(self, temp_path):
        with open(temp_path, 'rb') as model_in:
            grad = pickle.load(model_in)
        return grad

    def train(self, client_data, model: torch.nn.Module, conf, epochNo):
        clientId = conf.clientId
        device = conf.device

        # TODO Verify that this works, I've directly imported the args from config_parser and logDir from execution
        total_mask_ratio = args.total_mask_ratio
        fl_method = args.fl_method
        regenerate_epoch = args.regenerate_epoch

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps* conf.batch_size)


        logging.info(f"Start to train (CLIENT: {clientId}) ...")

        model = model.to(device=device)
        model.train()

        # TODO ===== load compensation =====

        last_model_copy = [param.data.cpu().numpy() for param in model.stat_dict().values()]


        optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss().to(device=device)

        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0

        targets = torch.zeros(32, dtype=torch.long)
        for i in range(len(targets)):
            targets[i] = 0

        # TODO: One may hope to run fixed number of epochs, instead of iterations
        while completed_steps < conf.local_steps:
            # self.train_step(client_data, conf, model, optimizer, criterion) # TODO it is probably ok to just use this if we do not consider changes for APF
            for data_pair in client_data:
                (data, target) = data_pair
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                output = model(data)
                loss = criterion(output, target)

                # only measure the loss of the first epoch
                epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * loss.item()

                # ========= Define the backward loss ==============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                completed_steps += 1

                if completed_steps == conf.local_steps:
                    break

        state_dicts = model.state_dict()
        model_param = {p:state_dicts[p].data.cpu().numpy() for p in state_dicts} # TODO Not used

        model_gradient = []
        compressor = TopKCompressor(compress_ratio=total_mask_ratio)
        # ===== calculate gradient =====
        for idx, param in enumerate(model.state_dict().values()):
            gradient_tmp = last_model_copy[idx] - param.data

            # TODO ===== apply compensation =====

            # ===== apply compression =====
            if fl_method == 'STC' or epochNo % regenerate_epoch == 1:
                # local masking
                gradient_tmp, ctx_tmp = compressor.compress(
                        gradient_tmp)

                gradient_tmp = compressor.decompress(gradient_tmp, ctx_tmp)
            else:
                pass
                # shared masking + local mask
                # TODO uncomment for mask shifting
                # max_value = float(gradient_tmp.abs().max())
                # largest_tmp = gradient_tmp.clone().detach()
                # largest_tmp[mask_model[idx] == True] = max_value
                # largest_tmp, ctx_tmp = compressor.compress(largest_tmp)
                # largest_tmp = compressor.decompress(largest_tmp, ctx_tmp)
                # largest_tmp = largest_tmp.to(torch.bool)

                # gradient_tmp[largest_tmp != True] = 0.0

            # ===== update compensation ======
            # compensation_model[idx] = gradient_original - gradient_tmp

            model_gradient.append(gradient_tmp.cpu().numpy())

        # ===== TODO save compensation =====
        # self.save_compensation(compensation_model, temp_path)
        


        # ===== collect results =====
        results = {'clientId':clientId, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(epoch_train_loss)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['update_gradeint'] = model_gradient
        results['wall_duration'] = 0

        return results

