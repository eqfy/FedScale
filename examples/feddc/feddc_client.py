from cmath import log
import logging
import math
import os
import pickle
import sys

import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable

from fedscale.core.execution.client import Client
from fedscale.core.logger.execution import logDir
from fedscale.core.config_parser import args
from fedscale.utils.compressor.topk import TopKCompressor

def set_bn_eval(m):
    # print(m.__dict__)
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('BatchNorm2d') != -1 or classname.find('bn') != -1:
        m.eval()

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

    # def load_running(self, temp_path):
    #     with open(temp_path, 'rb') as model_in:
    #         grad = pickle.load(model_in)
    #     return grad

    def train(self, client_data, model: torch.nn.Module, conf, mask_model, epochNo, agg_weight):
        clientId = conf.clientId
        device = conf.device

        np.random.seed(1)
        total_mask_ratio = args.total_mask_ratio
        fl_method = args.fl_method
        regenerate_epoch = args.regenerate_epoch
        compensation_dir = os.path.join(args.compensation_dir, args.job_name, args.time_stamp)

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps* conf.batch_size)


        logging.info(f"Start to train (CLIENT: {clientId}) (WEIGHT: {agg_weight}) (LR: {conf.learning_rate})...")
        # logging.info(f"{total_mask_ratio} {fl_method} {regenerate_epoch}")

        model = model.to(device=device)
        model.train()
        # for module in model.modules():
        #     # print(module)
        #     if isinstance(module, nn.BatchNorm2d):
        #         if hasattr(module, 'weight'):
        #             module.weight.requires_grad_(False)
        #         if hasattr(module, 'bias'):
        #             module.bias.requires_grad_(False)
        #         module.eval()
        # model.apply(set_bn_eval)

        # ===== load compensation =====
        os.makedirs(compensation_dir, exist_ok=True)
        temp_path = os.path.join(compensation_dir, 'compensation_c'+str(clientId)+'.pth.tar')
        compensation_model = []
        # temp_path_2 = os.path.join(logDir, 'gradient_c'+str(clientId)+'.pth.tar')
        if (agg_weight > 100.0) or (not os.path.exists(temp_path)):
            # create a new compensation model
            for idx, param in enumerate(model.state_dict().values()):
                tmp = torch.zeros_like(param.data).to(device=device)
                compensation_model.append(tmp)
                
                # tmp_2 = torch.zeros_like(param.data).to(device)
                # gradient_model.append(tmp_2)
        else:
            compensation_model = self.load_compensation(temp_path)
            compensation_model = [c.to(device=device) for c in compensation_model]

        # ===== load running mean ====
        keys = [] 
        for idx, key in enumerate(model.state_dict()):
            keys.append(key)
        
        # temp_path_running = os.path.join(logDir, 'running_c'+str(clientId)+'.pth.tar')
        # if os.path.exists(temp_path):
        #     last_model = self.load_compensation(temp_path_running)
        #     for idx, param in enumerate(model.state_dict().values()):
        #         if (('num_batches_tracked' in keys[idx]) or ('running' in keys[idx])):
        #             param.data = last_model[idx].clone().detach()

        # model = model.to(device=device)

        last_model_copy = [param.data.clone() for param in model.state_dict().values()]

        optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9)
        if args.model == "lr":
            criterion = torch.nn.CrossEntropyLoss().to(device=device)
        else:
            criterion = torch.nn.CrossEntropyLoss().to(device=device)


        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0

        # targets = torch.zeros(32, dtype=torch.long)
        # for i in range(len(targets)):
        #     targets[i] = 0
        
        # agg_weight = 1.0
        # test
        # TODO: One may hope to run fixed number of epochs, instead of iterations
        while completed_steps < conf.local_steps:
            for data_pair in client_data:
                (data, target) = data_pair
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                if args.task == "speech":
                    data = torch.unsqueeze(data, 1)

                output = model(data)
                loss = criterion(output, target)

                # only measure the loss of the first epoch
                epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * loss.item()
                # logging.info(f"local {clientId} {completed_steps} {loss.item()}")
                # ========= Define the backward loss ==============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                completed_steps += 1

                if completed_steps == conf.local_steps:
                    break

        # state_dicts = model.state_dict()
        # model_param = {p:state_dicts[p].data.cpu().numpy() for p in state_dicts} # TODO Not used

        model_gradient = []
        compressor = TopKCompressor(compress_ratio=total_mask_ratio)
        # ===== calculate gradient =====
        for idx, param in enumerate(model.state_dict().values()):
            # logging.info(f"last model copy device {last_model_copy[idx].get_device()} param data device {param.data.get_device()}")
            gradient_tmp = (last_model_copy[idx] - param.data).type(torch.FloatTensor).to(device=device)
            # logging.info(f"gradient_tmp {gradient_tmp.get_device()}")


            # gradient_tmp = (last_model_copy[idx] - param.data).to('cpu').type(torch.FloatTensor)
            # if agg_weight > 100:
            #     print(keys[idx], gradient_tmp)

            # TODO ===== apply compensation =====
            if not (('num_batches_tracked' in keys[idx]) or ('running' in keys[idx])):
            # if not ('num_batches_tracked' in keys[idx]):
                # compensation_model[idx] = compensation_model[idx].to('cpu')
                # logging.info(f"1 compensation device {compensation_model[idx].get_device()} gradient_tmp {gradient_tmp.get_device()}")
                gradient_tmp += (compensation_model[idx] / agg_weight)
                # logging.info(f"2 compensation device {compensation_model[idx].get_device()} gradient_tmp {gradient_tmp.get_device()}")
            
                gradient_original = gradient_tmp.clone().detach()

                # ===== apply compression =====
                if fl_method == 'STC' or epochNo % regenerate_epoch == 1:
                    # local masking
                    gradient_tmp, ctx_tmp = compressor.compress(
                            gradient_tmp)

                    gradient_tmp = compressor.decompress(gradient_tmp, ctx_tmp)
                else:
                    # pass
                    # shared masking + local mask
                    max_value = float(gradient_tmp.abs().max())
                    largest_tmp = gradient_tmp.clone().detach()
                    largest_tmp[mask_model[idx] == True] = max_value
                    largest_tmp, ctx_tmp = compressor.compress(largest_tmp)
                    largest_tmp = compressor.decompress(largest_tmp, ctx_tmp)
                    largest_tmp = largest_tmp.to(torch.bool)
                    gradient_tmp[largest_tmp != True] = 0.0

            # ===== update compensation ======
            # gradient_original = gradient_original.to('cpu')
            # compensation_model[idx] = compensation_model[idx].to('cpu')
            # gradient_tmp = gradient_tmp.to('cpu')
            compensation_model[idx] = (gradient_original.type(torch.FloatTensor) - gradient_tmp.type(torch.FloatTensor)).type(torch.FloatTensor) * agg_weight
            # if agg_weight > 100.0:
            #     print(compensation_model[idx].type())
            #     compensation_model[idx] *= 89.32
            model_gradient.append(gradient_tmp)
            # model_gradient.append(gradient_tmp.cpu().numpy())


        # ===== TODO save compensation =====
        compensation_model = [e.to(device='cpu') for e in compensation_model]
        self.save_compensation(compensation_model, temp_path)
        # model = model.to(device="cpu")

        model_gradient = [g.to(device='cpu').numpy() for g in model_gradient]

        # ===== TODO save running mean =====   FIXME Doesn't seem to be used?    
        # save_model = []
        # for idx, param in enumerate(model.state_dict().values()):
        #     tmp = param.data.clone().detach().to(device="cpu")
        #     save_model.append(tmp)
        # # self.save_compensation(save_model, temp_path_running)

        # ===== collect results =====
        results = {'clientId':clientId, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(epoch_train_loss)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        # results['update_weight'] = model_param
        results['update_gradient'] = model_gradient
        results['wall_duration'] = 0

        return results
