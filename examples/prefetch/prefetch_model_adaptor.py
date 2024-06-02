
from typing import List

import numpy as np
import torch
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter


"""
A model adaptor for prefetch
"""
class PrefetchModelAdaptor(TorchModelAdapter):

    def __init__(self, model: torch.nn.Module, optimizer: TorchServerOptimizer = None):
        super().__init__(model, optimizer)

    def get_keys(self):
        return self.model.state_dict().keys()
    
    def get_state_dict(self):
        return self.model.state_dict()
    
    
    def get_empty_state_dict_values(self):
        return [
                torch.zeros_like(param.data).to(device=self.device, dtype=torch.float32)
                for param in self.model_wrapper.get_model().state_dict().values()
            ]
    
    def set_weights(self, weights: List[np.ndarray], is_aggregator=True, client_training_results=None):
        new_state_dict = {
            name: torch.from_numpy(np.asarray(weights[i], dtype=np.float32))
            for i, name in enumerate(self.model.state_dict().keys())
        }
        self.model.load_state_dict(new_state_dict)
        return super().set_weights(weights, is_aggregator, client_training_results)
