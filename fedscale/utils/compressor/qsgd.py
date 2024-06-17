import numpy as np
import torch

from fedscale.utils.compressor import Compressor

class QSGDCompressor(Compressor):

    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num

    def compress(self, tensor, name=""):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, norm

        self._compressed_size = 32 + tensor.numel() * (1 + np.ceil(np.log2(self.quantum_num)))

        ctx = shape, name
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        tensor_compressed, norm = tensor_compressed
        shape, name = ctx

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
    
    def calculate_size(self, numel):
        res_bits = numel * (1 + np.ceil(np.log2(self.quantum_num + 1)))
        self._compressed_size  = 32 + res_bits # in bits
        return self._compressed_size
    