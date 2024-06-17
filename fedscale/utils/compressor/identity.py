from fedscale.utils.compressor import Compressor

class IdentityCompressor(Compressor):

    def __init__(self):
        super().__init__()

    def compress(self, tensor, name=""):
        return tensor, {}

    def decompress(self, tensor_compressed, ctx={}):
        return tensor_compressed
    
    def calculate_size(self, numel):
        self._compressed_size = numel * 32 # fp32
        return self._compressed_size