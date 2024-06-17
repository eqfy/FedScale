from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass


class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self):
        self._compressed_size = -1 # In bits

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")
    
    @abstractmethod
    def calculate_size(self, numel):
        """Calculate the total size given number of elements"""
        raise NotImplemented("calculate_size was not implemented.")
    
    @property
    def compressed_size(self):
        if self._compressed_size < 0:
            raise Exception("You must first compress a tensor in order to view its compressed size")
        return self._compressed_size

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)
