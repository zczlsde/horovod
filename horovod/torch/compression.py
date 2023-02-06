# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed

def _compression_topk(tensor, ratio):
    # Get the topk percent values of tensor
    k = max(1, int(tensor.numel() * ratio))
    _, indices = torch.topk(tensor.abs(), k)
    values = tensor[indices]
    
    return values, indices.type(torch.int32)

def _compression_random(tensor, compress_ratio):
    # Get the total elements of current tensor
    num = tensor.numel()
    
    # Calculate the number of elements that required to be filtered
    k = max(1, int(num * compress_ratio))
    # Use permutation to do the random generation
    indices = torch.randperm(num, device=tensor.device)[:k]
    # Get the k values
    values = tensor[indices]
    
    return values, indices.type(torch.int32) # Return a tuple


class RandomCompressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    
    def __init__(self, ratio = 0.3):
        super().__init__()
        # Instead just cut 30 percents, this compressor class could cut the given percentage of ratio
        self.ratio = ratio
        
    @staticmethod
    def compress(self, tensor):
        """Compress tensor by selecting given percentage (ratio) values"""
        tensor_compressed = _compression_random(tensor, self.ratio)
        ctx = tensor.numel()
        
        return tensor_compressed, ctx

    @staticmethod
    def decompress(self, tensor, ctx):
        """Upcasts the tensor to the initialization size"""
        num = ctx
        values, indices = tensor
        # Reshape to the original size and filter the missing values using 0
        tensor_decompressed = torch.zeros(num, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        
        return tensor_decompressed


class TopKCompressor(Compressor):
    
    """TopK method mentioned in the email"""
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def compress(self, tensor):
        tensors = _compression_topk(tensor, self.ratio)
        ctx = tensor.numel()
        return tensors, ctx

    def decompress(self, tensor, ctx):
        num = ctx
        values, indices = tensor
        tensor_decompressed = torch.zeros(num, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor
    
    """Compress using the random compressor"""
    randomcom = RandomCompressor
    
    """Compress using the topk compressor"""
    topkcom = TopKCompressor