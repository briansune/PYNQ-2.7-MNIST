import numpy as np


class Cnn:

    def __init__(self):
        self.dtype = np.int16

    def conv2d(self, din, kernel, stride=1):
        kout_channel, kin_channel, kernel_height, kernel_width = kernel.shape
        in_channel, in_height, in_width = din.shape
        output_channel = kout_channel
        output_height = in_height - (kernel_height - stride)
        output_width = in_width - (kernel_width - stride)
        layer = np.zeros(
            (output_channel, output_height, output_width)).astype(self.dtype)
        for z in range(0, output_channel):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    a = din[
                        :,
                        y * stride:y * stride + kernel_height,
                        x * stride:x * stride + kernel_width]
                    tmp = np.sum(a * kernel[z]).astype(self.dtype)
                    layer[z][y][x] = tmp if tmp > 0 else 0
        return layer

    def maxpool2d(self, din, kszie=2):
        in_channel, in_height, in_width = din.shape
        output_channel = in_channel
        output_height = int(in_height / kszie)
        output_width = int(in_width / kszie)
        layer = np.zeros(
            (output_channel, output_height, output_width)).astype(self.dtype)
        for z in range(0, output_channel):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    tmp = din[
                          z,
                          y * kszie:y * kszie + kszie,
                          x * kszie:x * kszie + kszie]
                    layer[z][y][x] = max(tmp.ravel())
        return layer.astype(self.dtype)

    def flatten(self, din):
        return din.ravel().astype(self.dtype)

    def fc(self, din, kernel, reduction=0, relu=1):
        kout_channel, kernel_len = kernel.shape
        layer = np.zeros(kout_channel).astype(self.dtype)
        for x in range(0, kout_channel):
            tmp = (np.sum(din * kernel[x])).astype(self.dtype)
            if relu:
                layer[x] = tmp >> reduction if tmp > 0 else 0
            else:
                layer[x] = tmp >> reduction
        return layer


