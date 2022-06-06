import numpy as np


class Cnn:

    def __init__(self):
        self.dtype = np.int16

    def conv2d(self, din, kernel, stride=1):
        kout_channel, kernel_height, kernel_width, kin_channel = kernel.shape
        in_height, in_width, in_channel = din.shape
        output_channel = kout_channel
        output_height = in_height - (kernel_height - stride)
        output_width = in_width - (kernel_width - stride)
        layer = np.zeros(
            (output_height, output_width, output_channel)).astype(self.dtype)
        for z in range(0, output_channel):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    a = din[
                        y * stride:y * stride + kernel_height,
                        x * stride:x * stride + kernel_width,
                        :].astype(self.dtype)
                    b = kernel[z].astype(self.dtype)
                    layer[y][x][z] = np.sum(a * b)
        return layer

    def relu2d(self, din, reduce=0):
        if len(din.shape) == 3:
            in_height, in_width, in_channel = din.shape
            output_channel = in_channel
            output_height = in_height
            output_width = in_width
            layer = np.zeros(
                (output_height, output_width, output_channel)).astype(self.dtype)
            for z in range(0, output_channel):
                for y in range(0, output_height):
                    for x in range(0, output_width):
                        layer[y][x][z] = din[y][x][z] if din[y][x][z] > 0 else 0
        else:
            layer = np.zeros(din.shape).astype(self.dtype)
            for z in range(0, din.shape[0]):
                layer[z] = din[z] if din[z] > 0 else 0

        if self.dtype != np.float32:
            layer = layer >> reduce
        return layer

    def maxpool2d(self, din, kszie=2):
        in_height, in_width, in_channel = din.shape
        output_channel = in_channel
        output_height = int(in_height / kszie)
        output_width = int(in_width / kszie)
        layer = np.zeros(
            (output_height, output_width, output_channel)).astype(self.dtype)
        for z in range(0, output_channel):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    tmp = din[
                          y * kszie:y * kszie + kszie,
                          x * kszie:x * kszie + kszie,
                          z]
                    layer[y][x][z] = max(tmp.ravel())
        return layer.astype(self.dtype)

    def flatten(self, din):
        return din.ravel().astype(self.dtype)

    def fc(self, din, kernel):
        kout_channel, kernel_len = kernel.shape
        layer = np.zeros(kout_channel).astype(self.dtype)
        for x in range(0, kout_channel):
            layer[x] = (np.sum(din * kernel[x])).astype(self.dtype)
        return layer
