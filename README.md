# PYNQ-2.7-MNIST

## MNIST Network
![MNIST Layers](image/mnist_layers.png)

### Baseline of the above network
Test loss: 0.204

Test accuracy: 97.64

### Quantized MNIST network

Test loss: 0.206

Test accuracy: 96.73

### Vivado Resource usage

| LUT    | 6415 | 17600 | 36.448864 |
|--------|------|-------|-----------|
| LUTRAM | 934  | 6000  | 15.566667 |
| FF     | 6125 | 35200 | 17.400568 |
| BRAM   | 27   | 60    | 45.0      |
| DSP    | 47   | 80    | 58.749996 |
| BUFG   | 1    | 32    | 3.125     |

