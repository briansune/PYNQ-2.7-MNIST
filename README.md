# PYNQ-2.7-MNIST

## Compiled version is targeting XC7Z010-CLG484

## MNIST Network
![MNIST Layers](image/mnist_layers.png)

### Baseline of the above network
Test loss: 0.204

Test accuracy: 0.956

### Quantized MNIST network

Test loss: 0.206

Test accuracy: 0.954

### Vivado Resource usage

> XC7Z010-CLG484

| LUT    | 7048 | 17600 | 40.045456 |
|--------|------|-------|-----------|
| LUTRAM | 415  | 6000  | 6.916667  |
| FF     | 8055 | 35200 | 22.883522 |
| BRAM   | 39   | 60    | 65.0      |
| DSP    | 73   | 80    | 91.25     |
| BUFG   | 1    | 32    | 3.125     |

