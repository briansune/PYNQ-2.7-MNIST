from tensorflow import keras
from fxpmath import Fxp
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle


# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

f = open('mnist.pkl', 'rb')
data = pickle.load(f, encoding='bytes')
f.close()

(x_train, y_train), (x_test, y_test) = data
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


model = keras.models.load_model('./mnist_model/mnist_lr.h5')

model.summary()

fxp_loss, fxp_accuracy = model.evaluate(x_test, y_test, verbose=False)
print(f'Test loss: {fxp_loss:.3}')
print(f'Test accuracy: {fxp_accuracy:.3}')

w_dict = {}
for layer in model.layers:
    print(model.get_layer(layer.name))
    if model.get_layer(layer.name).get_weights():
        w_dict[layer.name] = model.get_layer(layer.name).get_weights()
        # print mean and standard deviation from weights and bias
        print('{} (weights):\tmean = {}\tstd = {}'.format(
            layer.name, np.mean(w_dict[layer.name][0]),
            np.std(w_dict[layer.name][0])))
        print('{} (bias):\t\tmean = {}\tstd = {}\n'.format(
            layer.name, np.mean(w_dict[layer.name][1]),
            np.std(w_dict[layer.name][1])))

fxp_ref = Fxp(None, dtype='fxp-s8/7')

w_fxp_dict = {}
print(w_dict.keys())

for layer in w_dict.keys():
    w_fxp_dict[layer] = [
        Fxp(w_dict[layer][0], like=fxp_ref),
        Fxp(w_dict[layer][1]-w_dict[layer][1], like=fxp_ref),
    ]

for layer, values in w_fxp_dict.items():
    model.get_layer(layer).set_weights(values)

fxp_loss, fxp_accuracy = model.evaluate(x_test, y_test, verbose=False)
print(f'Test loss: {fxp_loss:.3}')
print(f'Test accuracy: {fxp_accuracy:.3}')

model.save('./mnist_model_fx/mnist_fx.h5')
model.save('./mnist_model_fx')
