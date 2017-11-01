from keras.layers import Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.models import load_model,Sequential
from seg_model import build_model1
from keras.applications.vgg16 import VGG16




IMG_WIDTH = 320
IMG_HEIGHT = 180


def _conv_bn_elu(nb_filter, kernel_size, strides=(1, 1)):
    def f(input):
        conv = Conv2D(nb_filter, kernel_size, strides=strides,
                      kernel_initializer='he_normal', padding='same')(input)
        norm = BatchNormalization()(conv)
        return ELU()(norm)
    return f


def build_model1():
    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = _conv_bn_elu(16, 3)(inp)
    x = _conv_bn_elu(16, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(32, 3)(x)
    x = _conv_bn_elu(32, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(32, 3)(x)
    x = _conv_bn_elu(32, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(48, 3)(x)
    x = _conv_bn_elu(48, 3)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='sigmoid')(x)
    x = Lambda(lambda x: K.sum(x, axis=[1, 2]))(x)
    return Model(inp, x)


def build_model2():
    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = _conv_bn_elu(16, 3)(inp)
    x = _conv_bn_elu(16, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(32, 3)(x)
    x = _conv_bn_elu(32, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(32, 3)(x)
    x = _conv_bn_elu(32, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(48, 3)(x)
    x = _conv_bn_elu(48, 3)(x)
    x = MaxPooling2D()(x)

    # Mask detection
    x = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='sigmoid')(x)

    # Mask refinement
    x = _conv_bn_elu(4, 3)(x)
    x = _conv_bn_elu(4, 3)(x)
    x = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='sigmoid')(x)
    x = Lambda(lambda x: K.sum(x, axis=[1, 2]))(x)
    return Model(inp, x)

def build_pre_train_model(run_id="",layer_name="max_pooling2d_4"):
    #pretrained car detector
    model_path = '../models/seg_{}.hdf5'.format(run_id)
    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = _conv_bn_elu(16, 3)(inp)
    x = _conv_bn_elu(16, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(32, 3)(x)
    x = _conv_bn_elu(32, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(32, 3)(x)
    x = _conv_bn_elu(32, 3)(x)
    x = MaxPooling2D()(x)

    x = _conv_bn_elu(48, 3)(x)
    x = _conv_bn_elu(48, 3)(x)
    x = MaxPooling2D()(x)
    model = Model(inp, x)

    model.load_weights(filepath=model_path, by_name=True)

    #for layer in model.layers: layer.trainable = False

    x = model.output

    # # Mask detection
    x = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='sigmoid')(x)
    #
    # # Mask refinement
    x = _conv_bn_elu(4, 3)(x)
    x = _conv_bn_elu(4, 3)(x)
    x = Conv2D(1, 3, kernel_initializer='he_normal', padding='same', activation='sigmoid')(x)
    x = Lambda(lambda x: K.sum(x, axis=[1, 2]))(x)


    return Model(inp, x)



if __name__ == '__main__':
    model = build_pre_train_model(run_id="2017-10-07-11-33-19")
    model.summary()
