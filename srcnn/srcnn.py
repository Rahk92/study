import tensorflow as tf
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt

width = 33
height = 33

def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(64, (9,9), padding='same', activation='relu', input_shape=(height, width, 1)))
    SRCNN.add(Conv2D(32, (1,1), padding='same', activation='relu'))
    SRCNN.add(Conv2D(1, (5,5), padding='same', activation='linear'))

    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=[PSNR, 'mean_squared_error'])
    return SRCNN


def data_prepare(file, dataset):
    data = HDF5Matrix(file, dataset)
    data = np.transpose(data, (2, 0, 1))
    data = np.reshape(data, (data.shape[0], height, width, 1))
    return data


def data_p(file, dataset):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get(dataset))
        train_data = np.reshape(data, (data.shape[2], width, height, 1))
        return train_data

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    rmse = K.sqrt(K.mean(K.square(y_pred - y_true)))
    return 20. * tf_log10(255. /rmse)


def train():
    srcnn_model = model()
    print(srcnn_model.summary())
    data = data_prepare('train_patch.h5', 'patch')
    label = data_prepare('train_label.h5', 'label')
    val_data = data_prepare('val_patch.h5', 'patch')
    val_label = data_prepare('val_label.h5', 'label')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    hist = srcnn_model.fit(data, label, batch_size=128, epochs=500, validation_data=(val_data, val_label), callbacks=[early_stopping], verbose=2)

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    srcnn_model.save_weights('./200_epoch_weights.h5')


if __name__ == "__main__":
    train()
