import numpy as np
import math
import random
import sys
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

from keras.layers import BatchNormalization as BatchNorm
from keras.utils import to_categorical
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.callbacks import LambdaCallback
from keras.layers import Bidirectional
from keras import backend as K
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from textblob import TextBlob

class RNNProcessor:

  def create_and_train_rnn_model(self, X_train, y_train, X_valid, y_valid, batch_size=256, epochs=100, neurons=500, lstm_layers=1, learning_rate=0.01, dropout=0.2):
    K.clear_session();

    model = Sequential()

    # Input Layer
    rs = True
    if(lstm_layers == 1):
      rs = False

    model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=rs))
    if(dropout > 0):
      model.add(Dropout(dropout))

    for layer in range(1, lstm_layers):
      rs = True
      if(layer == lstm_layers - 1):
          rs = False

      model.add(LSTM(neurons, return_sequences=rs))
      if(dropout > 0):
        model.add(Dropout(dropout))

    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid))

    self.__plot_history(history)

    return model

  def __plot_history(self, history):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set_title('Model Train vs Validation Loss', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    axes[0].legend(['Train', 'Validation'], loc='upper right')

    axes[1].plot(history.history['accuracy'])
    axes[1].plot(history.history['val_accuracy'])
    axes[1].set_title('Model Train vs Validation Accuracy', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    axes[1].legend(['Train', 'Validation'], loc='upper right')

    fig.tight_layout()
    plt.show()