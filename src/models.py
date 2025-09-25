import tensorflow as tf
from tensorflow import keras

from keras.layers import Bidirectional
from keras.layers import LSTM, GRU, Dense, LayerNormalization
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding, Bidirectional, Concatenate
from keras import backend as K


class BI_LSTM_Insertion_Deletion2(tf.keras.Model):
  def __init__(self, d_bilstm = 512, d_ffn = [128], num_bi_layers = 3, output_size = 4, rnn_type = 'gru'):
    """

    Initializes the encoder for insertion deletion channels
    --------------------------------------------------------------
    args
    * d_bilstm: hidden size number of bidirectional rnns
    * d_ffn: hidden size number of multi layer perceptron
    * num_bi_layers: number of bi-directional layers
    * rnn_type: type of rnn - LSTM or GRU
    """
    super().__init__()

    # parameters init.
    self.num_bi_layers = num_bi_layers
    self.d_bilstm = d_bilstm
    self.rnn_type = rnn_type
    self.d_ffn = d_ffn

    # layers
    if self.rnn_type == 'lstm':
      self.bir_layers = [Bidirectional(LSTM(self.d_bilstm, return_sequences = True))
                        for _ in range(self.num_bi_layers)]
    elif self.rnn_type == 'gru':
      self.bir_layers = [Bidirectional(GRU(self.d_bilstm, return_sequences = True))
                        for _ in range(self.num_bi_layers)]

    # batch normalization layers
    self.nor_layers = [LayerNormalization() for _ in range(self.num_bi_layers)]
    # concat layers
    self.mlp_layers = [Dense(k, activation = 'relu') for k in d_ffn]
    self.output_layer = Dense(output_size)

  def call(self, x):
    #x = self.input_layer(x)
    for bir_layer,nor_layer in zip(self.bir_layers, self.nor_layers):
      x = bir_layer(x)
      x = nor_layer(x)

    for layer in self.mlp_layers:
      x = layer(x)

    x = self.output_layer(x)

    return x
  
class BI_LSTM_Insertion_Deletion3(tf.keras.Model):
  def __init__(self, d_bilstm = 512, d_ffn = [128], num_bi_layers = 3, output_size = 4):
    """

    Initializes the encoder for insertion deletion channels
    --------------------------------------------------------------
    args
    * d_bilstm: hidden size number of bidirectional rnns
    * d_ffn: hidden size number of multi layer perceptron
    * num_bi_layers: number of bi-directional layers
    * rnn_type: type of rnn - LSTM or GRU
    """
    super().__init__()

    # parameters init.
    self.num_bi_layers = num_bi_layers
    self.d_bilstm = d_bilstm
    self.d_ffn = d_ffn
    self.bir_layers = [Bidirectional(LSTM(self.d_bilstm, return_sequences = True))
                      for _ in range(self.num_bi_layers)]

    # batch normalization layers
    self.nor_layers = [LayerNormalization() for _ in range(self.num_bi_layers)]
    # concat layers
    self.mlp_layers = [Dense(k, activation = 'relu') for k in d_ffn]
    self.output_layer = Dense(output_size)

  def call(self, x):
    #x = self.input_layer(x)
    for bir_layer,nor_layer in zip(self.bir_layers, self.nor_layers):
      x = bir_layer(x)
      x = nor_layer(x)

    for layer in self.mlp_layers:
      x = layer(x)

    x = self.output_layer(x)

    return x
  
  