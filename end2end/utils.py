"""
Date: 03/09/2018
Version: 1.1
Description: 1) Contains the implementation of KMaxPooling layer for CNN, Attention layer for HAN,and Folding layer for CNN in keras
	     	 2) Contains the implementation of threshold score function.
Python Version: 3.6
Author: Team-1b-Deep Text Eval(Shlomi Hod, Maximilian Seidler,Vageesh Saxena)
"""

####################### Importing libraries ####################################
import tensorflow.contrib.keras as keras
from keras import regularizers, initializers, constraints
from keras import backend as K
from keras.engine import Layer, InputSpec
import tensorflow as tf

import keras
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda
)
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
################################################################################


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, axis=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

        assert axis in [1,2],  'expected dimensions (samples, filters, convolved_values),\
                   cannot fold along samples dimension or axis not in list [1,2]'
        self.axis = axis

        # need to switch the axis with the last elemnet
        # to perform transpose for tok k elements since top_k works in last axis
        self.transpose_perm = [0,1,2] #default
        self.transpose_perm[self.axis] = 2
        self.transpose_perm[2] = self.axis

    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        input_shape_list[self.axis] = self.k
        return tuple(input_shape_list)

    def call(self, x):
        # swap sequence dimension to get top k elements along axis=1
        transposed_for_topk = tf.transpose(x, perm=self.transpose_perm)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(transposed_for_topk, k=self.k, sorted=True, name=None)[0]

        # return back to normal dimension but now sequence dimension has only k elements
        # performing another transpose will get the tensor back to its original shape
        # but will have k as its axis_1 size
        transposed_back = tf.transpose(top_k, perm=self.transpose_perm)

        return transposed_back


class Folding(Layer):
    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], int(input_shape[2]/2))

    def call(self, x):
        input_shape = x.get_shape().as_list()

        # split the tensor along dimension 2 into dimension_axis_size/2
        # which will give us 2 tensors
        splits = tf.split(x, num_or_size_splits=int(input_shape[2]/2), axis=2)

        # reduce sums of the pair of rows we have split onto
        reduce_sums = [tf.reduce_sum(split, axis=2) for split in splits]

        # stack them up along the same axis we have reduced
        row_reduced = tf.stack(reduce_sums, axis=2)
        return row_reduced

class AttentionLayer(keras.layers.Layer):
    def __init__(self, context_vector_length=100, **kwargs):
        """
        An implementation of a attention layer. This layer
        accepts a 3d Tensor (batch_size, time_steps, input_dim) and
        applies a single layer attention mechanism in the time
        direction (the second axis).
        :param context_vector_lenght: (int) The size of the hidden context vector.
            If set to 1 this layer reduces to a standard attention layer.
        :param kwargs: Any argument that the baseclass Layer accepts.
        """
        self.context_vector_length = context_vector_length
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[2]

        # Add a weights layer for the
        self.W = self.add_weight(
            name='W', shape=(dim, self.context_vector_length),
            initializer=keras.initializers.get('uniform'),
            trainable=True
        )

        self.u = self.add_weight(
            name='context_vector', shape=(self.context_vector_length, 1),
            initializer=keras.initializers.get('uniform'),
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def _get_attention_weights(self, X):
        """
        Computes the attention weights for each timestep in X
        :param X: 3d-tensor (batch_size, time_steps, input_dim)
        :return: 2d-tensor (batch_size, time_steps) of attention weights
        """
        # Compute a time-wise stimulus, i.e. a stimulus for each
        # time step. For this first compute a hidden layer of
        # dimension self.context_vector_length and take the
        # similarity of this layer with self.u as the stimulus
        u_tw = K.tanh(K.dot(X, self.W))
        tw_stimulus = K.dot(u_tw, self.u)

        # Remove the last axis an apply softmax to the stimulus to
        # get a probability.
        tw_stimulus = K.reshape(tw_stimulus, (-1, tw_stimulus.shape[1]))
        att_weights = K.softmax(tw_stimulus)

        return att_weights

    def call(self, X):
        att_weights = self._get_attention_weights(X)

        # Reshape the attention weights to match the dimensions of X
        att_weights = K.reshape(att_weights, (-1, att_weights.shape[1], 1))
        att_weights = K.repeat_elements(att_weights, X.shape[-1], -1)

        # Multiply each input by its attention weights
        weighted_input = keras.layers.Multiply()([X, att_weights])

        # Sum in the direction of the time-axis.
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def get_config(self):
        config = {
            'context_vector_length': self.context_vector_length
        }
        base_config = super(AttentionLayer, self).get_config()
        return {**base_config, **config}

class HAN(Model):
    def __init__(
            self, max_words, max_sentences, output_size,
            embedding_matrix, word_encoding_dim=200,
            sentence_encoding_dim=200, inputs=None,
            outputs=None, name='han-for-docla'
    ):
        """
        A Keras implementation of Hierarchical Attention networks
        for document classification.
        :param max_words: The maximum number of words per sentence
        :param max_sentences: The maximum number of sentences
        :param output_size: The dimension of the last layer (i.e.
            the number of classes you wish to predict)
        :param embedding_matrix: The embedding matrix to use for
            representing words
        :param word_encoding_dim: The dimension of the GRU
            layer in the word encoder.
        :param sentence_encoding_dim: The dimension of the GRU
            layer in the sentence encoder.
        """
        self.max_words = max_words
        self.max_sentences = max_sentences
        self.output_size = output_size
        self.embedding_matrix = embedding_matrix
        self.word_encoding_dim = word_encoding_dim
        self.sentence_encoding_dim = sentence_encoding_dim


        in_tensor, out_tensor = self._build_network()

        super(HAN, self).__init__(
            inputs=in_tensor, outputs=out_tensor, name=name
        )

    def build_word_encoder(self, max_words, embedding_matrix, encoding_dim=200):
        """
        Build the model that embeds and encodes in context the
        words used in a sentence. The return model takes a tensor of shape
        (batch_size, max_length) that represents a collection of sentences
        and returns an encoded representation of these sentences.
        :param max_words: (int) The maximum sentence length this model accepts
        :param embedding_matrix: (2d array-like) A matrix with the i-th row
            representing the embedding of the word represented by index i.
        :param encoding_dim: (int, should be even) The dimension of the
            bidirectional encoding layer. Half of the nodes are used in the
            forward direction and half in the backward direction.
        :return: Instance of keras.Model
        """
        assert encoding_dim % 2 == 0, "Embedding dimension should be even"

        vocabulary_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        embedding_layer = Embedding(
            vocabulary_size, embedding_dim,
            weights=[embedding_matrix], input_length=max_words,
            trainable=False
        )

        sentence_input = Input(shape=(max_words,), dtype='int32')
        embedded_sentences = embedding_layer(sentence_input)
        encoded_sentences = Bidirectional(
            GRU(int(encoding_dim / 2), return_sequences=True)
        )(embedded_sentences)

        return Model(
            inputs=[sentence_input], outputs=[encoded_sentences], name='word_encoder'
        )

    def build_sentence_encoder(self, max_sentences, summary_dim, encoding_dim=200):
        """
        Build the encoder that encodes the vector representation of
        sentences in their context.
        :param max_sentences: The maximum number of sentences that can be
            passed. Use zero-padding to supply shorter sentences.
        :param summary_dim: (int) The dimension of the vectors that summarizes
            sentences. Should be equal to the encoding_dim of the word
            encoder.
        :param encoding_dim: (int, even) The dimension of the vector that
            summarizes sentences in context. Half is used in forward direction,
            half in backward direction.
        :return: Instance of keras.Model
        """
        assert encoding_dim % 2 == 0, "Embedding dimension should be even"

        text_input = Input(shape=(max_sentences, summary_dim))
        encoded_sentences = Bidirectional(
            GRU(int(encoding_dim / 2), return_sequences=True)
        )(text_input)
        return Model(
            inputs=[text_input], outputs=[encoded_sentences], name='sentence_encoder'
        )

    def _build_network(self):
        """
        Build the graph that represents this network
        :return: in_tensor, out_tensor, Tensors representing the input and output
            of this network.
        """
        in_tensor = Input(shape=(self.max_sentences, self.max_words))

        word_encoder = self.build_word_encoder(
            self.max_words, self.embedding_matrix, self.word_encoding_dim
        )

        word_rep = TimeDistributed(
            word_encoder, name='word_encoder'
        )(in_tensor)

        # Sentence Rep is a 3d-tensor (batch_size, max_sentences, word_encoding_dim)
        sentence_rep = TimeDistributed(
            AttentionLayer(), name='word_attention'
        )(word_rep)

        doc_rep = self.build_sentence_encoder(
            self.max_sentences, self.word_encoding_dim, self.sentence_encoding_dim
        )(sentence_rep)

        # We get the final representation by applying our attention mechanism
        # to the encoded sentences
        doc_summary = AttentionLayer(name='sentence_attention')(doc_rep)

        out_tensor = Dense(
            self.output_size, activation='softmax', name='class_prediction'
        )(doc_summary)

        return in_tensor, out_tensor

    def get_config(self):
        config = {
            'max_words': self.max_words,
            'max_sentences': self.max_sentences,
            'output_size': self.output_size,
            'embedding_matrix': self.embedding_matrix,
            'word_encoding_dim': self.word_encoding_dim,
            'sentence_encoding_dim': self.sentence_encoding_dim,
            'base_config': super(HAN, self).get_config()
        }

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Keras' API isn't really extendible at this point
        therefore we need to use a bit hacky solution to
        be able to correctly reconstruct the HAN model
        from a config. This therefore does not reconstruct
        a instance of HAN model, but actually a standard
        Keras model that behaves exactly the same.
        """
        base_config = config.pop('base_config')

        return Model.from_config(
            base_config, custom_objects=custom_objects
        )

    def predict_sentence_attention(self, X):
        """
        For a given set of texts predict the attention
        weights for each sentence.
        :param X: 3d-tensor, similar to the input for predict
        :return: 2d array (num_obs, max_sentences) containing
            the attention weights for each sentence
        """
        att_layer = self.get_layer('sentence_attention')
        prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.input, dummy_layer).predict(X)

def threshold_score(y_true, y_pred):
	"""
	y_true: actual output
	y_pred: predicted output
	"""
	# threshold_score : an evaluation criterion which says its still better
	# if texts from level 1 are classified in level 2 than getting it assigned
	# to 4 or 5.
	return np.sum(np.abs(y_true - y_pred) <= 1) / len(y_true)

def plot_accuracy_and_loss_curves(history,evaluation_method,model):
	# plots the accuracy and loss curve for the mentioned evaluation method.
	"""
	history: fitted model
	evaluation_method: categorical_crossentropy, threshold_score etc(dtype:string)
	model: neural network architecture(dtype:string)
	"""
	fig1 = plt.figure()
	plt.plot(history.history['loss'],'r',linewidth=3.0)
	plt.plot(history.history['val_loss'],'b',linewidth=3.0)
	plt.legend(['Training loss', 'Testing Loss'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Loss',fontsize=16)
	plot_title = "Loss Curves :" + model + "_for_" + evaluation_method
	fig_name =  "loss_" + model + "_" + evaluation_method + ".png"
	save_address = "plots/" + fig_name
	plt.title(plot_title,fontsize=16)
	fig1.savefig(save_address)
	plt.show()

	fig2=plt.figure()
	plt.plot(history.history['acc'],'r',linewidth=3.0)
	plt.plot(history.history['val_acc'],'b',linewidth=3.0)
	plt.legend(['Training Accuracy', 'Testing Accuracy'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Accuracy',fontsize=16)
	plot_title = "Accuracy Curves :" + model + "_for_" + evaluation_method
	fig_name =  "accuracy_" + model + "_" + evaluation_method
	save_address = "plots/" + fig_name
	plt.title(plot_title,fontsize=16)
	fig2.savefig(save_address)
	plt.show()

################################################################################
