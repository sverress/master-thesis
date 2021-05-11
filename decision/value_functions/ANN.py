import math
import tempfile

import numpy as np
from tensorflow import keras
import tensorflow as tf


class ANN:
    def __init__(
        self, network_structure, input_dimension, trace_decay, discount_factor
    ):
        self.eligibilities = []
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.reset_eligibilities()

        self.model = keras.models.Sequential()
        # First layer needs input size as argument
        nodes_in_first_layer, *nodes_in_rest_of_layers = network_structure
        self.model.add(
            keras.layers.Dense(
                nodes_in_first_layer,
                input_dim=input_dimension,
            )
        )
        for layer in nodes_in_rest_of_layers:
            self.model.add(keras.layers.Dropout(0.5))
            self.model.add(
                keras.layers.Dense(
                    layer, activation="relu", kernel_regularizer=keras.regularizers.L2()
                )
            )
        # The last layer needs to have a single value function output
        self.model.add(keras.layers.Dense(1))
        self.model.compile(
            loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]
        )

    def predict(self, state_features):
        return float(self.model(tf.convert_to_tensor([state_features]))[0][0])

    def reset_eligibilities(self):
        self.eligibilities = [
            tf.convert_to_tensor(
                np.zeros(self.model.trainable_weights[i].numpy().shape),
                dtype=tf.float32,
            )
            for i in range(len(self.model.trainable_weights))
        ]

    # This returns a tensor of losses, OR the value of the averaged tensor.  Note: use .numpy() to get the
    # value of a tensor.
    def gen_loss(self, features, targets, avg=False):
        predictions = self.model(
            tf.convert_to_tensor(features, dtype="float32")
        )  # Feed-forward pass to produce outputs/predictions
        loss = tf.reduce_sum((targets - predictions) ** 2)
        return tf.reduce_mean(loss).numpy() if avg else loss

    def fit(self, features, targets, td_errors, epochs=1, mbs=1):
        raise NotImplemented("not able to handle multiple td_errors")
        params = self.model.trainable_weights
        for _ in range(epochs):
            for _ in range(math.floor(epochs / mbs)):
                with tf.GradientTape() as tape:
                    loss = self.gen_loss(features, targets)
                    gradients = tape.gradient(loss, params)
                    gradients = self.modify_gradients(gradients, td_errors)
                    self.model.optimizer.apply_gradients(zip(gradients, params))

    def modify_gradients(self, gradients, td_errors):
        # Taking every 2 array since every other array is biases
        for i in range(0, len(self.eligibilities), 2):
            self.eligibilities[i] = tf.math.add(
                self.eligibilities[i] * self.trace_decay * self.discount_factor,
                gradients[i],
            )
            gradients[i] = tf.math.multiply(
                self.eligibilities[i], tf.convert_to_tensor(td_errors)
            )
        return gradients

    def convert_model_to_string(self):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            keras.models.save_model(self.model, fd.name, overwrite=True)
            return fd.read()

    @staticmethod
    def load_model_from_string(model_string):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(model_string)
            fd.flush()
            return keras.models.load_model(fd.name)
