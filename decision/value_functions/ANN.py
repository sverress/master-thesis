import math
import tempfile

import numpy as np
from tensorflow import keras
import tensorflow as tf


class ANN:
    def __init__(
        self,
        network_structure,
        input_dimension,
        trace_decay,
        discount_factor,
        learning_rate,
    ):
        self.eligibilities = []
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor

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
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(
            loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"]
        )

        self.reset_eligibilities()

    def predict(self, state_features):
        return float(self._predict(tf.convert_to_tensor([state_features])))

    def _predict(self, features):
        return self.model(features)[0][0]

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
    def gen_loss(self, features, target):
        return (target - self._predict(features)) ** 2

    def fit(self, features, target, td_error, epochs=1):
        params = self.model.trainable_weights
        features, td_error, target = [
            tf.convert_to_tensor(input_value, dtype="float32")
            for input_value in [features, td_error, target]
        ]
        for epoch_id in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.gen_loss(features, target)
            gradients = tape.gradient(loss, params)
            gradients = self.modify_gradients(gradients, td_error, epoch_id)
            self.model.optimizer.apply_gradients(zip(gradients, params))

    def modify_gradients(self, gradients, td_error, epoch_id):
        # Taking every 2 array since every other array is biases
        for i in range(0, len(self.eligibilities), 2):
            if epoch_id == 0:
                self.eligibilities[i] = tf.math.add(
                    self.eligibilities[i] * self.trace_decay * self.discount_factor,
                    gradients[i],
                )
            gradients[i] = tf.math.multiply(self.eligibilities[i], td_error)
        return gradients

    def __getstate__(self):
        state = self.__dict__.copy()
        # Adding bytes from file to model attribute
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            keras.models.save_model(self.model, fd.name, overwrite=True)
            state["model"] = fd.read()
        eligibilities = []
        for eligibility_array in state["eligibilities"]:
            numpy_array = eligibility_array.numpy()
            # Adding bytes from file to as eligibility tensor list
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=True) as fd:
                np.save(fd.name, numpy_array)
                eligibilities.append(fd.read())
        state["eligibilities"] = eligibilities
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state["model"])
            fd.flush()
            self.model = keras.models.load_model(fd.name)
        eligibilities = []
        for eligibility_string in state["eligibilities"]:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=True) as fd:
                fd.write(eligibility_string)
                fd.flush()
                numpy_array = np.load(fd.name)
                eligibilities.append(tf.convert_to_tensor(numpy_array, dtype="float32"))
        self.eligibilities = eligibilities
