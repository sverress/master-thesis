import tempfile
import time

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
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
                keras.layers.Dense(layer, kernel_regularizer=keras.regularizers.L2())
            )
            self.model.add(keras.layers.Activation("sigmoid"))
        # The last layer needs to have a single value function output
        self.model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(
            loss="mean_squared_error",
            optimizer=optimizer,
            metrics=["mean_absolute_percentage_error", "mean_absolute_error"],
        )

        self.reset_eligibilities()
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/{network_structure}_{int(time.time())}",
            profile_batch=100000000,  # https://github.com/tensorflow/tensorboard/issues/2819
        )

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

    def fit(self, features, target, epochs=1):
        params = self.model.trainable_weights
        features, target = [
            tf.convert_to_tensor(input_value, dtype="float32")
            for input_value in [features, target]
        ]
        for epoch_id in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.gen_loss(features, target)
            gradients = tape.gradient(loss, params)
            gradients = self.modify_gradients(gradients, np.sqrt(loss), epoch_id)
            self.model.optimizer.apply_gradients(zip(gradients, params))

    def batch_fit(self, features_list, target_list, **kwargs):
        self.model.fit(
            features_list, target_list, callbacks=[self.tensorboard], **kwargs
        )

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
        state["tensorboard"] = None
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
        self.tensorboard = ModifiedTensorBoard()


class ModifiedTensorBoard(TensorBoard):
    """
    Since normal tensorboard will create a new log file for every .fit call we create a custom Tensorboard class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
        self._train_dir = self.log_dir
        self._train_step = 0

        # Overriding this method to stop creating default log writer

    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self._train_step += 1
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
