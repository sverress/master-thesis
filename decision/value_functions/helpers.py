import math
import numpy as np
import tensorflow.keras


def convert_string_list_to_tuple_list(string_list):
    return [(int(string_pos[0]), int(string_pos[1])) for string_pos in string_list]


def string_to_np_array(string):
    return np.array([int(string_num) for string_num in string])


class SplitGD(tensorflow.keras.Sequential):
    """
    !!CODE TAKEN FROM it3105 WEBSITE!!
    https://www.idi.ntnu.no/emner/it3105/materials/code/splitgd.py
    This "exposes" the gradients during gradient descent by breaking the call to "fit" into two calls: tape.gradient
    and optimizer.apply_gradients.  This enables intermediate modification of the gradients.  You can find many other
    examples of this concept online and in the (excellent) book "Hands-On Machine Learning with Scikit-Learn, Keras,
    and Tensorflow", 2nd edition, (Geron, 2019).

    This class serves as a wrapper around a keras model.  Then, instead of calling keras_model.fit, just call
    SplitGD.fit.  To use this class, just subclass it and write your own code for the "modify_gradients" method.
    """

    def __init__(self, trace_decay, discount_factor):
        """
        Wrapper method with updated fit method for eligibility traces
        :param trace_decay:
        :param gamma:
        """
        super().__init__()
        self.eligibilities = []
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.reset_eligibilities()

    def reset_eligibilities(self):
        import tensorflow as tf

        self.eligibilities = [
            tf.convert_to_tensor(
                np.zeros(self.trainable_weights[i].numpy().shape), dtype=tf.float32,
            )
            for i in range(len(self.trainable_weights))
        ]

    # This returns a tensor of losses, OR the value of the averaged tensor.  Note: use .numpy() to get the
    # value of a tensor.
    def gen_loss(self, features, targets, avg=False):
        import tensorflow as tf

        predictions = self(
            tf.convert_to_tensor(features, dtype="float32")
        )  # Feed-forward pass to produce outputs/predictions
        loss = tf.reduce_sum((targets - predictions) ** 2)
        return tf.reduce_mean(loss).numpy() if avg else loss

    def fit(self, features, targets, td_error, epochs=1, mbs=1, vfrac=0.1, **kwargs):
        import tensorflow as tf

        params = self.trainable_weights

        for _ in range(epochs):
            for _ in range(math.floor(epochs / mbs)):
                with tf.GradientTape() as tape:
                    loss = self.gen_loss(features, targets)
                    gradients = tape.gradient(loss, params)
                    gradients = self.modify_gradients(gradients, td_error)
                    self.optimizer.apply_gradients(zip(gradients, params))

    # Use the 'metric' to run a quick test on any set of features and targets.  A typical metric is some form of
    # 'accuracy', such as 'categorical_accuracy'.  Read up on Keras.metrics !!
    # Note that the model.metrics__names slot includes the name of the loss function (as 0th entry),
    # whereas the model.metrics slot does not include the loss function, hence the index+1 in the final line.
    # Use your debugger and go through the long list of slots for a keras model.  There are a lot of useful things
    # that you have access to.

    def gen_evaluation(self, features, targets, avg=False, index=0):
        import tensorflow as tf

        predictions = self(features)
        evaluation = self.metrics[index](targets, predictions)
        #  Note that this returns both a tensor (or value) and the NAME of the metric
        return (
            tf.reduce_mean(evaluation).numpy() if avg else evaluation,
            self.metrics_names[index + 1],
        )

    def status_display(self, features, targets, mode="Train"):
        print(mode + " *** ", end="")
        print("Loss: ", self.gen_loss(features, targets, avg=True), end=" : ")
        val, name = self.gen_evaluation(features, targets)
        print("Eval({0}): {1} ".format(name, val))

    def end_of_epoch_display(self, train_ins, train_targs, val_ins, val_targs):
        self.status_display(train_ins, train_targs, mode="Train")
        if len(val_ins) > 0:
            self.status_display(val_ins, val_targs, mode="Validation")

    def modify_gradients(self, gradients, td_error):
        """ Calculates the td_error in the weight and eligibility updates according to ei ← ei + ∂V(st)/∂wi
        and wi ← wi +αδei.
        :param gradients: tape.gradients of loss and weights
        :param td_error: δ ← r +γV(s')−V(s) calculated in critic
        :return: the amount of td_error in the weights of the NN
        """

        import tensorflow as tf

        td_error = tf.Variable(td_error, dtype=tf.float32)
        # Taking every 2 array since every other array is biases
        for i in range(0, len(self.eligebilities), 2):
            self.eligebilities[i] = tf.math.add(
                self.eligebilities[i] * self.trace_decay * self.discount_factor,
                gradients[i],
            )
            gradients[i] = tf.math.multiply(self.eligebilities[i], td_error)

        return gradients


# A few useful auxiliary functions
def gen_random_minibatch(inputs, targets, mbs=1):
    indices = np.random.randint(len(inputs), size=mbs)
    return inputs[indices], targets[indices]


# This returns: train_features, train_targets, validation_features, validation_targets
def split_training_data(inputs, targets, vfrac=0.1, mix=True):
    vc = round(vfrac * len(inputs))  # vfrac = validation_fraction
    # pairs = np.array(list(zip(inputs,targets)))
    if vfrac > 0:
        pairs = list(zip(inputs, targets))
        if mix:
            np.random.shuffle(pairs)
        vcases = pairs[0:vc]
        tcases = pairs[vc:]
        return (
            np.array([tc[0] for tc in tcases]),
            np.array([tc[1] for tc in tcases]),
            np.array([vc[0] for vc in vcases]),
            np.array([vc[1] for vc in vcases]),
        )
        #  return tcases[:,0], tcases[:,1], vcases[:,0], vcases[:,1]  # Can't get this to work properly
    else:
        return inputs, targets, [], []
