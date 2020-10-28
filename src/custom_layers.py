import keras.backend as k
from keras.engine.topology import Layer


class L2Diff(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(L2Diff, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = None

    def call(self, x, mask=None):
        left = x[0]
        right = x[1]
        score = k.sum(k.square(left - right), axis=1, keepdims=True)
        return score

    def compute_output_shape(self, input_shape):
        return None, 1


class L2Off(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(L2Off, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = None

    def call(self, x, mask=None):
        left = x[0]
        right = x[1]
        offset = k.sum(k.square(left - right), axis=1, keepdims=True)
        return offset

    def compute_output_shape(self, input_shape):
        return None, 1


class Mean(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = False
        super(Mean, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = None

    def call(self, x, mask=None):
        mean = k.mean(x, 1, False)
        return mean

    def compute_output_shape(self, input_shape):
        return None, input_shape[-1]
