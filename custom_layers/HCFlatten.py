import math
import tensorflow as tf

from tensorflow.python.keras import layers
from hilbertcurve.hilbertcurve import HilbertCurve


class HCFlatten(layers.Layer):

    def __init__(self, **kwargs):
        super(HCFlatten, self).__init__(**kwargs)

    def build(self, input_shape):
        self.side_length = input_shape[1]
        if 2**math.log2(self.side_length) != self.side_length:
            raise
        dim = len(input_shape) -2
        iters = int(math.log2(input_shape[1]))
        self.hc = HilbertCurve(n=dim, p=iters)

        self.side_length = input_shape[1]
        self.channel_dim = input_shape[-1]
        self.place_holder = tf.ones((self.side_length ** 2, 1))
        self.idxs = []
        for i in range(self.side_length**2):
            x, y = self.hc.coordinates_from_distance(i)
            idx = x+self.side_length*y
            self.idxs.append(idx)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.side_length**2, 1)

    def call(self, inputs):
        shape = (-1, self.side_length**2, self.channel_dim)
        inputs = tf.reshape(inputs, shape)
        def apply_gather(x):
            return tf.gather(x, self.idxs)

        #outputs = tf.reshape(tf.map_fn(apply_gather, inputs), shape + (1,))
        outputs = tf.map_fn(apply_gather, inputs)

        return outputs

if __name__ == "__main__":
    shape = (10,256,256,1)
    input = tf.reshape(tf.range(0, 256**2*10, 1), shape)
    layer = HCFlatten(input_shape=shape)
    layer.build(shape)
    output = layer(input)
    for i in range(10):
        print("{} -> {}".format(input[0][0][i][0], output[0][i][0]))

