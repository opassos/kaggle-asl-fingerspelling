from typing import Tuple, Optional, List
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class MySequential(tf.keras.layers.Layer):
    def __init__(self, steps) -> None:
        super().__init__()
        self.steps = steps
        self.supports_masking = True
    def call(self, x, training=False):
        for step in self.steps:
            x = step(x, training)
        return x
    
class DominantHandSelector(layers.Layer):
    def __init__(
            self, 
            left_hand_idx: list[int], 
            right_hand_idx: list[int],
            bypass_idx: list[int]=[],
            nax: int=3
        ) -> None:
        super().__init__()
        assert len(left_hand_idx) == len(right_hand_idx)
        assert len(left_hand_idx) % nax == 0
        self.n_coords = len(left_hand_idx)//nax
        self.nax = nax
        self.left_hand_idx = tf.constant(left_hand_idx, dtype=tf.int32)
        self.right_hand_idx = tf.constant(right_hand_idx, dtype=tf.int32)
        self.bypass_idx = tf.constant(bypass_idx, dtype=tf.int32)
        self.supports_masking = True

    def call(self, x: tf.Tensor, training: Optional[bool]=False) -> tf.Tensor:
        '''
        Picks the dominant hand from the landmarks
        the dominant hand has less NAs
        '''
        bypass = tf.gather(x, self.bypass_idx, axis=-1)
        left_hand = tf.gather(x, self.left_hand_idx, axis=-1)
        right_hand = tf.gather(x, self.right_hand_idx, axis=-1)
        # flip the x coordinates of the left hand
        if self.nax == 3:
            x_flipper = tf.constant([[[-1.0]*self.n_coords + [1.0]*self.n_coords + [1.0]*self.n_coords]])
        elif self.nax == 2:
            x_flipper = tf.constant([[[-1.0]*self.n_coords + [1.0]*self.n_coords]])
        else:
            raise ValueError("nax must be 2 or 3") 
        n_nan_left = tf.math.count_nonzero(tf.math.is_nan(left_hand), axis=-1, keepdims=True)
        n_nan_right = tf.math.count_nonzero(tf.math.is_nan(right_hand), axis=-1, keepdims=True)
        dominant_hand = tf.where(n_nan_left < n_nan_right, x_flipper*left_hand, right_hand)

        return tf.concat([dominant_hand, bypass], axis=-1)

class NaFiller(layers.Layer):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.supports_masking = True
    def call(self, x: tf.Tensor, training: Optional[bool]=False) -> tf.Tensor:
        return tf.where(tf.math.is_nan(x), tf.fill(tf.shape(x), self.value), x)

class NaFillerLean(layers.Layer):
    def __init__(self, dim) -> None:
        super().__init__()
        self.empty_embedding = tf.Variable(tf.zeros((1, 1, dim)))
        self.supports_masking = True
    def call(self, x: tf.Tensor, training: Optional[bool]=False) -> tf.Tensor:
        return tf.where(tf.math.is_nan(x), self.empty_embedding, x)

class ChunkSelect(layers.Layer):
    def __init__(self, chunks_idx: list[list[int]]) -> None:
        super().__init__()
        self.chunks_idx = chunks_idx
        self.supports_masking = True
    def call(self, x: tf.Tensor, training: Optional[bool]=False) -> list[tf.Tensor]:
        return [tf.gather(x, chunk_idx, axis=-1) for chunk_idx in self.chunks_idx]

class ChunkNormalize(layers.Layer):
    """
    Splits a tensor in chunks of size chunk_sizes and normalizes each chunk
    concatenates the mean and std of each chunk
    """
    def __init__(self, epsilon: float=1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.supports_masking = True
    def call(self, x: list[tf.Tensor], training: Optional[bool]=False) -> list[tf.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, n_landmark_coords)
        Returns:
            (batch_size, seq_len, n_landmark_coords + 2*len(chunk_sizes))
        """
        # normalize each chunk
        stats = [tf.nn.moments(chunk, axes=-1, keepdims=True) for chunk in x]
        x = [(chunk - mean) / (tf.sqrt(var + self.epsilon)) for chunk, (mean, var) in zip(x, stats)]
        return x

class Normalize3D(layers.Layer):
    """
    Splits a tensor in chunks of size chunk_sizes and normalizes each chunk
    concatenates the mean and std of each chunk
    """
    def __init__(self, epsilon: float=1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.supports_masking = True
    def call(self, x: tf.Tensor, training: Optional[bool]=False) -> list[tf.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, n_landmark_coords)
        Returns:
            (batch_size, seq_len, n_landmark_coords + 2*len(chunk_sizes))
        """
        # normalize each chunk
        bs = tf.shape(x)[0]
        sl = tf.shape(x)[1]
        nc = tf.shape(x)[2]
        x = tf.reshape(x, (bs, sl, nc//3, 3))
        mean, var = tf.nn.moments(x, axes=-2, keepdims=True)
        x = (x - mean) / (tf.sqrt(var + self.epsilon))
        x = tf.reshape(x, (bs, sl, nc))
        return x

class ChunkConcat(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.supports_masking = True
    def call(self, x: list[tf.Tensor], training: Optional[bool]=False) -> tf.Tensor:
        return tf.concat(x, axis=-1)

def chunk_name2idx(names, config):
    return [
        np.where([chunk_name in feat for feat in config.feature_columns])[0]
        for chunk_name in names
    ]
     
def get_chunk_idx(chunks): 
    i = 0
    _chunks = []
    for c in chunks:
        _chunks.append(list(range(i, i+len(c))))
        i += len(c)
    return _chunks

class ComputeSpeed(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.supports_masking = True
    def call(self, x: tf.Tensor, training: Optional[bool]=False) -> tf.Tensor:
        # compute speed
        return tf.concat([x[:, 1:] - x[:, :-1], tf.zeros_like(x[:, :1])], axis=1)

if __name__ == "__main__":

    preprocessor = keras.Sequential([
        keras.layers.Masking(mask_value=-1),
        DominantHandSelector(
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8, 9]
        ),
        ComputeSpeed(),
        NaFillerLean(2*7),
    ])

    x = tf.concat([
        tf.ones((2, 1, 10)),
        2*tf.ones((2, 1, 10)),
        tf.random.uniform((2, 100, 10)),
        -1*tf.ones((2, 100, 10))
    ], axis=1)
    y = preprocessor(x, training=True)
    print(y.shape, tf.math.count_nonzero(y._keras_mask, axis=1)[0])

    y = preprocessor(x, training=False)
    print(y.shape, tf.math.count_nonzero(y._keras_mask, axis=1)[0])
    print(x[0,:2], y[0,:2])