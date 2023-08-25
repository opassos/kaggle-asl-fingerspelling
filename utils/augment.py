from typing import Tuple, Optional, List
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow_addons as tfa

class TimeWarp(tf.keras.layers.Layer):
    def __init__(self, n_control_points: int=10, strenght: float=0.1, p: float=0.5):
        super().__init__()
        self.n_control_points = n_control_points
        self.strenght = strenght
        self.p = p
        self.target = None
        
    def _call(self, x: tf.Tensor) -> tf.Tensor:
        '''
        randomly sample control points and warp them in the time dimension
        '''
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        x = tf.repeat(tf.expand_dims(x, axis=2), 2, axis=2)

        flow = tfp.math.interp_regular_1d_grid(
            x=tf.linspace(0.0, 1.0, h),
            x_ref_min=0.0,
            x_ref_max=1.0,
            y_ref=self.target,
            axis=-1
        ) * tf.cast(h, tf.float32)
        flow = tf.stack([flow, tf.zeros_like(flow)], axis=-1)
        flow = tf.expand_dims(flow, axis=0)
        flow = tf.expand_dims(flow, axis=2)
        flow = tf.repeat(flow, 2, axis=2)
        flow = tf.repeat(flow, b, axis=0)

        x_warped = tfa.image.dense_image_warp(
            x, flow
        )
        x_warped = x_warped[:,:,0]

        return x_warped

    def call(self, x, training=False): 
        if not training:
            self.target = None
            return x
        if np.random.uniform() > self.p:
            self.target = None
            return x
        self.target = tf.random.uniform([self.n_control_points], -self.strenght, self.strenght)
        return self._call(x)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return mask
        if self.target is None:
            return mask
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.repeat(mask, 2, axis=-1)
        mask = tf.cast(mask, tf.float32)
        mask = self._call(mask)
        mask = mask[...,0]
        mask = tf.cast(mask, tf.bool)
        return mask
    
class RandomAffine(tf.keras.layers.Layer):
    def __init__(
            self, 
            scale: tuple[int, int] = (0.8,1.2),
            shear: tuple[int, int] = (-0.05,0.05),
            shift: tuple[int, int] = (-0.2,0.2),
            degree: tuple[int, int] = (-15,15),
            p: float = 0.5
            ):
        super().__init__()
        self.scale_range = scale
        self.shear_range = shear
        self.shift_range = shift
        self.degree_range = degree
        self.p = p
        self.supports_masking = True

    def generate_affine_matrix(self) -> tf.Tensor:
        scale = np.random.uniform(*self.scale_range)
        scale_tfm = tf.eye(3) * scale

        shear = np.random.uniform(*self.shear_range)
        shear_tfm = tf.constant([
            [1.,    shear,  0.],
            [shear, 1.,     0.],
            [0.,    0.,     1.]
        ])

        shiftx = np.random.uniform(*self.shift_range)
        shifty = np.random.uniform(*self.shift_range)
        shift_tfm = tf.constant([
            [1.,    0.,     shiftx],
            [0.,    1.,     shifty],
            [0.,    0.,     1.]
        ])
        angle = np.random.uniform(*self.degree_range) * np.pi / 180.
        cs = np.cos(angle)
        sn = np.sin(angle)
        angle_tfm = tf.constant([
            [cs,    sn,     0.],
            [-sn,   cs,     0.],
            [0.,    0.,     1.]
        ], dtype=tf.float32)
        
        affine_tfm = scale_tfm @ shear_tfm @ shift_tfm @ angle_tfm
        return affine_tfm

    def call(self, landmarks: tf.Tensor, training: bool=False) -> tf.Tensor:
        '''landmarks [bs, seq_len, n_points*3]'''
        if not training:
            return landmarks
        
        if np.random.uniform() > self.p:
            return landmarks

        bs = tf.shape(landmarks)[0]
        seq_len = tf.shape(landmarks)[1]
        n_points = tf.shape(landmarks)[2]//3
        xyz = tf.reshape(landmarks, [bs*seq_len, n_points, 3])
        xy = xyz[...,:2]
        z = xyz[...,2:]

        affine_matrix = self.generate_affine_matrix()
        xy = tf.concat([xy, tf.ones([bs*seq_len, n_points, 1])], axis=-1)
        xy = tf.transpose(xy, [0,2,1])
        xy = tf.matmul(affine_matrix, xy)
        xy = tf.transpose(xy, [0,2,1])[...,:2]
        
        xyz = tf.concat([xy, z], axis=-1)
        xyz = tf.reshape(xyz, [bs, seq_len, n_points*3])
        return xyz
    
if __name__ == "__main__":

    preprocessor = keras.Sequential([
        keras.layers.Masking(mask_value=-1),
        TimeWarp(),
    ])

    x = tf.concat([
        tf.random.uniform((2, 100, 128)),
        -1.0*tf.ones((2, 100, 128))
    ], axis=1)
    y = preprocessor(x, training=True)
    print(y.shape, tf.math.count_nonzero(y._keras_mask, axis=1)[0])

    y = preprocessor(x, training=False)
    print(y.shape, tf.math.count_nonzero(y._keras_mask, axis=1)[0])