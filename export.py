import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras

from utils.augment import TimeWarp, RandomAffine
from utils.preprocess import (
    chunk_name2idx,
    get_chunk_idx,
    MySequential,
    NaFillerLean,
    NaFiller,
    ChunkSelect,
    ChunkNormalize,
    Normalize3D,
    ChunkConcat,
    DominantHandSelector,
    ComputeSpeed,
)
from encoder_models.conformer import Config, Model


CHECKPOINT = ".models/fallen-jazz-49/last.h5"

# TODO: save config
model_config = Config(
    max_frames = 512,
    max_text_len = 32,
    embed_dim = 144,
    n_layers = (3, 3, 4),
    n_heads = 4,
    ff_ratio = 4,
    kernel_size = 16,
    embedder_dropout = 0.2,
    encoder_dropout = 0.2,
    classifier_dropout = 0.4,
)

FEATURE_COLUMNS = model_config.feature_columns

class Preprocessor(keras.layers.Layer):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs, name="preprocessor")
        self.mask = keras.layers.Masking(mask_value=model_config.landmarks_padding_value)
        self.na_filler = NaFillerLean(config.n_features)
        self.select = ChunkSelect(config.blocks_idx)
        self.normalize = Normalize3D()
        self.concat = ChunkConcat()
        # augmentations
        self.time_warp = TimeWarp()
        self.random_affine = RandomAffine()

    def call(self, x: tf.Tensor, training=False):
        x = self.mask(x)
        mask = x._keras_mask
        xs = self.select(x)
        xs = [self.normalize(x) for x in xs]
        x = self.concat(xs)
        x._keras_mask = mask
        x = self.na_filler(x)
        if training:
            x = self.random_affine(x, training=training)
            x = self.time_warp(x, training=training)
        return x

class TFLiteModel(tf.Module):
    def __init__(self, config: Config, checkpoint_path: str = None):
        super().__init__()
        self.config = config
        self.model = Model(config, Preprocessor(config))
        self.model(np.zeros((1, config.max_frames, len(FEATURE_COLUMNS)), dtype=np.float32), training=False)
        if checkpoint_path is not None:
            self.model.load_weights(checkpoint_path)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(FEATURE_COLUMNS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        # Preprocess Data
        x = inputs
        # truncate to max_frames
        x = x[:self.config.max_frames]
        x = self.model.generate(x)
        # x = self.model.generate_decoder(x)
        x = tf.one_hot(x-1, self.config.vocab_size-1) 
        return {'outputs': x}

tflite_model = TFLiteModel(model_config, CHECKPOINT)
print("="*50)
print(tflite_model(np.zeros((128,len(FEATURE_COLUMNS)), dtype=np.float32))['outputs'].shape)
print("="*50)

converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

with open('inference_args.json', "w") as json_file:
    json.dump({"selected_columns" : FEATURE_COLUMNS}, json_file)

print("="*50)
os.system("du -h model.tflite")
print("="*50)

interpreter = tf.lite.Interpreter("model.tflite")
prediction_fn = interpreter.get_signature_runner("serving_default")

print("="*50)
# print(prediction_fn(inputs=np.zeros((0,len(FEATURE_COLUMNS)), dtype=np.float32))['outputs'].shape)
print(prediction_fn(inputs=np.zeros((1,len(FEATURE_COLUMNS)), dtype=np.float32))['outputs'].shape)
print(prediction_fn(inputs=np.zeros((1000,len(FEATURE_COLUMNS)), dtype=np.float32))['outputs'].shape)
print("="*50)

if os.path.exists("submission.zip"):
    os.system("rm submission.zip")
os.system("zip submission.zip  'model.tflite' 'inference_args.json'")
