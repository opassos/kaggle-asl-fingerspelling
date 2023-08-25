import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import Levenshtein as Lev
from datasets import DatasetDict, concatenate_datasets

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
from encoder_models.conformer import DataCollator, Tokenizer
from encoder_models.conformer import ASLConfig as Config
from encoder_models.conformer import ASLConformer as Model

CHECKPOINT = ".models/sandy-gorge-43/last.h5"
TOTAL = None

# TODO: save config
config = Config(
    max_frames = 512,
    max_text_len = 32,
    embed_dim = 144,
    n_layers = (3, 3, 4),
    n_heads = 4,
    ff_ratio = 4,
    kernel_size = 16,
    embedder_dropout = 0.1,
    subsampler_dropout = 0.1,
    encoder_dropout = 0.2,
    classifier_dropout = 0.4,
)

FEATURE_COLUMNS = config.feature_columns

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
        x = self.na_filler(x)
        mask = x._keras_mask
        xs = self.select(x)
        xs = [self.normalize(x) for x in xs]
        x = self.concat(xs)
        x._keras_mask = mask
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

tflite_model = TFLiteModel(config, CHECKPOINT)

datasets = (
    DatasetDict.load_from_disk('.data/hf_datasets')
    .map(Tokenizer(config=Config), num_proc=10)
)

data_collator = DataCollator(config)

valid_ds = datasets['valid'].to_tf_dataset(
    batch_size=1,
    shuffle=False,
    drop_remainder=False,
    collate_fn=data_collator,
).prefetch(buffer_size=tf.data.AUTOTUNE)


with open ("dataset/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i,j in character_map.items()}

def wer(s1, s2):
    seqlen = len(s1)
    lvd = Lev.distance(s1, s2)
    return lvd, seqlen


lens = 0
dists = 0

print(f"{'idx':3s} | {'ld_acc':5s} | {'true':32s} | {'predicted':32s}")
print("" + "-"*100)
total = TOTAL or valid_ds.cardinality().numpy()
for i, batch in enumerate(valid_ds.take(total)):

    output = tflite_model(inputs=batch["landmarks"][0])["outputs"]
    phrase = "".join([rev_character_map.get(s-1, "") for s in batch["text"][0].numpy()])
    prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output, axis=1)])
    lvd, seqlen = wer(phrase, prediction_str) 
    lens += seqlen
    dists += lvd    
    print(f"{i:3d} | {(lens - dists)/lens:.5f} | {phrase:32s} | {prediction_str:32s}")

print(f'WER: {(lens - dists)/lens:.2f}')