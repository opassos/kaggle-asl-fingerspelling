import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import Levenshtein as Lev
from datasets import DatasetDict, concatenate_datasets
from tqdm.auto import tqdm

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
from encoder_models.conformer import Config, Model, DataCollator, Tokenizer

CHECKPOINT = ".models/sandy-gorge-43/last.h5"
TOTAL = None
BS = 128

# TODO: save config
model_config = Config(
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


model = Model(model_config, Preprocessor(model_config))
model(np.zeros((1, model_config.max_frames, len(FEATURE_COLUMNS)), dtype=np.float32), training=False)
model.load_weights(CHECKPOINT)

datasets = (
    DatasetDict.load_from_disk('.data/hf_datasets')
    .map(Tokenizer(config=model_config), num_proc=10)
)

data_collator = DataCollator(model_config, with_id=True)

valid_ds = datasets['valid'].to_tf_dataset(
    batch_size=BS,
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

idx_to_token = model_config.num2char.copy()
idx_to_token.pop(model_config.pad_token_id)
if hasattr(model_config, "bos_token_id"):
    idx_to_token.pop(model_config.bos_token_id)
if hasattr(model_config, "eos_token_id"):
    idx_to_token.pop(model_config.eos_token_id)

print(f"{'idx':3s} | {'ld_acc':5s} | {'true':32s} | {'predicted':32s}")
print("" + "-"*100)
all_file_ids = []
all_sequence_ids = []
all_prediction_strings = []
all_gt_strings = []
for i, batch in enumerate(valid_ds):

    all_file_ids += batch["file_id"].numpy().tolist()
    all_sequence_ids += batch["sequence_id"].numpy().tolist()

    # decode predictions
    decoded_predictions = model.generate_batch(batch).numpy()
    prediction_strings = [
        "".join([idx_to_token.get(s, "") for s in dp]) 
        for dp in decoded_predictions
    ]
    all_prediction_strings += prediction_strings
    gt_strings = [
        "".join([idx_to_token.get(s, "") for s in batch["text"][i].numpy()]) 
        for i in range(batch["text"].shape[0])
    ]
    all_gt_strings += gt_strings
    for j, (prediction_str, gt_str) in enumerate(zip(prediction_strings, gt_strings)):
        lvd, seqlen = wer(gt_str, prediction_str)
        lens += seqlen
        dists += lvd
        print(f"{(i*BS+j):3d} | {(lens - dists)/lens:.5f} | {gt_str:32s} | {prediction_str:32s}")

# pd.DataFrame({
#     "file_id": all_file_ids,
#     "sequence_id": all_sequence_ids,
#     "prediction": all_prediction_strings,
#     "gt": all_gt_strings,
# }).to_csv("dataset/stage2/valid_predictions.csv", index=False)

print(f'WER: {(lens - dists)/lens:.5f}')