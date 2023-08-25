import wandb

api = wandb.Api(timeout=300)

from datasets import DatasetDict, concatenate_datasets
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from wandb.keras import WandbCallback


from utils.data import shuffle_group
from utils.optim import OneCycleScheduler
from utils.metrics import GenerateCallback
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
from encoder_models.conformer import CTCLoss, DataCollator, Tokenizer, Config, Model

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
training_config = dict(
    batch_size = 128,
    epochs = 50,
    max_lr = 2e-3,
    clipvalue = 1.0,
    pct_start = 0.3,
    div_factor = 25,
    weight_decay = 0.005,
    final_div_factor = 100,
)

data_collator = DataCollator(model_config, with_id=True)

datasets = (
    DatasetDict.load_from_disk('.data/hf_datasets')
    .map(Tokenizer(config=Config), num_proc=10)
)

df = pd.read_csv('dataset/train.csv')

train_ds = shuffle_group(
    concatenate_datasets([
        datasets['train'],#.filter(lambda x: x['n_frames']/x['seq_len'] > 4, num_proc=10),
        # datasets['valid']#.filter(lambda x: x['n_frames']/x['seq_len'] > 4, num_proc=10),
        datasets['aux'].filter(lambda x: (x['n_frames']-x['n_nan'])/x['seq_len'] > 4, num_proc=10),
    ]), training_config['batch_size'], 42).to_tf_dataset(
    batch_size=training_config['batch_size'],
    shuffle=False,
    drop_remainder=True,
    collate_fn=data_collator,
).shuffle(64, reshuffle_each_iteration=True).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

valid_ds = shuffle_group(datasets['valid'], training_config['batch_size'], 42).to_tf_dataset(
    batch_size=training_config['batch_size'],
    shuffle=False,
    drop_remainder=False,
    collate_fn=data_collator,
).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

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

landmark_preprocessor = Preprocessor(model_config)
model = Model(model_config, landmark_preprocessor)

optimizer = keras.optimizers.AdamW(
    learning_rate=training_config['max_lr'], 
    weight_decay=training_config['weight_decay'],
    clipvalue=training_config['clipvalue'],
    )

model.compile(
    optimizer=optimizer,
    loss=CTCLoss(model_config.pad_token_id),
    )

# run one batch to compile model
batch = next(iter(train_ds))
output = model(batch["landmarks"], training=True)
for k, v in output.items():
    print(k, v.shape)
    if hasattr(v, "_keras_mask"):
        print(k, tf.math.count_nonzero(v._keras_mask, axis=-1))

with wandb.init(
    project='kaggle-basics', 
    entity='curitibocas', 
    config={**model_config.to_dict(), **training_config},
    dir=".models",
    ) as run:
    print(model.summary())
    model.fit(
        train_ds, 
        validation_data=valid_ds, 
        epochs=training_config['epochs'],
        callbacks=[
            keras.callbacks.TerminateOnNaN(),
            GenerateCallback(train_ds, model_config, freq=5, n_batches=10, name="train"),
            GenerateCallback(valid_ds, model_config, freq=5, n_batches=None, name="valid"),
            OneCycleScheduler(**training_config),
            WandbCallback("val_loss", save_model=False),
            keras.callbacks.ModelCheckpoint(
                filepath=f".models/{run.name}/best.h5",
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                ),
        ]
        )
    # save model
    model.save_weights(f".models/{run.name}/last.h5")