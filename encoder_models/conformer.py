from typing import Optional, Tuple, TypedDict
import numpy as np
import tensorflow as tf

from .attention import RelPositionMultiHeadAttention

class CTCLoss(tf.keras.layers.Layer):

    def __init__(self, pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id

    def call(self, labels, logits):
        labels_len = tf.math.count_nonzero(labels != self.pad_token_id, axis=-1, dtype=tf.int32)
        logits_len = tf.math.count_nonzero(logits._keras_mask, axis=-1, dtype=tf.int32)
        loss = tf.nn.ctc_loss(
                labels=labels,
                logits=logits,
                label_length=labels_len,
                logit_length=logits_len,
                blank_index=self.pad_token_id,
                logits_time_major=False
            )
        loss_mask = tf.cast(labels_len <= logits_len, tf.float32) 
        loss = tf.reduce_mean(loss * loss_mask)
        return loss

def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        alpha: int = 1,
        beta: int = 0,
        name="positional_encoding",
        **kwargs,
    ):
        super().__init__(trainable=False, name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def build(
        self,
        input_shape,
    ):
        dmodel = input_shape[-1]
        assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

    @staticmethod
    def encode(
        max_len,
        dmodel,
    ):
        pos = tf.expand_dims(tf.range(max_len - 1, -1, -1.0, dtype=tf.float32), axis=1)
        index = tf.expand_dims(tf.range(0, dmodel, dtype=tf.float32), axis=0)

        pe = pos * (1 / tf.pow(10000.0, (2 * (index // 2)) / dmodel))

        # Sin cos will be [max_len, size // 2]
        # we add 0 between numbers by using padding and reshape
        sin = tf.pad(tf.expand_dims(tf.sin(pe[:, 0::2]), -1), [[0, 0], [0, 0], [0, 1]], mode="CONSTANT", constant_values=0)
        sin = tf.reshape(sin, [max_len, dmodel])
        cos = tf.pad(tf.expand_dims(tf.cos(pe[:, 1::2]), -1), [[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=0)
        cos = tf.reshape(cos, [max_len, dmodel])
        # Then add sin and cos, which results in [time, size]
        pe = tf.add(sin, cos)
        return tf.expand_dims(pe, axis=0)  # [1, time, size]

    def call(
        self,
        inputs,
        **kwargs,
    ):
        # inputs shape [B, T, V]
        _, max_len, dmodel = shape_list(inputs)
        pe = self.encode(max_len * self.alpha + self.beta, dmodel)
        return tf.cast(pe, dtype=inputs.dtype)

    def get_config(self):
        conf = super().get_config()
        conf.update({"alpha": self.alpha, "beta": self.beta})
        return conf

class ConvolutionModule(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, kernel_size: int, downsample: bool, dropout: float) -> None:
        super().__init__()
        self.downsample = downsample
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.pointwise_conv1 = tf.keras.layers.Conv1D(2*embed_dim, kernel_size=1, activation=tf.nn.gelu)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv1D(kernel_size, strides=2 if downsample else 1, padding='same')
        self.subsampler = tf.keras.layers.AveragePooling1D(strides=2, padding="same") if downsample else tf.keras.layers.Identity()
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.act_swish = tf.keras.layers.Activation(tf.nn.swish)
        self.pointwise_conv2 = tf.keras.layers.Conv1D(embed_dim, kernel_size=1)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        residual = self.subsampler(x)
        x = self.layernorm(x)
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x, training=training)
        x = self.act_swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x, training=training)
        return residual + x
    
    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.downsample:
                mask = mask[:, ::2]
        return mask
    
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.layernrom = tf.keras.layers.LayerNormalization()
        self.expander = tf.keras.layers.Dense(ff_dim, activation=tf.nn.swish)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.compressor = tf.keras.layers.Dense(embed_dim)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        residual = x
        x = self.layernrom(x)
        x = self.expander(x)
        x = self.dropout1(x, training=training)
        x = self.compressor(x)
        x = self.dropout2(x, training=training)
        return residual + 1/2 * x

class ConformerBlock(tf.keras.layers.Layer):
    """
    Transformer encoder block: uses self-attention and MLP to encode the input sequence
    """
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int, 
            ff_dim: int, 
            dropout: float,
            kernel_size: int = 8,
            downsample: bool = False,
        ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = RelPositionMultiHeadAttention(num_heads=num_heads, head_size=embed_dim//num_heads)
        self.ffn1 = FeedForward(embed_dim, ff_dim, dropout)
        self.ffn2 = FeedForward(embed_dim, ff_dim, dropout)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.conv_module = ConvolutionModule(embed_dim, kernel_size=kernel_size, downsample=downsample, dropout=dropout)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, pos: tf.Tensor, training: bool=False) -> tf.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            training: bool
            mask: (batch_size, landamrk_seq_len, landmark_seq_len)
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        x = self.ffn1(x)
        xn = self.layernorm1(x)
        x = x + self.dropout(self.mha((xn, xn, xn, pos), training=training), training=training)
        x = self.conv_module(x, training=training)
        x = self.ffn2(x)
        x = self.layernorm2(x)
        return x

    def compute_mask(self, inputs, mask=None):
        return self.conv_module.compute_mask(inputs, mask)

class Input(TypedDict):
    landmarks: tf.Tensor

class Output(TypedDict):
    logits: tf.Tensor
    embeddings: Optional[tf.Tensor]

# data generated on
# https://www.kaggle.com/code/coldfir3/aslfs-dataset-compression-complete-4
FACE = [
    46, 52, 53, 65, # right_eyebrow
    295, 283, 282, 276, # left_eyebrow
    7, 159, 155, 145, # right_eye
    382, 386, 249, 374, # left_eye
    324, 13, 78, 14, # mouth
    168, # between_eyes
    205, # right_cheek
    425, # leftCheek
    ]
BODY = [
    0, # nose
    11, # right_shoulder
    12, # left_Shoulder
    13, # right_elbow
    14, # left_elbow
]
HANDS = list(range(21))

BLOCKS = (
    [f"{x}_right_hand_{i}" for i in range(21) for x in "xyz"],
    [f"{x}_left_hand_{i}" for i in range(21) for x in "xyz"],
    [f"{x}_pose_{i}" for i in BODY for x in "xyz"],
    [f"{x}_face_{i}" for i in FACE for x in "xyz"]
)
FEATURES_TO_LOAD = [item for sublist in BLOCKS for item in sublist]

class Config:

    landmarks_padding_value: float = 0.0
    
    pad_token: str = '^'
    tokens: str = pad_token + " !#$%&'()*+,-./0123456789:;=?@[_abcdefghijklmnopqrstuvwxyz~"
    pad_token_id: int = tokens.index(pad_token)
    vocab_size: int = len(tokens)

    char2num = {k:v for v, k in enumerate(tokens)}
    num2char = {v:k for k, v in char2num.items()}

    feature_columns = FEATURES_TO_LOAD
    blocks_len = [len(b) for b in BLOCKS]
    blocks_idx = [list(range(i-l, i)) for i, l in zip(np.cumsum([len(b) for b in BLOCKS]), [len(b) for b in BLOCKS])]
    n_features = len(FEATURES_TO_LOAD)
    n_landmarks = n_features // 3
    dim = 3

    def __init__(
        self,
        max_frames: int = 512,
        max_text_len: int = 32,
        embed_dim: int = 144,
        n_layers: Tuple[int, int, int] = (3, 3, 3),
        n_heads: int = 4,
        ff_ratio: int = 4,
        kernel_size: int = 8,
        embedder_dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
    ) -> None:
        self.max_frames = max_frames
        self.max_text_len = max_text_len
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_ratio = ff_ratio
        self.kernel_size = kernel_size
        self.embedder_dropout = embedder_dropout
        self.encoder_dropout = encoder_dropout
        self.classifier_dropout = classifier_dropout
        
    def to_dict(self):
        return self.__dict__

class Model(tf.keras.Model):
    def __init__(self, config: Config, landmark_preprocessor: Optional[tf.keras.layers.Layer]=None) -> None:
        super().__init__()
        self.config = config
        self.landmark_preprocessor = landmark_preprocessor
        self.embedder = tf.keras.layers.Dense(config.embed_dim, use_bias=False)
        self.embedder_dropout = tf.keras.layers.Dropout(config.embedder_dropout)
        self.pe = PositionalEncoding()
        self.stage1 = [ConformerBlock(
                config.embed_dim, 
                config.n_heads, 
                config.embed_dim*config.ff_ratio, 
                config.encoder_dropout,
                config.kernel_size,
                downsample=False,
            ) for _ in range(config.n_layers[0])]
        self.stage1ds = ConformerBlock(
                config.embed_dim, 
                config.n_heads, 
                config.embed_dim*config.ff_ratio, 
                config.encoder_dropout,
                config.kernel_size,
                downsample=True,
            )
        self.stage2 = [ConformerBlock(
                config.embed_dim, 
                config.n_heads, 
                config.embed_dim*config.ff_ratio, 
                config.encoder_dropout,
                config.kernel_size,
                downsample=False,
            ) for _ in range(config.n_layers[1])]
        self.stage2ds = ConformerBlock(
                config.embed_dim, 
                config.n_heads, 
                config.embed_dim*config.ff_ratio, 
                config.encoder_dropout,
                config.kernel_size,
                downsample=True,
            )
        self.stage3 = [ConformerBlock(
                config.embed_dim, 
                config.n_heads, 
                config.embed_dim*config.ff_ratio, 
                config.encoder_dropout,
                config.kernel_size,
                downsample=False,
            ) for _ in range(config.n_layers[2])]
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dropout(config.classifier_dropout),
            tf.keras.layers.Dense(config.vocab_size, use_bias=False),
        ], name="head")

    def call(self, landmarks: tf.Tensor, training: bool):

        if self.landmark_preprocessor is not None:
            landmarks = self.landmark_preprocessor(landmarks, training=training)

        embeddings = self.embedder(landmarks)
        pe = self.pe(embeddings)
        embeddings = self.embedder_dropout(embeddings, training=training)
        for block in self.stage1:
            embeddings = block(embeddings, pe, training=training)
        embeddings = self.stage1ds(embeddings, pe, training=training)
        for block in self.stage2:
            embeddings = block(embeddings, pe, training=training)
        embeddings = self.stage2ds(embeddings, pe, training=training)
        for block in self.stage3:
            embeddings = block(embeddings, pe, training=training)

        logits = self.head(embeddings, training=training)

        return Output(
            logits=logits,
            embeddings=embeddings,
        )

    @tf.function
    def _step(self, x: tf.Tensor, y: tf.Tensor, training: bool):
        output = self(x, training)
        logits = output['logits']
        loss = self.compiled_loss(
            y,
            logits,
        )
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, logits)
        return loss

    def train_step(self, batch: Input):
        with tf.GradientTape() as tape:
            loss = self._step(batch["landmarks"], batch['text'], training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, batch: Input):
        self._step(batch["landmarks"], batch['text'], training=False)
        return {m.name: m.result() for m in self.metrics}

    def generate_batch(self, batch: Input):
        logits = self(batch["landmarks"], training=False)['logits']
        preds = tf.argmax(logits, axis=-1)
        if hasattr(logits, "_keras_mask"):
            preds =  tf.where(logits._keras_mask, preds, tf.ones_like(preds)*self.config.pad_token_id)
        diff = tf.not_equal(preds[:,:-1], preds[:,1:])
        diff = tf.concat([tf.ones((preds.shape[0], 1), dtype=tf.bool), diff], axis=-1)
        preds = tf.where(diff, preds, tf.ones_like(preds)*self.config.pad_token_id)
        preds = tf.ragged.boolean_mask(preds, preds!=0)
        return preds
    
    def generate(self, landmarks):
        return self.generate_batch({"landmarks": landmarks[None]})[0]


class Tokenizer:
    def __init__(self, config: Config):
        self.char2num = config.char2num
    def __call__(self, example):
        '''
        given the example, tokenizes the text
        '''
        text = example['phrase']
        tokens = list(text)
        ids = [self.char2num[token] for token in tokens]
        return {"text": ids}


class DataCollator:
    def __init__(self, config: Config, with_id: bool=False):
        self.max_frames = config.max_frames
        self.max_text_len = config.max_text_len
        self.landmarks_padding_value = config.landmarks_padding_value
        self.pad_token_id = config.pad_token_id
        self.with_id = with_id

    def __call__(self, pre_batch):
        batch = {k: [x[k] for x in pre_batch] for k in pre_batch[0].keys()}

        # shuffle the examples
        indices = np.arange(len(batch['landmarks']))
        np.random.shuffle(indices)
        landmarks: list[np.ndarray] = [batch['landmarks'][i] for i in indices]
        text: list[np.ndarray] = [batch['text'][i] for i in indices]

        # get the maximum sizes for the batch
        max_frames = max([len(landmark) for landmark in landmarks])
        max_frames = min(max_frames, self.max_frames)
        max_text_len = max([len(input_id) for input_id in text])
        max_text_len = min(max_text_len, self.max_text_len)

        # pad or resize the examples to their max size
        landmarks = tf.keras.utils.pad_sequences(
            landmarks, 
            value=self.landmarks_padding_value,
            maxlen=max_frames,
            padding="post",
            truncating="post",
            dtype="float32",
            )
        text = tf.keras.utils.pad_sequences(
            text, 
            value=self.pad_token_id,
            maxlen=max_text_len,
            padding="post",
            truncating="post"
            )

        if self.with_id:
            file_id = [batch['file_id'][i] for i in indices]
            sequence_id = [batch['sequence_id'][i] for i in indices]
            uid = [f"{file_id}_{sequence_id}" for file_id, sequence_id in zip(file_id, sequence_id)]
            return {
                "landmarks": landmarks,
                "text": text,
                "id": uid,
                "file_id": file_id,
                "sequence_id": sequence_id,
            }
        return {
            "landmarks": landmarks,
            "text": text,
        }     

# test the model if main
if __name__ == "__main__":

    from datasets import DatasetDict

    config = Config()

    data_collator = DataCollator(config)

    dataset = (
        DatasetDict.load_from_disk('.data/hf_datasets')['train']
        .select(range(512))
        .map(Tokenizer(config=config))
        .to_tf_dataset(
            batch_size=8,
            shuffle=True,
            drop_remainder=True,
            collate_fn=data_collator,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE).cache()
    )

    batch = next(iter(dataset))
    for k, v in batch.items():
        print(k, v.shape, v.dtype)


    class Preprocessor(tf.keras.layers.Layer):
        def __init__(self, pad_value: float=0.0):
            super().__init__()
            self.pad_value = pad_value
        def call(self, landmarks: tf.Tensor, training: bool=True) -> tf.Tensor:
            # fill nas with zeros
            landmarks = tf.where(tf.math.is_nan(landmarks), tf.zeros_like(landmarks), landmarks)
            return landmarks
        def compute_mask(self, inputs, mask=None):
            is_pad = tf.not_equal(inputs, self.pad_value)
            return tf.reduce_all(is_pad, axis=-1)

    model = Model(config, Preprocessor())
    output = model(batch["landmarks"], training=True)
    for k, v in output.items():
        print(k, v.shape)
        if hasattr(v, "_keras_mask"):
            print(k, tf.math.count_nonzero(v._keras_mask, axis=-1))

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(),
        loss=CTCLoss(config.pad_token_id),
        )
    
    print(model.summary())

    loss = model.train_step(batch)['loss']
    print(f"Loss: {loss}")

    gens = model.generate_batch(batch)
    print([gen.shape[0] for gen in gens])

    gen = model.generate(batch["landmarks"][0])
    print(gen.shape)

    model.fit(
        dataset,
        epochs=5,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
        ]
    )