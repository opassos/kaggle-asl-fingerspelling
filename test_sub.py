import tflite_runtime.interpreter as tflite

import time
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import Levenshtein as Lev
from sklearn.model_selection import GroupShuffleSplit

TOTAL = 1000 #None

SEL_FEATURES = json.load(open('inference_args.json'))['selected_columns']

def load_relevant_data_subset(example):
    pq_path = f".data/parquet/{example['file_id']}/{example['sequence_id']}.parquet"
    loaded = pd.read_parquet(pq_path, columns=SEL_FEATURES).values
    return loaded

with open ("dataset/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i,j in character_map.items()}

df = pd.read_csv('dataset/train.csv')
train_idx, val_idx = next(GroupShuffleSplit(test_size=.1, n_splits=2, random_state = 42).split(df, groups=df['phrase']))
val_df = df.iloc[val_idx].reset_index(drop=True)

idx = 0
example = df.loc[idx]
loaded = load_relevant_data_subset(example)
print(loaded.shape)
frames = loaded

def wer(s1, s2):
    seqlen = len(s1)
    lvd = Lev.distance(s1, s2)
    return lvd, seqlen

wer('foo', 'bar'), wer('foo', 'foz'), wer('f', 'ff'), wer('ff', 'f')


interpreter = tflite.Interpreter('model.tflite')
found_signatures = list(interpreter.get_signature_list().keys())

REQUIRED_SIGNATURE = 'serving_default'
REQUIRED_OUTPUT = 'outputs'

prediction_fn = interpreter.get_signature_runner("serving_default")
output_lite = prediction_fn(inputs=frames)
prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output_lite[REQUIRED_OUTPUT], axis=1)])
prediction_str, example['phrase'], wer(prediction_str, example['phrase'])

st = time.time()
cnt = 0
total = TOTAL or len(val_df)
model_time = 0

lens = 0
dists = 0

print(f"{'idx':3s} | {'ld_acc':5s} | {'true':32s} | {'predicted':32s}")
print("" + "-"*100)
for i in range(total):
    example = val_df.loc[i]
    loaded = load_relevant_data_subset(example)

    md_st = time.time()
    output_ = prediction_fn(inputs=loaded)
    model_time += time.time() - md_st

    prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output_[REQUIRED_OUTPUT], axis=1)])
    lvd, seqlen = wer(example['phrase'], prediction_str) 
    lens += seqlen
    dists += lvd    
    print(f"{i:3d} | {(lens - dists)/lens:.5f} | {example['phrase']:32s} | {prediction_str:32s}")

print(f'WER: {(lens - dists)/lens:.5f}')
print(f'Mean time: {(time.time() - st)/total:.7f}')
print(f'Mean time only infer: {model_time/total:.7f}')