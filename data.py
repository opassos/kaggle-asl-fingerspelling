import pandas as pd

class GetLandmarks:
    def __init__(self, base_path, min_frame_ratio=4):
        self.base_path = base_path
        self.mfr = min_frame_ratio
    def __call__(self, example):
        '''
        given the example, reads the parquet file associated with it
        '''
        path = f"{self.base_path}/{example['file_id']}/{example['sequence_id']}.parquet"
        landmarks = pd.read_parquet(path).values
        return {"landmarks": landmarks}

if __name__ == "__main__":
    from datasets import DatasetDict, Dataset
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_csv('dataset/train.csv')
    train_idx, val_idx = next(GroupShuffleSplit(test_size=0.1, n_splits=2, random_state = 42).split(df, groups=df['phrase']))

    train_df = df.iloc[train_idx].reset_index().sort_values(["n_frames", "seq_len"], ascending=False).reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index().sort_values(["n_frames", "seq_len"], ascending=False).reset_index(drop=True)

    aux_df = (
        pd.read_csv('dataset/supplemental_metadata.csv')
        .query("n_frames > 0")
        .sort_values(["n_frames", "seq_len"], ascending=False)
        .reset_index(drop=True)
    )
    datasets = (
        DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'valid': Dataset.from_pandas(val_df),
            'aux': Dataset.from_pandas(aux_df)
        })
        .map(GetLandmarks(base_path=".data/parquet"), num_proc=10)
    ).save_to_disk('.data/hf_datasets')