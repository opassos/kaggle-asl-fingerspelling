import tensorflow as tf
from tqdm.auto import tqdm
import Levenshtein as Lev
import wandb

def compute_ld(s1, s2):
    seqlen = len(s1)
    lvd = Lev.distance(s1, s2)
    return lvd, seqlen

class GenerateCallback(tf.keras.callbacks.Callback):
    def __init__(
            self, 
            dataset, 
            config, 
            freq=1, 
            n_batches=None, 
            name="ld_gen",
            signature="generate_batch",
            txt_key="text",
        ):
        """
        Generates predictions for a dataset and computes the levenshtein distance
        Displays the last batch of outputs
        """
        super().__init__()
        self.freq = freq
        self.dataset = dataset
        self.idx_to_token = config.num2char.copy()
        self.idx_to_token.pop(config.pad_token_id) # remove pad
        if hasattr(config, "eos_token_id"):
            self.idx_to_token.pop(config.eos_token_id) # remove eos
        if hasattr(config, "bos_token_id"):
            self.idx_to_token.pop(config.bos_token_id) # remove bos
        self.n_batches = n_batches or len(self.dataset) - 1
        self.name = name
        self.signature = signature
        self.txt_key = txt_key

    def on_epoch_end(self, epoch, logs=None):

        if ((epoch + 1) % self.freq != 0) and (epoch != 0):
            return

        if wandb.run is not None:
            text_table = wandb.Table(columns=["epoch", "id", "ld", "gt", "pred"])

        generator = getattr(self.model, self.signature)
        lens = 0
        dists = 0
        for i, batch in tqdm(enumerate(self.dataset), desc="Generating predictions", total=self.n_batches):
            if i >= self.n_batches:
                break
            # decode predictions
            predictions = generator(batch)
            prediction_strings = [
                "".join([self.idx_to_token.get(s, "") for s in predictions[i].numpy()]) 
                for i in range(predictions.shape[0])
            ]
            gt_strings = [
                "".join([self.idx_to_token.get(s, "") for s in batch[self.txt_key][i].numpy()]) 
                for i in range(batch[self.txt_key].shape[0])
            ]
            for id, prediction_str, gt_str in zip( batch["id"], prediction_strings, gt_strings):
                lvd, seqlen = compute_ld(gt_str, prediction_str)
                lens += seqlen
                dists += lvd
                if text_table is not None:
                    text_table.add_data(epoch, id, max(0, (seqlen-lvd)/seqlen), gt_str, prediction_str)

        global_ld = (lens-dists)/lens

        # if wandb is enabled, log prediction_str and gt_str in a table, and log the ld
        if wandb.run is not None:
            wandb.log({
                f"{self.name}_ldgen": global_ld,
                f"text_table/{self.name}": text_table
                }, commit=False)



class GenerateSpellFixCallback(tf.keras.callbacks.Callback):
    def __init__(
            self, 
            dataset, 
            config, 
            freq=1, 
            n_batches=None, 
            name="ld_gen",
        ):
        """
        Generates predictions for a dataset and computes the levenshtein distance
        Displays the last batch of outputs
        """
        super().__init__()
        self.freq = freq
        self.dataset = dataset
        self.idx_to_token = config.num2char.copy()
        self.idx_to_token.pop(config.pad_token_id) # remove pad
        self.idx_to_token.pop(config.eos_token_id) # remove eos
        self.idx_to_token.pop(config.bos_token_id) # remove bos
        self.n_batches = n_batches or len(self.dataset) - 1
        self.name = name

    def on_epoch_end(self, epoch, logs=None):

        if ((epoch + 1) % self.freq != 0) and (epoch != 0):
            return

        if wandb.run is not None:
            text_table = wandb.Table(columns=["epoch", "ld", "gt", "original", "corrected"])

        lens = 0
        dists = 0
        for i, batch in tqdm(enumerate(self.dataset), desc="Generating predictions", total=self.n_batches):
            if i >= self.n_batches:
                break
            # decode predictions
            predictions = self.model.generate_batch(batch)
            prediction_strings = [
                "".join([self.idx_to_token.get(s, "") for s in predictions[i].numpy()]) 
                for i in range(predictions.shape[0])
            ]
            original_strings = [
                "".join([self.idx_to_token.get(s, "") for s in batch["input_ids"][i].numpy()]) 
                for i in range(batch["input_ids"].shape[0])
            ]
            gt_strings = [
                "".join([self.idx_to_token.get(s, "") for s in batch["output_ids"][i].numpy()]) 
                for i in range(batch["output_ids"].shape[0])
            ]
            for prediction_str, original_str, gt_str in zip(prediction_strings, original_strings, gt_strings):
                lvd, seqlen = compute_ld(gt_str, prediction_str)
                if seqlen == 0:
                    continue
                lens += seqlen
                dists += lvd
                if text_table is not None:
                    text_table.add_data(epoch, max(0, (seqlen-lvd)/seqlen), gt_str, original_str, prediction_str)

        global_ld = (lens-dists)/lens

        # if wandb is enabled, log prediction_str and gt_str in a table, and log the ld
        if wandb.run is not None:
            wandb.log({
                f"{self.name}_ldgen": global_ld,
                f"text_table/{self.name}": text_table
                }, commit=False)

if __name__ == "__main__":
    pass