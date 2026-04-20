from datasets import Dataset
import os
import torch
import numpy as np
from scipy.stats import expon
import pandas as pd

from pathlib import Path
from configs.enums import data_paths


def load_music_dataset(split="train", cfg=None):
    dict_ds = {'input_ids': []}
    test_min = []
    test_max = []

    dataset = cfg.data.train
    tokenizer = cfg.model.tokenizer

    if dataset == "Maestro":
        print("Loading Maestro dataset...")

        # Maestro paths
        csv_path = data_paths.MAESTRO_CSV.value

        if tokenizer == "wavtokenizer":
            tokens_dir = data_paths.MAESTRO_TOKENS_WAVTOKENIZER.value
        elif tokenizer == "unicodec":
            tokens_dir = data_paths.MAESTRO_TOKENS_UNICODEC.value
        else:
            raise ValueError(f"Unknown tokenizer '{tokenizer}'")

        # Load CSV metadata
        df = pd.read_csv(csv_path)

        # Filter for split
        split_df = df[df["split"] == split]


        for _, row in split_df.iterrows():
            token_file = row["audio_filename"].replace(".wav", ".pt")
            base_name = Path(token_file).stem  # basename without suffix
            # extract the year folder (first path component of the CSV audio filename)
            year = Path(token_file).parts[0]
            year_dir = Path(tokens_dir) / year

            print(f"Processing base name: {base_name} in year folder: `{year_dir}`")

            if not year_dir.exists():
                print(f"Warning: year folder `{year_dir}` does not exist")
                continue

            # find all .pt files under the specific year folder whose filename starts with the base_name
            matches = sorted(str(p) for p in year_dir.rglob("*.pt") if p.name.startswith(base_name))
            print(f"Found {len(matches)} files for {base_name} in `{year_dir}`")

            if matches:
                for token_path in matches:
                    try:
                        d = torch.squeeze(torch.load(token_path))
                        dict_ds['input_ids'].append(d)
                        test_min.append(torch.min(d).item())
                        test_max.append(torch.max(d).item())
                    except Exception as e:
                        print(f"Warning: Failed to load {token_path}: {e}")
            else:
                print(f"Warning: No token files found for base `{base_name}` in `{year_dir}`")

        sliced_dict = dict_ds
        print(f"Loaded {len(sliced_dict['input_ids'])} samples for split '{split}'")

    elif dataset=="MusicNet":

        if tokenizer == "wavtokenizer":
            path = data_paths.MUSICNET_WAVTOKENIZER.value
        elif tokenizer == "unicodec":
            path = data_paths.MUSICNET_UNICODEC.value
        else:
            raise ValueError(f"Unknown tokenizer '{tokenizer}'")

        for file in sorted(os.listdir(path)):  # sorted for reproducibility
            d = torch.squeeze(torch.load(os.path.join(path, file)))
            dict_ds['input_ids'].append(d)
            test_min.append(torch.min(d).item())
            test_max.append(torch.max(d).item())

        # Get split index
        total_len = len(dict_ds['input_ids'])
        split_idx = int(0.8 * total_len)

        if split == "train":
            sliced_dict = {'input_ids': dict_ds['input_ids'][:split_idx]}
        elif split == "validation":
            sliced_dict = {'input_ids': dict_ds['input_ids'][split_idx:]}
        else:
            raise ValueError(f"Invalid split name '{split}', must be 'train' or 'validation'.")


    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    return Dataset.from_dict(sliced_dict)


