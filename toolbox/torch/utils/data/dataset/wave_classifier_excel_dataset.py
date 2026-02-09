#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from toolbox.torch.utils.data.vocabulary import Vocabulary


class WaveClassifierExcelDataset(Dataset):
    def __init__(self,
                 vocab: Vocabulary,
                 excel_file: str,
                 expected_sample_rate: int,
                 resample: bool = False,
                 root_path: str = None,
                 category: str = None,
                 category_field: str = "category",
                 label_field: str = "labels",
                 max_wave_value: float = 1.0,
                 ) -> None:
        self.vocab = vocab
        self.excel_file = excel_file

        self.expected_sample_rate = expected_sample_rate
        self.resample = resample
        self.root_path = root_path
        self.category = category
        self.category_field = category_field
        self.label_field = label_field
        self.max_wave_value = max_wave_value

        df = pd.read_excel(excel_file)

        samples = list()
        for i, row in tqdm(df.iterrows(), total=len(df)):
            filename = row["filename"]
            label = row[self.label_field]

            if self.category is not None and self.category != row[self.category_field]:
                continue

            samples.append({
                "filename": filename,
                "label": label,
            })
        self.samples = samples

    def __getitem__(self, index):
        sample = self.samples[index]
        filename = sample["filename"]
        label = sample["label"]

        if self.root_path is not None:
            filename = os.path.join(self.root_path, filename)

        waveform = self.filename_to_waveform(filename)

        namespace = self.label_field if self.category is None else self.category
        token_to_index = self.vocab.get_token_to_index_vocabulary(namespace=namespace)
        label: int = token_to_index[label]

        result = {
            "waveform": waveform,
            "label": torch.tensor(label, dtype=torch.int64),
        }
        return result

    def __len__(self):
        return len(self.samples)

    def filename_to_waveform(self, filename: str):
        try:
            if self.resample:
                waveform, sample_rate = librosa.load(filename, sr=self.expected_sample_rate)
                # waveform, sample_rate = torchaudio.load(filename, normalize=True)
            else:
                sample_rate, waveform = wavfile.read(filename)
                waveform = waveform / self.max_wave_value
        except ValueError as e:
            print(filename)
            raise e
        if sample_rate != self.expected_sample_rate:
            raise AssertionError

        waveform = torch.tensor(waveform, dtype=torch.float32)
        return waveform


if __name__ == "__main__":
    pass
