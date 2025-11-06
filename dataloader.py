import random
from typing import List, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharTokenizer, SubwordTokenizer

class ASRDataset(Dataset):
    def __init__(
        self,
        wav_paths: List[str],
        transcripts: List[str],
        wav_lengths: List[float],
        tokenizer: Union["CharTokenizer", "SubwordTokenizer"],
        max_wav_length: float = 20.0,
        shuffle: bool = True,
    ):
        """
        wav_paths: list of paths to wav files
        transcripts: list of texts
        wav_lengths: list of lengths (in seconds) of wav files
        tokenizer: tokenizer to convert text to tokens
        """
        assert len(wav_paths) == len(transcripts) == len(wav_lengths)
        self.sr = 16000
        self.tokenizer = tokenizer
        self.sos = tokenizer.sos_id
        self.eos = tokenizer.eos_id

        # 过滤太长的样本
        self.samples = [
            (wav_paths[i], transcripts[i], wav_lengths[i])
            for i in range(len(wav_paths))
            if wav_lengths[i] <= max_wav_length
        ]
        self.is_shuffle = shuffle
        print(f"ASRDataset: {len(self.samples)} valid samples (max {max_wav_length}s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, transcript, length = self.samples[index]

        # 1️读取语音并提取 fbank 特征
        wav, sr = torchaudio.load(path)
        assert sr == self.sr, f"sample rate mismatch: {sr} != {self.sr}"
        wav = wav * (1 << 15)  # rescale to int16 for kaldi compatibility
        fbank_feat = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80)  # (seq_len, 80)

        # 2️转为 token 序列
        tokens = self.tokenizer.tokenize(transcript)
        ys_in = [self.sos] + tokens
        ys_out = tokens + [self.eos]

        # 3️返回单样本字典
        return {
            "input_values": fbank_feat,             # (seq_len, 80)
            "feat_lens": fbank_feat.shape[0],       # 单个样本长度
            "decoder_input_ids": torch.tensor(ys_in, dtype=torch.long),
            "labels": torch.tensor(ys_out, dtype=torch.long),
        }



def asr_data_collator(batch):
    """
    batch: list of dicts from ASRDataset.__getitem__
    """
    batch_size = len(batch)
    # pad fbank
    max_feat_len = max([b["input_values"].shape[0] for b in batch])
    feat_dim = batch[0]["input_values"].shape[1]
    fbank_feat = torch.zeros(batch_size, max_feat_len, feat_dim)
    feat_lens = torch.zeros(batch_size, dtype=torch.long)
    for i, b in enumerate(batch):
        feat_len = b["input_values"].shape[0]
        fbank_feat[i, :feat_len] = b["input_values"]
        feat_lens[i] = feat_len

    # pad tokens
    pad_token_for_in = batch[0]["labels"].new_full((1,), batch[0]["labels"][-1].item()).item()  # eos_id
    pad_token_for_out = -1
    max_token_len = max(len(b["decoder_input_ids"]) for b in batch)
    ys_in_pad = torch.full((batch_size, max_token_len), pad_token_for_in, dtype=torch.long)
    ys_out_pad = torch.full((batch_size, max_token_len), pad_token_for_out, dtype=torch.long)

    for i, b in enumerate(batch):
        in_len = len(b["decoder_input_ids"])
        ys_in_pad[i, :in_len] = b["decoder_input_ids"]
        ys_out_pad[i, :in_len] = b["labels"]

    return {
        "input_values": fbank_feat,
        "feat_lens": feat_lens,
        "decoder_input_ids": ys_in_pad,
        "labels": ys_out_pad,
    }
