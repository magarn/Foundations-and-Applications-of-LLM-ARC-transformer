import os
import sys
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from tokenizer import SubwordTokenizer
from dataloader import ASRDataset, asr_data_collator
from models import Encoder, Decoder, Transformer
from feature_extractors import LinearFeatureExtractionModel, ResNet1D



# -----------------------------
# 1. 构建模型函数
# -----------------------------
def init_model(vocab_size, enc_dim, num_enc_layers, num_dec_layers, feature_extractor_type):
    fbank_dim = 80
    num_heads = enc_dim // 64
    max_seq_len = 2048

    if feature_extractor_type == "linear":
        FeatureExtractor = LinearFeatureExtractionModel
    elif feature_extractor_type == "resnet":
        FeatureExtractor = ResNet1D
    else:
        raise ValueError(f"Unsupported feature extractor type: {feature_extractor_type}")

    feature_extractor = FeatureExtractor(fbank_dim, enc_dim)

    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.0,
        num_layers=num_enc_layers, enc_dim=enc_dim, num_heads=num_heads, dff=2048, tgt_len=max_seq_len
    )

    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.0,
        num_layers=num_dec_layers, dec_dim=enc_dim, num_heads=num_heads, dff=2048, tgt_len=max_seq_len,
        tgt_vocab_size=vocab_size
    )

    model = Transformer(feature_extractor, encoder, decoder, enc_dim, vocab_size)
    return model


# -----------------------------
# 2. 自定义 Dataset 适配 Trainer
# -----------------------------
class SpeechDataset(Dataset):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get_item 输出与原 dataloader batch 内容一致
        fbank_feat, feat_lens, ys_in_pad, ys_out_pad = self.dataset[idx]
        print(fbank_feat.shape, feat_lens.shape, ys_in_pad.shape, ys_out_pad.shape)
        return {
            "input_values": fbank_feat,
            "feat_lens": feat_lens,
            "decoder_input_ids": ys_in_pad,
            "labels": ys_out_pad,
        }


# -----------------------------
# 3. 在transformers的基础上自定义Trainer
# -----------------------------
class SpeechTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        fbank_feat = inputs["input_values"].cuda()
        feat_lens = inputs["feat_lens"].cuda()
        ys_in_pad = inputs["decoder_input_ids"].cuda()
        ys_out_pad = inputs["labels"].cuda()

        logits = model(fbank_feat, feat_lens, ys_in_pad)
        logits = logits.view(-1, logits.size(-1))
        labels = ys_out_pad.view(-1).long()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
        loss = loss_fct(logits, labels)

        return (loss, logits) if return_outputs else loss

class TorchDataLoaderWrapper(torch.utils.data.Dataset):
    def __init__(self, dataloader):
        self.batches = list(iter(dataloader))  # 每个元素本身是一个 batch

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        fbank_feat, feat_lens, ys_in_pad, ys_out_pad = self.batches[idx]
        return {
            "input_values": fbank_feat,
            "input_lengths": feat_lens,
            "labels": ys_out_pad
        }


# -----------------------------
# 4. 主函数入口
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_with_trainer.py <feature_extractor_type>")
        sys.exit(1)

    feature_extractor_type = sys.argv[1]
    dataset_type = "miniLibriSpeech"
    assert feature_extractor_type in ["linear", "resnet"]
    # feature_extractor_type = 'resnet'

    t_ph = "./spm/minilibrispeech/1000_bpe.model"
    audio_path_file = "./data/miniLibriSpeech/train-clean-5.paths"
    text_file = "./data/miniLibriSpeech/train-clean-5.text"
    lengths_file = "./data/miniLibriSpeech/train-clean-5.lengths"

    # tokenizer
    tokenizer = SubwordTokenizer(t_ph)
    vocab = tokenizer.vocab

    # 数据加载
    with open(audio_path_file, 'r') as f:
        audio_paths = f.read().splitlines()
    with open(text_file, 'r') as f:
        transcripts = f.read().splitlines()
    with open(lengths_file, 'r') as f:
        wav_lengths = [float(x) for x in f.read().splitlines()]

    # dataloader
    batch_size = 16
    batch_seconds = 512

    train_dataset = ASRDataset(audio_paths, transcripts, wav_lengths, tokenizer, batch_seconds, shuffle=True)


    # model
    enc_dim = 256
    num_enc_layers = 12
    num_dec_layers = 6
    model = init_model(vocab, enc_dim, num_enc_layers, num_dec_layers, feature_extractor_type)
    if torch.cuda.is_available():
        model.cuda()

    # checkpoint dir
    ckpt_dir = f"./ckpts/checkpoints_{feature_extractor_type}_{dataset_type}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # -----------------------------
    # 5. TrainingArguments & Trainer
    # -----------------------------
    args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=batch_size,   # 已在 dataloader 控制 batch
        num_train_epochs=50,
        learning_rate=5e-5,
        warmup_steps=600,
        logging_dir=f"./logs/tensorboard/{feature_extractor_type}",
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        report_to="tensorboard",
        dataloader_drop_last=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = SpeechTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=asr_data_collator
    )

    # -----------------------------
    # 6. 开始训练
    # -----------------------------
    trainer.train()
    trainer.save_model(ckpt_dir)
