import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import PreTrainedTokenizerBase, PaddingStrategy

@dataclass
class DataCollatorCTCWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 拆分输入和标签，因为它们长度不同
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 对输入部分进行 padding
        input_values = [torch.tensor(f["input_values"]) for f in input_features]
        input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)

        # 对标签进行 padding
        label_values = [torch.tensor(f["input_ids"]) for f in label_features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(label_values, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # 将 padding 的位置标记为 -100（CTC 训练时忽略这些位置）
        labels_padded = labels_padded.masked_fill(labels_padded == self.tokenizer.pad_token_id, -100)

        batch = {
            "input_values": input_values_padded,
            "labels": labels_padded,
        }

        return batch
