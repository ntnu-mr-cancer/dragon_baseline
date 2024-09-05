#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import inspect
from itertools import product
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel,
                          AutoModelForTokenClassification, PretrainedConfig,
                          PreTrainedModel)
from transformers.modeling_outputs import SequenceClassifierOutput

# For documentation on sharing custom models, plaese see:
# https://huggingface.co/docs/transformers/custom_models#registering-a-model-with-custom-code-to-the-auto-classes

def generate_label_to_id_dict(num_bits):
    # Generate all combinations of True/False for the given number of bits.
    keys = product([True, False], repeat=num_bits)

    # Initialize an empty dictionary.
    label_to_id = {}

    # Populate the dictionary.
    for key in keys:
        # Convert the True/False combination to a binary string, with True as '1' and False as '0'.
        binary_str = ''.join('1' if bit else '0' for bit in key)

        # Convert the binary string to an integer.
        binary_int = int(binary_str, 2)

        # Add to the dictionary.
        label_to_id[key] = binary_int

    return label_to_id


def generate_id_to_label_dict(num_bits):
    label_to_id = generate_label_to_id_dict(num_bits)
    id_to_label = {v: k for k, v in label_to_id.items()}
    return id_to_label


def decode_labels(labels, id2label: dict, num_labels: int):
    # decode the labels to the binary vectors, while keeping -100 to ignore the loss computation
    decoded_labels = []

    # Apply the mapping to each element
    for label in labels:
        decoded_label = []
        for lbl in label:
            if lbl == -100:
                decoded_label.append([-100] * num_labels)
            else:
                if isinstance(lbl, torch.Tensor):
                    lbl = lbl.item()
                decoded_label.append(id2label[lbl])
        decoded_labels.append(decoded_label)

    return decoded_labels


class AutoModelForMultiHeadTokenClassificationConfig(PretrainedConfig):
    model_type = "AutoModelForMultiHeadTokenClassification"

    def __init__(self, pretrained_model_name_or_path: Union[Path, str] = "bert-base-multilingual-cased", num_labels: int = 5, **kwargs):
        # don't change the order of the lines below
        super().__init__(num_labels=num_labels, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.real_num_labels = num_labels


class AutoModelForMultiHeadTokenClassification(PreTrainedModel):
    config_class = AutoModelForMultiHeadTokenClassificationConfig
    supports_gradient_checkpointing = True

    """
    This class is a wrapper around the AutoModelForTokenClassification class
    that allows for multi-head binary named entity recognition..
    """

    def __init__(self, config: AutoModelForMultiHeadTokenClassificationConfig, *args, **kwargs):
        super().__init__(config)

        # set up pretrained model
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.pretrained_model_name_or_path,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True,
            *args, **kwargs
        )

        self.post_init()

        self.id2label = generate_id_to_label_dict(config.num_labels)


    # adapted from https://github.com/huggingface/transformers/blob/faed2ca46fb163082d154aa234fd5d30682d6bf1/src/transformers/models/bert/modeling_bert.py#L1541
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the multi-head sequence regression loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create a dictionary of all potential arguments
        kwargs = {
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'head_mask': head_mask,
            'inputs_embeds': inputs_embeds,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
        }

        # Get the list of parameter names accepted by the model method
        accepted_params = inspect.signature(self.model.forward).parameters.keys()

        # Filter the arguments to only those accepted by the model method
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
        outputs: SequenceClassifierOutput = self.model(
            input_ids,
            **filtered_kwargs,
        )

        # logits has shape (1, batch_size, sequence_length, num_labels)
        logits = outputs.logits

        loss = None
        if labels is not None:
            decoded_labels = decode_labels(labels=labels, id2label=self.id2label, num_labels=self.config.real_num_labels)
            decoded_labels = torch.tensor(decoded_labels, dtype=torch.float32).to(logits.device)

            weight = (decoded_labels != -100).to(logits.device).float()
            loss_fct = nn.BCEWithLogitsLoss(weight=weight)
            loss = loss_fct(logits, decoded_labels)

        if not return_dict:
            output = logits + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# register your config and your model
AutoConfig.register("AutoModelForMultiHeadTokenClassification", AutoModelForMultiHeadTokenClassificationConfig)
AutoModel.register(AutoModelForMultiHeadTokenClassificationConfig, AutoModelForMultiHeadTokenClassification)

if __name__ == "__main__":
    # debug
    config = AutoModelForMultiHeadTokenClassificationConfig(pretrained_model_name_or_path="bert-base-multilingual-cased", num_labels=3)
    model = AutoModelForMultiHeadTokenClassification(config)
    print(model)
    model.save_pretrained("test-workdir/model_save_load_test")
    model_reloaded = AutoModelForMultiHeadTokenClassification.from_pretrained("test-workdir/model_save_load_test")
    print(model_reloaded)
    print("Model successfully saved and reloaded.")
