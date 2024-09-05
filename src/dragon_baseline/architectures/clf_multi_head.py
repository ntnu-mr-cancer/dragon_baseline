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
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel,
                          AutoModelForSequenceClassification, PretrainedConfig,
                          PreTrainedModel)
from transformers.modeling_outputs import SequenceClassifierOutput


class AutoModelForMultiHeadSequenceClassificationConfig(PretrainedConfig):
    model_type = "AutoModelForMultiHeadSequenceClassification"

    def __init__(self, pretrained_model_name_or_path: Union[Path, str] = "bert-base-multilingual-cased", num_classes_per_label: Iterable[int] = (5, 7), **kwargs):
        super().__init__(num_labels=sum(num_classes_per_label), **kwargs)
        self.num_classes_per_label = num_classes_per_label
        self.pretrained_model_name_or_path = pretrained_model_name_or_path


class AutoModelForMultiHeadSequenceClassification(PreTrainedModel):
    config_class = AutoModelForMultiHeadSequenceClassificationConfig
    supports_gradient_checkpointing = True

    """
    This class is a wrapper around the AutoModelForSequenceClassification class
    that allows for multi-head classification with two sets of labels.
    """

    def __init__(self, config: AutoModelForMultiHeadSequenceClassificationConfig, *args, **kwargs):
        super().__init__(config)

        # set up pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.pretrained_model_name_or_path,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True,
            *args, **kwargs
        )

        self.post_init()

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
        labels (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*):
            Labels for computing the sequence classification loss.
        """
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

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

        # split logits for each head
        logits = outputs.logits
        num_heads = len(self.config.num_classes_per_label)
        cumulative_indices = [0] + list(np.cumsum(self.config.num_classes_per_label))
        logits_per_head = [logits[:, cumulative_indices[i]:cumulative_indices[i+1]] for i in range(num_heads)]

        loss_fct = nn.CrossEntropyLoss()

        loss = None
        if labels is not None:
            loss = 0
            for i in range(num_heads):
                loss += loss_fct(logits_per_head[i].float(), labels[:, i].long())
            loss = loss / self.config.num_labels

        if not return_dict:
            output = logits_per_head + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits_per_head,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# register your config and your model
AutoConfig.register("AutoModelForMultiHeadSequenceClassification", AutoModelForMultiHeadSequenceClassificationConfig)
AutoModel.register(AutoModelForMultiHeadSequenceClassificationConfig, AutoModelForMultiHeadSequenceClassification)


if __name__ == "__main__":
    # debug
    config = AutoModelForMultiHeadSequenceClassificationConfig(pretrained_model_name_or_path="bert-base-multilingual-cased", num_classes_per_label=[2, 3, 4])
    model = AutoModelForMultiHeadSequenceClassification(config)
    print(model)
    model.save_pretrained("test-workdir/model_save_load_test")
    model_reloaded = AutoModelForMultiHeadSequenceClassification.from_pretrained("test-workdir/model_save_load_test")
    print(model_reloaded)
    print("Model successfully saved and reloaded.")
