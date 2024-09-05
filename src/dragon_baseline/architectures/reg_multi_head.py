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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel,
                          AutoModelForSequenceClassification, PretrainedConfig,
                          PreTrainedModel)
from transformers.modeling_outputs import SequenceClassifierOutput

# For documentation on sharing custom models, plaese see:
# https://huggingface.co/docs/transformers/custom_models#registering-a-model-with-custom-code-to-the-auto-classes


class AutoModelForMultiHeadSequenceRegressionConfig(PretrainedConfig):
    model_type = "AutoModelForMultiHeadSequenceRegression"

    def __init__(self, pretrained_model_name_or_path: Union[Path, str] = "bert-base-multilingual-cased", num_labels: int = 5, filter_targets: bool = True, **kwargs):
        super().__init__(num_labels=num_labels, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.filter_targets = filter_targets


class AutoModelForMultiHeadSequenceRegression(PreTrainedModel):
    config_class = AutoModelForMultiHeadSequenceRegressionConfig
    supports_gradient_checkpointing = True

    """
    This class is a wrapper around the AutoModelForSequenceClassification class
    that allows for multi-head regression with multiple sets of labels.
    """

    def __init__(self, config: AutoModelForMultiHeadSequenceRegressionConfig, *args, **kwargs):
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

        # have one logit per label
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            if self.config.filter_targets:
                mask = ~labels.isnan()
                loss = loss_fct(logits[mask], labels[mask])
            else:
                loss = loss_fct(logits, labels)

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
AutoConfig.register("AutoModelForMultiHeadSequenceRegression", AutoModelForMultiHeadSequenceRegressionConfig)
AutoModel.register(AutoModelForMultiHeadSequenceRegressionConfig, AutoModelForMultiHeadSequenceRegression)

if __name__ == "__main__":
    # debug
    config = AutoModelForMultiHeadSequenceRegressionConfig(pretrained_model_name_or_path="bert-base-multilingual-cased", num_labels=3)
    model = AutoModelForMultiHeadSequenceRegression(config)
    print(model)
    model.save_pretrained("test-workdir/model_save_load_test")
    model_reloaded = AutoModelForMultiHeadSequenceRegression.from_pretrained("test-workdir/model_save_load_test")
    print(model_reloaded)
    print("Model successfully saved and reloaded.")
