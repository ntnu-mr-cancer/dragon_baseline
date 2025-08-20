from typing import Dict, List, Optional, Union
from transformers import Trainer
from transformers.trainer_utils import speed_metrics
from torch.utils.data import Dataset
import time
import math
from dragon_eval import DragonEval
import numpy as np
import json
import pandas as pd
from pathlib import Path
from dragon_baseline.nlp_algorithm import ProblemType

class DragonTrainer(Trainer):
    """DragonTrainer is a custom Trainer class that extends the default Trainer from Hugging Face Transformers.
    It allows for custom learning rate scheduling and evaluation using dragon eval to be performed as part of the training
    loop."""
    def __init__(
        self,
        dragon_baseline,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_loss_func=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        optimizer_cls_and_kwargs=None,
        preprocess_logits_for_metrics=None,
    ):
        # You can use or process your custom args before calling super().__init__
        # self.my_custom_arg = my_custom_arg
        self.dragon_baseline=dragon_baseline

        # Pass everything else to the parent constructor
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        
        eval_predictions = self.dragon_baseline.predict(df=self.dragon_baseline.df_val if self.dragon_baseline.data_args.max_eval_samples is None else self.dragon_baseline.df_val.sample(self.dragon_baseline.data_args.max_eval_samples))
        
        hot_eval = HotEval(dragon_baseline=self.dragon_baseline, eval_predictions=eval_predictions)
        dragon_eval_results = float(hot_eval.evaluate())
        output.metrics.update({
            f"{metric_key_prefix}_dragon": dragon_eval_results,
        })
        print(f"Dragon eval results: {dragon_eval_results}")
        
        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

class HotEval(DragonEval):
    '''Evaluate the model using the DragonEval framework to get the most
    appropriate eval metrics to use for model selection.'''
    def __init__(self, dragon_baseline, eval_predictions, **kwargs):
        self._scores: Dict[str, Dict[str, float] ] = {}
        self._join_key = "uid"
        self._predictions_cases = eval_predictions.copy()
        self.task = dragon_baseline.task.task_name
        self.max_predictions = dragon_baseline.data_args.max_eval_samples

        with open(dragon_baseline.dataset_val_path, 'r') as f:
            self._ground_truth_cases = pd.DataFrame(json.load(f))
        label_column = [col for col in self._ground_truth_cases.columns if col.endswith("_target")][0]

        # Due to pandas automtic casting of dtypes strings/categories are sometimes converted to integers, which can cause issues with DragonEval.
        # In this case we ensure that the target and prediction columns are of the same dtype by explicitely casting.
        dtype_mismatch = self._ground_truth_cases[label_column].dtype != self._predictions_cases[label_column.replace('_target', '')].dtype
        gt_is_integer = np.issubdtype(self._ground_truth_cases[label_column].dtype, np.integer)
        if dtype_mismatch and gt_is_integer:
            self._predictions_cases[label_column.replace('_target', '')] = self._predictions_cases[label_column.replace('_target', '')].astype(self._ground_truth_cases[label_column].dtype)
        
        # self._ground_truth_cases = model.dragon_baseline.df_val
        # self.model = model
    
    def evaluate(self):
        self.merge_ground_truth_and_predictions()
        self.cross_validate(ignore_missing=self.max_predictions is not None)
        self.score(task_name=self.task, job_name='fed_eval')
        return self._scores[self.task]['fed_eval']
    
    def save(self, predictions: pd.DataFrame, path = None):
        """Save the predictions."""
        if path is None:
            path = self.test_predictions_path
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_json(path, orient="records")

    def verify_predictions(self, dataset_test_path = None):
        if dataset_test_path is not None:
            self.dataset_test_path = dataset_test_path
        super().verify_predictions()
    
    
    def cross_validate(self, ignore_missing=False):
        if self.max_predictions is None:
            missing = [
                p for _, p in self._cases.iterrows() if p["_merge"] == "left_only"
            ]

            if missing:
                if self._join_key:
                    missing = [p[self._join_key] for p in missing]
                self._raise_missing_predictions_error(missing=missing)
        else:
            # If max_predictions is set, we ignore missing predictions and do a simple count check
            assert self._cases._merge.value_counts().both == self.max_predictions, ValueError("Number of predictions does not match max_eval_samples")
            self._cases = self._cases[self._cases["_merge"] == "both"]

        extra = [
            p for _, p in self._cases.iterrows() if p["_merge"] == "right_only"
        ]

        if extra:
            if self._join_key:
                extra = [p[self._join_key] for p in extra]
            self._raise_extra_predictions_error(extra=extra)
        