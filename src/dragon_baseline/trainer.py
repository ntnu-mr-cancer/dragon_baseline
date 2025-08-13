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

        # Move eval ground truths into the folder structure expected by dragon eval
        # task = self.dragon_baseline.task.task_name
        # task_and_fold = self.dragon_baseline.workdir.name

        # eval_gt_dir = Path(self.dragon_baseline.workdir).parent / Path("eval_gt")
        # eval_gt_dir.mkdir(parents=True, exist_ok=True)
        # val_gt_path = self.dragon_baseline.dataset_val_path

        # eval_predictions_dir = Path(self.dragon_baseline.workdir).parent / Path("eval_predictions") / task_and_fold
        # eval_predictions_dir.mkdir(parents=True, exist_ok=True)

        # with open(val_gt_path, 'r') as f:
        #     val_gt_path = json.load(f)
        #     json.dump(val_gt_path, open(eval_gt_dir / Path(task + '.json'), 'w'))

        # self.dragon_baseline.save(eval_predictions, test_predictions_path=eval_predictions_dir / Path("nlp-predictions-dataset.json"))
        
        # dragon_eval = DragonEval(
        #     ground_truth_path= eval_gt_dir,
        #     predictions_path= eval_predictions_dir.parent,
        #     folds=[int(task_and_fold.split('-')[-1].replace('fold', ''))],
        #     tasks=[task.split('_')[0].replace('Task', '')],
        #     output_file=self.dragon_baseline.workdir / Path("dragon_eval_val_res.json"),
        # )
        # dragon_eval.evaluate()
        # output.metrics.update({
        #     f"{metric_key_prefix}_dragon_cold": dragon_eval._scores[task][task_and_fold],
        # })
        
        output.metrics.update({
            f"{metric_key_prefix}_dragon_hotter": self.dragon_eval(output.label_ids, output.predictions),
        })
        
        self.log(output.metrics)

        import ipdb; ipdb.set_trace()
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def dragon_eval(self, targets, raw_predictions):
        from dragon_eval.evaluation import EvalType, score_rsmape, score_multi_label_f1, TASK_TYPE, REGRESSION_EPSILON
        from sklearn.metrics import cohen_kappa_score, roc_auc_score
        from scipy.special import softmax, expit
        from seqeval.metrics import f1_score
        task_name = self.dragon_baseline.task.task_name 
        
        def prepare_labels_and_predictions(targets, raw_predictions):
            problem_type = self.dragon_baseline.task.target.problem_type
            id2label = self.model.config.id2label

            if problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
                y_pred = softmax(raw_predictions, axis=1)[:, 1]  # probability of the "positive" class
                y_true = targets.astype(int)

            elif problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
                y_pred = np.vectorize(id2label.get)(np.argmax(raw_predictions, axis=1)).astype(int)
                y_true = targets.astype(int)

            elif problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
                y_pred = expit(raw_predictions).reshape(-1)  # calculate sigmoid to map the logits to [0, 1]
                y_true = targets.astype(int).reshape(-1)

            elif  problem_type == ProblemType.SINGLE_LABEL_REGRESSION or problem_type == ProblemType.MULTI_LABEL_REGRESSION:
                scaler = self.dragon_baseline.label_scalers[self.dragon_baseline.task.target.label_name]
                y_pred = scaler.inverse_transform(raw_predictions.reshape(-1, 1)).reshape(-1).astype(float)
                y_true = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(-1).astype(float)
            elif problem_type == ProblemType.SINGLE_LABEL_NER or problem_type == ProblemType.MULTI_LABEL_NER:
                y_pred = np.vectorize(id2label.get)(np.argmax(raw_predictions, axis=2)).tolist()  # Assuming raw_predictions is of shape (batch_size, seq_length, num_labels)
                y_true = np.vectorize(id2label.get)(np.where(targets != -100, targets, self.model.config.label2id['O'])).tolist()  # Assuming targets is of shape (batch_size, seq_length) and ignore class presumed to be 17 is mapped to -100 during training.
                import ipdb; ipdb.set_trace()
            else:
                raise ValueError(f"Unexpected problem type '{self.task.target.problem_type}'")

            return y_true, y_pred

        y_true, y_pred = prepare_labels_and_predictions(targets, raw_predictions)

        if TASK_TYPE[task_name] == EvalType.BINARY_CLASSIFICATION:
            # evaluate (multi-label) binary classification tasks
            # note: each subtask is the same, so we pool the labels and predictions
            # metric: AUC
            score = roc_auc_score(y_true=y_true, y_score=y_pred)

        elif TASK_TYPE[task_name] == EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION:
            # evaluate ordinal multi-class classification tasks
            # metric: Linear-weighted Cohen's kappa
            score = cohen_kappa_score(
                y1=y_true,
                y2=y_pred,
                weights="linear",
            )

        elif TASK_TYPE[task_name] == EvalType.REGRESSION:
            # evaluate regression tasks
            # note: for the multi-label regression task, each subtask is the same,
            #       so we pool the labels and predictions
            # metric: R-SMAPE
            epsilon = REGRESSION_EPSILON[task_name]

            score = score_rsmape(
                y_true=y_true,
                y_pred=y_pred,
                epsilon=epsilon,
                ignore_missing_targets=True,
            )

        elif TASK_TYPE[task_name] == EvalType.SINGLE_LABEL_NER:
            # evaluate single-label named entity recognition tasks
            # metric: F1 score
            score = f1_score(
                y_true=y_true,
                y_pred=y_pred,
                average="macro",
            )

        # elif TASK_TYPE[task_name] == EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION:
        #     # evaluate ordinal multi-class classification tasks
        #     # metric: Linear-weighted Cohen's kappa
            
        #     score = cohen_kappa_score(
        #         y1=y_true,
        #         y2=y_pred,
        #         weights="linear",
        #     )

        # elif TASK_TYPE[task_name] == EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION:
        #     # evaluate non-ordinal (multi-class) classification tasks
        #     # note: each subtask is the same, so we pool the labels and predictions
        #     #       (this is not actually true for the example task, but it is for the real tasks)
        #     # metric: Unweighted Cohen's kappa
        #     score = cohen_kappa_score(
        #         y1=y_true.explode(),
        #         y2=y_pred.explode(),
        #         weights=None,
        #     )


        # elif TASK_TYPE[task_name] == EvalType.BINARY_CLASSIFICATION_NON_SHARED_TASK:
        #     # evaluate binary classification tasks with different objectives across labels
        #     # metric: mean AUC per objective
        #     score = np.mean([
        #         roc_auc_score(
        #             y_true=y_true.apply(lambda values: values[i]),
        #             y_score=y_pred.apply(lambda values: values[i]),
        #         )
        #         for i in range(len(y_true.iloc[0]))
        #     ])

        # elif TASK_TYPE[task_name] == EvalType.REGRESSION:
        #     # evaluate regression tasks
        #     # note: for the multi-label regression task, each subtask is the same,
        #     #       so we pool the labels and predictions
        #     # metric: R-SMAPE
        #     epsilon = REGRESSION_EPSILON[task_name]

        #     score = score_rsmape(
        #         y_true=y_true.explode().astype(float),
        #         y_pred=y_pred.explode().astype(float),
        #         epsilon=epsilon,
        #         ignore_missing_targets=True,
        #     )

        # elif TASK_TYPE[task_name] == EvalType.SINGLE_LABEL_NER:
        #     # evaluate single-label named entity recognition tasks
        #     # metric: F1 score
        #     score = seqeval.metrics.f1_score(
        #         y_true=y_true,
        #         y_pred=y_pred,
        #         average="macro",
        #     )

        # elif TASK_TYPE[task_name] == EvalType.MULTI_LABEL_NER:
        #     # evaluate multi-label named entity recognition tasks
        #     # metric: weighted F1 score
        #     score = score_multi_label_f1(
        #         y_true=y_true,
        #         y_pred=y_pred,
        #         average="weighted",
        #     )

        else:
            raise ValueError(f"Unexpexted task: {task_name}")

        return score

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
        