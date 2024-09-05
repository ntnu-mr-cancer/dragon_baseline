# DRAGON dataset convention

The DRAGON benchmark aims to catalyze the development of algorithms capable of addressing a broad spectrum of data curation tasks and introduces 28 clinically relevant tasks, as detailed [here](https://dragon.grand-challenge.org/tasks/). To facilitate this, eight distinct task types were identified and defined. These task types were designed to be universally applicable, and most data curation tasks should be able to be formulated as such a task:

1.	single-label binary classification (e.g., classify reports as indicating "cancer" or "no cancer"), 
2.	single-label multi-class classification (e.g., classify reports based on the type of diagnosis such as "cancer", "other disease", or "benign"), 
3.	multi-label binary classification (e.g., classify reports based on multiple symptoms present, such as "hyperplastic polyps", "high-grade dysplasia", and “cancer”, where each symptom is treated as a binary indicator), 
4.	multi-label multi-class classification (e.g., classify reports by multiple factors, such as disease severity "mild", "moderate", "or severe" and urgency of treatment "low", "medium", "or high"), 
5.	single-label regression (e.g., predict the prostate volume described in the report), 
6.	multi-label regression (e.g., predict the lesion diameter of multiple lesions), 
7.	single-label named entity recognition (e.g., identify and classify protected health information in a report, such as names, dates, and places), and 
8.	multi-label named entity recognition (e.g., identify multiple types of entities in a medical report, where entities can be diseases, symptoms, treatments, and test results, potentially with overlap and multiple occurrences).

## Data format
The DRAGON benchmark and baseline algorithm relies on the JSON data format. For each task (in the benchmark, a synthetic task or one defined by you), the following **input** files are needed:

1. Dataset configurations (`nlp-task-configuration.json`)
2. Training data (`nlp-training-dataset.json`)
3. Validation data (`nlp-validation-dataset.json`)
4. Test data (`nlp-test-dataset.json`)

The algorithm should then **output** predictions for the test set:

5. Test predictions (`nlp-predictions-dataset.json`)

## Input
### Dataset configurations
The configuration file contains the:
* `jobid`: Integer. Used as random seed.
* `task_name`: String. For displaying only.
* `input_name`: String. Name of the field in the train/validation/test data where to find the input for the algorithm. Should be "text" when the field contains a string, or "text_parts" when the field contains an array of strings.
* `label_name`: String. Name of the field in the train/validation/test data where to find the label. This reflects the task type that is chosen (see the task types at the start). Should be one of:
  * `single_label_multi_class_classification_target`
  * `single_label_regression_target`
  * `multi_label_regression_target`
  * `named_entity_recognition_target`
  * `multi_label_multi_class_classification_target`
  * `single_label_binary_classification_target`
  * `multi_label_binary_classification_target`
  * `multi_label_named_entity_recognition_target`
* `recommended_truncation_side`: String. Indicate which side of the input to truncate (if needed). Should be "left" or "right".
* `version`: String. Version of the configuration file format.

An example is given below:

```json
{
  "jobid": 1010,
  "task_name": "Task101_Example_sl_bin_clf",
  "input_name": "text",
  "label_name": "single_label_binary_classification_target",
  "recommended_truncation_side": "left",
  "version": "1.0"
}
```

For an example for each task type, see the `nlp-task-configuration.json` files in [`test-input`](/test-input).

### Training data
All cases to be used for training should be provided in the `nlp-training-dataset.json` file. Each case consists of a unique identifier, the algorithm input (e.g., clinical report), and the associated labels. Specifically, the following elements should be provided for each case:

* `uid`: String. A unique identifier of the case. May not overlap with the validation or test cases.

One of:
* `text`: String. The input for the algorithm. This is the typical way to provide the input.
* `text_parts`: Array of strings. The input for the algorithm where the input contains multiple parts. This is used for tasks like named entity recognition.

One of:
* `single_label_binary_classification_target`: Boolean. Label to predict.
* `single_label_multi_class_classification_target`: String. Label to predict.
* `multi_label_binary_classification_target`: Array of booleans. Label to predict.
* `multi_label_multi_class_classification_target`: Array of strings. Label to predict.
* `single_label_regression_target`: Number. Label to predict.
* `multi_label_regression_target`: Array of numbers. Label to predict.
* `named_entity_recognition_target`: Array of strings. Label to predict (one label per word).
* `multi_label_named_entity_recognition_target`: Array of arrays of strings. Label to predict.

An example is given below:

```json
[
  {
    "uid":"Task101_case1",
    "text":"The text goes here",
    "single_label_binary_classification_target":true
  },
  {
    "uid":"Task101_case2",
    "text":"The text goes here",
    "single_label_binary_classification_target":false
  },
]
```

For an example for each task type, see the `nlp-training-dataset.json` files in [`test-input`](/test-input).


### Validation data
All cases to be used for model selection should be provided in the `nlp-validation-dataset.json` file. The contents for each case is the same as for the training data. 

For an example for each task type, see the `nlp-validation-dataset.json` files in [`test-input`](/test-input).


### Test data
All cases to be predicted by the model should be provided in the `nlp-test-dataset.json` file. Each case consists of a unique identifier and the algorithm input (e.g., clinical report). Specifically, the following elements should be provided for each case:

* `uid`: String. A unique identifier of the case. May not overlap with the validation or test cases.

One of:
* `text`: String. The input for the algorithm. This is the typical way to provide the input.
* `text_parts`: Array of strings. The input for the algorithm where the input contains multiple parts. This is used for tasks like named entity recognition.

For an example for each task type, see the `nlp-test-dataset.json` files in [`test-input`](/test-input).


## Output
### Test predictions
Based on the provided test data, the algorithm should provide predictions and save the predictions to the `nlp-predictions-dataset.json` file. For each case, the unique identifier from the test data should be provided alongside the model prediction. The model prediction should be provided in the field with the same name as the training/validation labels, without the `_target` at the end. Specifically, the following elements should be provided for each case:

* `uid`: String. A unique identifier of the case. From the test data.

One of:
* `single_label_binary_classification`: Number between 0 and 1. Model prediction.
* `single_label_multi_class_classification`: String. Model prediction.
* `multi_label_binary_classification`: Array of numbers between 0 and 1. Model prediction.
* `multi_label_multi_class_classification`: Array of strings. Model prediction.
* `single_label_regression`: Number. Model prediction.
* `multi_label_regression`: Array of numbers. Model prediction.
* `named_entity_recognition`: Array of strings. Model prediction.
* `multi_label_named_entity_recognition`: Array of arrays of strings. Model prediction.

An example is given below:

```json
[
    {"uid":"Task101_case0","single_label_binary_classification":0.8099229932},
    {"uid":"Task101_case2","single_label_binary_classification":0.9840752482}
]
```

For an example for each task type, see the `nlp-predictions-dataset.json` files in [`test-output`](/test-output).
