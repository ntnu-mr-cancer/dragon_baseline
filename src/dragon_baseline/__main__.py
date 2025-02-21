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

import sys

from dragon_baseline.run_classification import main as parse_and_run_classification
from dragon_baseline.run_classification_multi_label import \
    main as parse_and_run_classification_multi_label
from dragon_baseline.run_ner import main as parse_and_run_ner

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("First argument must specify the problem type")

    problem_type = sys.argv.pop(1)
    if problem_type == "ner":
        parse_and_run_ner()
    elif problem_type == "classification":
        parse_and_run_classification()
    elif problem_type == "multi_label_classification":
        parse_and_run_classification_multi_label()
    else:
        raise ValueError(f"Unexpected problem type: {problem_type}.")
