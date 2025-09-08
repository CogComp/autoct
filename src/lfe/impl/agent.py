import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from collections import defaultdict
from copy import deepcopy
from enum import StrEnum
from typing import Any, List, Literal, NamedTuple, Optional, Tuple

import dill
import dotenv
import dspy
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import seaborn as sns
import yaml
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .globals import GLOBAL_FB_CACHE_DIR, cache, get_duckdb_ro_con, get_global_pool
from .tools import (
    get_clinical_trial_info_from_clinical_trials_gov_dict,
    get_detailed_nct_info,
    make_nct_search,
    make_pubmed_search,
)
from .utils import is_notebook

dotenv.load_dotenv()

lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.getenv('OPENAI_API_KEY'),
    max_tokens=16000,
)
dspy.configure(lm=lm)


cache_dir = "./.cache/agent_script_cache_dir"

logfile_stderr = "./.logs/agentstderr"
logfile_stdout = "./.logs/agentstdout"

error_dir = "./.errs"


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def orjson_default(obj):
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    raise TypeError


def dump_as_json(obj, pretty=True):
    return orjson.dumps(
        obj, default=orjson_default, option=None if not pretty else orjson.OPT_INDENT_2
    ).decode("utf-8")


def soft_assert(value, condition: bool, message: str):
    dspy.Suggest(condition, message)

    if condition:
        return value
    else:
        print(f"Failed to satisfy condition {message}")
        return None


@cache.memoize(tag="v0")
def get_relevant_pub_med(nct_id: str) -> str:
    local_con = get_duckdb_ro_con()
    print("get_relevant_pub_med")
    return (
        local_con.sql(
            f"""
        PRAGMA disable_progress_bar;

        select title_abstract, ArticleAuthorList, ArticleJournalISO from 'pubmed_meta_df_joined_with_sim.parquet' where nctId = '{nct_id}' order by pub_sim desc limit 10
            """
        )
        .df()
        .to_json(orient="records")
    )


def get_clinical_trial_info_from_clinical_trials_gov(nct_id: str) -> str:
    return json.dumps(get_clinical_trial_info_from_clinical_trials_gov_dict(nct_id))


class Task(StrEnum):
    CLINICAL_TRIAL_SUCCESS_PRED = "Predict the outcome of a clinical trial (1=success or 0=failure) at the beginning stages of a trial."
    CLINICAL_TRIAL_SUCCESS_PRED_PHASE_1 = "Predict the outcome of a phase 1 clinical trial (1=success or 0=failure) at the beginning stages of a trial."
    CLINICAL_TRIAL_SUCCESS_PRED_PHASE_2 = "Predict the outcome of a phase 2 clinical trial (1=success or 0=failure) at the beginning stages of a trial."
    CLINICAL_TRIAL_SUCCESS_PRED_PHASE_3 = "Predict the outcome of a phase 3 clinical trial (1=success or 0=failure) at the beginning stages of a trial."
    CLINICAL_TRIAL_SUCCESS_PRED_PHASE_4 = "Predict the outcome of a phase 4 clinical trial (1=success or 0=failure) at the beginning stages of a trial."
    SERIOUS_ADVERSE_EVENT_PHASE_1 = "Predict whether a phase 1 clinical trial will have a Serious Adverse Event (SAE) before the trial begins (1=SAE present, 0=no SAE) "
    PATIENT_DROPOUT_PHASE_1 = "Predict whether a phase 1 clinical trial will have a non-trivial patient dropout during the trial design phase (1=dropout, 0=no dropout) "
    MORTALITY_PREDICTION_PHASE_1 = "Predict whether a phase 1 clinical trial will have a mortality event during the trial design phase (1=mortality, 0=no mortality) "

    def get_train_val_test(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self == Task.CLINICAL_TRIAL_SUCCESS_PRED:
            train = pd.read_parquet("tasks/trial_approval/train_data.parquet")
            val = pd.read_parquet("tasks/trial_approval/val_data.parquet")
            test = pd.read_parquet("tasks/trial_approval/test_data.parquet")
        elif self == Task.CLINICAL_TRIAL_SUCCESS_PRED_PHASE_1:
            train = pd.read_parquet("tasks/trial_approval/phase1_train_data.parquet")
            val = pd.read_parquet("tasks/trial_approval/phase1_val_data.parquet")
            test = pd.read_parquet("tasks/trial_approval/phase1_test_data.parquet")
        elif self == Task.CLINICAL_TRIAL_SUCCESS_PRED_PHASE_2:
            train = pd.read_parquet("tasks/trial_approval/phase2_train_data.parquet")
            val = pd.read_parquet("tasks/trial_approval/phase2_val_data.parquet")
            test = pd.read_parquet("tasks/trial_approval/phase2_test_data.parquet")
        elif self == Task.CLINICAL_TRIAL_SUCCESS_PRED_PHASE_3:
            train = pd.read_parquet("tasks/trial_approval/phase3_train_data.parquet")
            val = pd.read_parquet("tasks/trial_approval/phase3_val_data.parquet")
            test = pd.read_parquet("tasks/trial_approval/phase3_test_data.parquet")
        elif self == Task.CLINICAL_TRIAL_SUCCESS_PRED_PHASE_4:
            train = pd.read_parquet("tasks/trial_approval/phase4_train_data.parquet")
            val = pd.read_parquet("tasks/trial_approval/phase4_val_data.parquet")
            test = pd.read_parquet("tasks/trial_approval/phase4_test_data.parquet")
        elif self == Task.SERIOUS_ADVERSE_EVENT_PHASE_1:
            train = pd.read_parquet("tasks/adverse_event/phase1_train_data.parquet")
            val = pd.read_parquet("tasks/adverse_event/phase1_val_data.parquet")
            test = pd.read_parquet("tasks/adverse_event/phase1_test_data.parquet")
        elif self == Task.PATIENT_DROPOUT_PHASE_1:
            train = pd.read_parquet("tasks/patient_dropout/phase1_train_data.parquet")
            val = pd.read_parquet("tasks/patient_dropout/phase1_val_data.parquet")
            test = pd.read_parquet("tasks/patient_dropout/phase1_test_data.parquet")
        elif self == Task.MORTALITY_PREDICTION_PHASE_1:
            train = pd.read_parquet("tasks/mortality/phase1_train_data.parquet")
            val = pd.read_parquet("tasks/mortality/phase1_val_data.parquet")
            test = pd.read_parquet("tasks/mortality/phase1_test_data.parquet")
        else:
            raise NotImplementedError(f"Task {self} not implemented")
        return train, val, test


class FeatureOp(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    REFINE = "refine"


class FeatureType(StrEnum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    MULTICATEGORICAL = "multi-categorical"


class FeatureSource(StrEnum):
    PUBMED = "pubmed"
    RELATED_CLINICAL_TRIALS = "related_clinical_trials"
    CURRENT_TRIAL_SUMMARY = "current_trial_summary"


class ProposerOutput(NamedTuple):
    feature_operation: FeatureOp
    feature_name: str
    feature_explanation: str


class FeaturePlan(NamedTuple):
    feature_name: str
    feature_type: dict[str, FeatureType]
    data_sources: list[FeatureSource]
    possible_values: dict[str, list[str]]
    example_values: list[dict[str, str]]
    plan_steps: list[str]
    proposer_outputs: list[ProposerOutput]

    def pretty_print(self) -> str:
        return f"""
    feature_name: {self.feature_name}
    feature_type: {self.feature_type}
    possible_values: {self.possible_values}
    """.strip()


class FeaturePlanV2(NamedTuple):
    feature_name: str
    feature_idea: str
    feature_type: dict[str, FeatureType]
    data_sources: list[FeatureSource]
    example_values: list[dict[str, str]]
    possible_values: dict[str, list[str]]
    feature_instructions: str


class PlannerOutput(NamedTuple):
    feature_plans: dict[str, FeaturePlan]


class ModelEvalResult(NamedTuple):
    roc_auc: float
    f1: float
    pr_auc: float
    feature_importance: list
    wrong_idxs: list[int]
    wrong_preds: list[int]
    wrong_df: pd.DataFrame
    pipeline: Pipeline


class EvalOutput(NamedTuple):
    model_eval_result: ModelEvalResult
    suggestions: List[str]


class Output(NamedTuple):
    eval_output: EvalOutput
    lr_eval_output: EvalOutput
    proposer_output: ProposerOutput
    planner_output: PlannerOutput
    pipeline: Pipeline
    lr_pipeline: Pipeline
    df: pd.DataFrame
    val_df: pd.DataFrame
    suggestion_index: int
    raw_features: dict
    raw_val_features: dict

    def get_next_suggestion(self) -> str:
        return self.eval_output.suggestions[self.suggestion_index]

    def get_modified_feature_name(
        self, previous_input: Optional["Output"]
    ) -> Optional[str]:
        if self.proposer_output.feature_operation.value == FeatureOp.ADD.value:
            planned_names = self.planner_output.feature_plans.keys()
            previous_planned_names = set()
            if previous_input is not None:
                previous_planned_names = (
                    previous_input.planner_output.feature_plans.keys()
                )

            new = planned_names - previous_planned_names
            assert len(new) == 1
            return list(new)[0]
        else:
            return self.proposer_output.feature_name


class OutputV2(NamedTuple):
    xgb_eval_output: EvalOutput
    lr_eval_output: EvalOutput
    rf_eval_output: EvalOutput
    test_xgb_eval_output: ModelEvalResult
    test_lr_eval_output: ModelEvalResult
    test_rf_eval_output: ModelEvalResult
    operation: Optional[ProposerOutput]
    feature_plans: dict[str, FeaturePlanV2]
    df: pd.DataFrame
    val_df: pd.DataFrame
    suggestion_index: int
    raw_features: dict
    raw_val_features: dict
    raw_test_features: dict
    none_explanations: dict[str, dict[str, str]]

    def get_best_eval_output(self) -> tuple[EvalOutput, ModelEvalResult]:
        eval_outputs = [
            (self.xgb_eval_output, self.test_xgb_eval_output),
            (self.lr_eval_output, self.test_lr_eval_output),
            (self.rf_eval_output, self.test_rf_eval_output),
        ]
        best_eval_output = max(
            eval_outputs, key=lambda x: x[0].model_eval_result.roc_auc
        )
        return best_eval_output

    def get_next_suggestion(self) -> str:
        return self.get_best_eval_output()[0].suggestions[self.suggestion_index]


class FeatureProposerZeroShotSingleOutputSignature(dspy.Signature):
    """
    You are an experienced clinical researcher skilled at proposing features for a machine learning model.

    There are no existing features for the model. Your task is to propose a single new feature idea for this model. Be as thorough and as detailed as possible in describing the feature.

    The feature should be built off data from ClinicalTrials.gov and from past scholarly research from PubMed.

    The feature should
    - be simple
    - be explainable
    - NOT be a composite of multiple factors or features
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from publicly available sources from NCT or PubMed
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_idea: str = dspy.OutputField()


class FeatureInitializerZeroShotSignature(dspy.Signature):
    """
    You are an experienced clinical researcher skilled at proposing features for a machine learning model.

    Your task is to propose a comprehensive list of feature ideas (at least 10) for this model. Be as exhaustive and as detailed as possible in describing the feature.

    The features should be built off data from ClinicalTrials.gov and from past scholarly research from PubMed.

    The features should
    - be one of integer, float, boolean, categorical or multicategorical
    - NOT be a composite of multiple factors or features
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from ClinicalTrials.gov or PubMed
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_ideas: list[dict[str, str]] = dspy.OutputField(
        desc="A list of factors with a key 'feature_name' containing the snake-cased feature name and a key 'description' with a brief description of the feature idea."
    )


class FactorAnalystSignature(dspy.Signature):
    """
    You are an experienced clinical researcher.

    You are analyzing clinical trials to deduce factors to help with building a machine learning model for a given task.
    You are given detailed information of a clinical trial and the label for the task.
    Your task is to analyze the key factors that contributed to the particular outcome that can be used to inform future trials.

    You should provide at least 5 factors that are generalizable to other trials. Keep your analysis concise.

    Your factors can be from the trial context, or from historical data in PubMed and other clinical trials in the NCT database.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")

    sample_clinical_trial: str = dspy.InputField()
    trial_task_label: int = dspy.InputField(desc="The task label")

    factors: list[dict[str, str]] = dspy.OutputField(
        desc="A list of factors, with each factor having a 'name' key and a 'description' key containing brief explanation of how it can contribute to the outcome of a trial."
    )


class FeatureInitializerWithFactorsSignature(dspy.Signature):
    """
    You are an experienced biomedical data scientist.

    You are given analyses of key factors that might have influcenced the label of a particular task for past clinical trials.
    Suppose we need to build a machine learning model for this prediction task.
    Based on the analyses provided, summarize a comprehensive list (at least 8) of features that can help with the prediction.

    The features should be built off data from ClinicalTrials.gov and from past scholarly research from PubMed.
    If any of the factors cannot be made into a feature with these two data sources, you should skip and move on to the next factor.

    Your features should
    - be one of integer, float, boolean, categorical or multicategorical
    - be generic enough to apply to most clinical trials
    - NOT be a composite of multiple factors or features
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from ClinicalTrials.gov or PubMed
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    factors: list[dict[str, str]] = dspy.InputField()
    feature_ideas: list[dict[str, str]] = dspy.OutputField(
        desc="A list of factors with a key 'feature_name' containing the snake-cased feature name and a key 'description' with a brief description of the feature idea."
    )


class FeatureInitializerCombinedSignature(dspy.Signature):
    """
    You are an experienced biomedical data scientist.

    You are given a combined list of feature ideas from different experts for a clinical trial machine learning task.
    Summarize and refine them as needed to produce a new list of features that comprehensive, well-defined and non-overlapping.

    Make sure that all features can be built off data from ClinicalTrials.gov and from past scholarly research from PubMed.

    Your features should
    - be one of integer, float, boolean, categorical or multicategorical
    - be generic enough to apply to most clinical trials
    - NOT be a composite of multiple factors or features
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from ClinicalTrials.gov or PubMed
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    combined_feature_ideas: list[dict[str, str]] = dspy.InputField()
    feature_ideas: dict[str, str] = dspy.OutputField(
        desc="A dict where the key is a snake-cased feature name feature name and the value is a description of the feature idea."
    )


class FeatureProposerSingleOutputSignature(dspy.Signature):
    """
    You are an experienced clinical researcher skilled at proposing features for a machine learning model.

    An initial model has been built, and you are working to improve the model further based on suggestions from the data scientist.

    Your job is to propose a SINGLE OPERATION that is ONE OF
    - 'ADD' adding a new feature
    - 'REMOVE' removing a feature
    - 'REFINE' refining an existing feature from the model

    The feature should be built off data from ClinicalTrials.gov and from past scholarly research from PubMed.

    The feature should
    - be simple
    - be explainable
    - NOT be a composite of multiple factors or features
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from publicly available sources from NCT or PubMed
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")

    current_model_features: list[str] = dspy.InputField(desc="Current Model Features")
    suggestion: str = dspy.InputField(desc="The suggestion from the data scientist")

    operation: FeatureOp = dspy.OutputField(desc="The feature operation to perform")
    feature_name: Optional[str] = dspy.OutputField(
        desc="The name of the feature to apply this operation on. This should be None if this is an ADD operation."
    )
    feature_explanation: str = dspy.OutputField(
        desc="The explanation of the feature add, removal, or refinement"
    )


class FeatureProposerSingleOutputV2Signature(dspy.Signature):
    """
    You are an experienced clinical researcher skilled at proposing features for a machine learning model.

    An initial model has been built, and you are working to improve the model further by incorporating a suggestion from an expert
    for an update to the features. The suggestion can be either a generic suggestion, or a detailed analysis of a prediction that a model got wrong.

    Your job is to propose a SINGLE OPERATION that is ONE OF
    - 'ADD' adding a new feature
    - 'REMOVE' removing a feature
    - 'REFINE' refining an existing feature from the model

    The feature should be built off data from ClinicalTrials.gov and from past scholarly research from PubMed.

    The feature should
    - be simple
    - be explainable
    - NOT be itself the output of another machine learning model
    - NOT require data that cannot be retrieved from publicly available sources from NCT or PubMed
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")

    current_features_with_plan: List[Tuple[str, str]] = dspy.InputField(
        desc="The current features with their current plans"
    )
    suggestion: str = dspy.InputField(desc="The suggestion from the data scientist")

    operation: FeatureOp = dspy.OutputField(desc="The feature operation to perform")
    feature_name: str = dspy.OutputField(
        desc="If REFINE or REMOVE: The name of the feature to apply this operation on. If ADD: the snake_cased named of this feature"
    )
    operation_description: str = dspy.OutputField(
        desc="The description of the feature add, removal, or refinement. This should be as detailed as possible to fully explain the operation."
    )


class FeaturePlannerV2Signature(dspy.Signature):
    """
    You are an expert data scientist.

    You are given an idea for a single feature to be used in a machine learning model for a clinical trial task.

    For this single feature, you are defining a feature schema for your co-workers to construct the feature for each clinical trial.

    The final built feature should be a JSON object
    - If there's only a single value, it should be a JSON with a single key "value" and the value.
    - If there are multiple values, it should be a JSON with multiple keys, each key corresponding to a sub-feature name, and the value corresponding to the sub-feature value.

    The schema and instruction should be as simple as possible to represent the feature idea.
    Your instruction should be clear, and allow for the feature to be computed consistently and reliably.
    The instruction needs to be explicit and avoid ambigiuity since multiple teams are working together.
    - For e.g., if weights need to be assigned, they should be explicitly defined in the instructions.

    The feature should be built off data from ClinicalTrials.gov and from past scholarly research from PubMed.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_name: str = dspy.InputField()
    feature_idea: str = dspy.InputField()

    feature_type: dict[str, FeatureType] = dspy.OutputField(
        desc="""
        The type of the feature.
        
        If this is a single valued feature, this should be a dict with a single key "value" and the expected type.
        If this is a multi-valued feature, each key should correspond to the sub-feature name, and the value should be the expected type.
        
        The allowed types are:
        'boolean' features are for 'True' / 'False' questions
        'integer' features are for integral numbers
        'float' features are for floating point numbers
        'categorical' features are for single-valued categorical features, returned as a String from `possible_values`
        'multi-categorical' features are for multi-valued categorical features, returned as one or more Strings from `possible_values` in a JSON array
        """
    )
    data_sources: list[FeatureSource] = dspy.OutputField(
        desc="a list of data sources to retrieve data from. This can include existing literature from 'pubmed' or 'current_trial_summary' or 'related_clinical_trials'"
    )
    example_values: list[dict] = dspy.OutputField(
        desc="one or more examples of an expected value for this data. This should be correctly formated as the output JSON"
    )
    possible_values: dict[str, list[str]] = dspy.OutputField(
        desc="""
        (for 'categorical' and 'multi-categorical' only) a list of possible values for this data

        If this is a single valued feature, this should be a dict with a single key "value" and the possible values.
        If this is a multi-valued feature, each key should correspond to the sub-feature name, and the value should be the possible values.
        """
    )
    feature_instructions: str = dspy.OutputField(
        desc="""
        Instructions for researching and building this feature. The instructions should be clear, explicitly define the kind of research and data to look up, and make sure it is as precise as possible.
        """
    )


class FeaturePlannerSignature(dspy.Signature):
    """
    You are an expert data scientist.

    You are given an idea for a single feature to be used in a machine learning model for a clinical trial task.

    For this single feature, you are writing a plan as a series of steps in `plan_steps` for your co-workers to construct the feature for each clinical trial.
    The final built feature should be a JSON
    - If there's only a single value, it should be a JSON with a single key "value" and the value.
    - If there are multiple values, it should be a JSON with multiple keys, each key corresponding to a sub-feature name, and the value corresponding to the sub-feature value.

    The plan should be as simple as possible to represent the feature idea.
    Your plan should be clear, and allow for the feature to be computed consistently and reliably.
    The plan needs to be explicit and avoid ambigiuity since multiple teams are working together.
    The final step of the plan should construct the JSON feature.

    For e.g., if weights need to be assigned, they should be explicitly defined in the instructions.

    The feature should be built off data from ClinicalTrials.gov and from past scholarly research from PubMed.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_idea: str = dspy.InputField()

    feature_name: str = dspy.OutputField(desc="a snakecased name for the feature")
    feature_type: dict[str, FeatureType] = dspy.OutputField(
        desc="""
        The type of the feature.
        
        If this is a single valued feature, this should be a dict with a single key "value" and the expected type.
        If this is a multi-valued feature, each key should correspond to the sub-feature name, and the value should be the expected type.
        
        The allowed types are:
        'boolean' features are for 'True' / 'False' questions
        'integer' features are for integral numbers
        'float' features are for floating point numbers
        'categorical' features are for single-valued categorical features, returned as a String from `possible_values`
        'multi-categorical' features are for multi-valued categorical features, returned as one or more Strings from `possible_values` in a JSON array
        """
    )
    data_sources: list[FeatureSource] = dspy.OutputField(
        desc="a list of data sources to retrieve data from. This can include existing literature from 'pubmed' or 'current_trial_summary' or 'related_clinical_trials'"
    )
    example_values: list[dict[str, str]] = dspy.OutputField(
        desc="one or more examples of an expected value for this data. This should be correctly formated as the output JSON"
    )
    possible_values: dict[str, list[str]] = dspy.OutputField(
        desc="""
        (for 'categorical' and 'multi-categorical' only) a list of possible values for this data

        If this is a single valued feature, this should be a dict with a single key "value" and the possible values.
        If this is a multi-valued feature, each key should correspond to the sub-feature name, and the value should be the possible values.
        """
    )
    plan_steps: list[str] = dspy.OutputField(
        desc="""
        The feature plan. This is a sequence of instructions on how to compute this feature.
        - Each step should clearly define its expected output.
        - Each step should be precise.
        - Avoid unnecessary / duplicative steps
        """
    )


class FeatureBuilderStepSignature(dspy.Signature):
    """
    You are part of a clinical research team creating features from clinical trials for machine learning models.

    You are investigating a particular clinical trial, and you are executing a specific step in a series of steps as part
    of a feature generation workflow.

    Follow the instructions in 'your_step' exactly and return your results in 'output_value'.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    current_clinical_trial_context: str = dspy.InputField(
        desc="The particular trial you are looking at"
    )
    feature_name: str = dspy.InputField(
        desc="The name of the feature your team is investigating"
    )
    your_step: str = dspy.InputField(desc="The step you need to execute")
    data_sources: list[FeatureSource] = dspy.InputField(
        desc="Suggested data sources to reference"
    )
    previous_steps: list[str] = dspy.InputField(
        desc="The steps that were previously executed"
    )
    last_step_outputs: list[str] = dspy.InputField(
        desc="The outputs from the previous steps"
    )

    output_value: str = dspy.OutputField()


class FeatureBuilderFinalStepSignature(dspy.Signature):
    """
    You are part of a clinical research team creating features from clinical trials for machine learning models.

    You are investigating a particular clinical trial, and you are executing a specific step in a series of steps as part
    of a feature generation workflow.

    Follow the instructions in 'your_step' exactly and provide as much detail as possible.

    The final feature value should be a JSON
    - If there's only a single value, it should be a JSON with a single key "value" and the value.
    - If there are multiple values, it should be a JSON with multiple keys, each key corresponding to a sub-feature name, and the value corresponding to the sub-feature value.

    If you are not able to generate an feature_value for any reason, output an empty JSON for feature_value.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    current_clinical_trial_context: str = dspy.InputField(
        desc="The particular trial you are looking at"
    )
    feature_name: str = dspy.InputField(
        desc="The name of the feature your team is investigating"
    )
    feature_type: dict[str, FeatureType] = dspy.InputField(
        desc="""
        The type of the feature.
        
        If this is a single valued feature, this should be a dict with a single key "value" and the expected type.
        If this is a multi-valued feature, each key should correspond to the sub-feature name, and the value should be the expected type.
        
        The allowed types are:
        'boolean' features are for 'True' / 'False' questions
        'integer' features are for integral numbers
        'float' features are for floating point numbers
        'categorical' features are for single-valued categorical features, returned as a String from `possible_values`
        'multi-categorical' features are for multi-valued categorical features, returned as one or more Strings from `possible_values` in a JSON array
        """
    )
    example_values: list[dict[str, str]] = dspy.InputField(
        desc="one or more examples of a value for this feature"
    )
    possible_values: dict[str, list[str]] = dspy.InputField(
        desc="""
        (for 'categorical' and 'multi-categorical' only) a list of possible values for this data

        If this is a single valued feature, this should be a dict with a single key "value" and the possible values.
        If this is a multi-valued feature, each key should correspond to the sub-feature name, and the value should be the possible values.
        """
    )

    your_step: str = dspy.InputField(desc="The step you need to execute")
    previous_steps: list[str] = dspy.InputField(
        desc="The steps that were previously executed"
    )
    last_step_outputs: list[str] = dspy.InputField(
        desc="The outputs from the previous steps"
    )

    feature_value: dict[str, str] = dspy.OutputField()


# class FeatureBuilderMultiResearchSignature(dspy.Signature):
#     """
#     You are part of a clinical research team creating features for clinical trial machine learning models.

#     You are investigating a particular clinical trial. You are given a dict of features and their corresponding plans that you need to construct.

#     Your job is to research and gather all required information in order to build the features.
#     """

#     task: str = dspy.InputField(desc="The task for this machine learning model.")
#     nctid: str = dspy.InputField(
#         desc="The NCT ID of the clinical trial you are looking at"
#     )
#     feature_plans: dict[str, dict] = dspy.InputField(
#         desc="A dict of feature names to their corresponding plans."
#     )

#     research_results: str = dspy.OutputField(
#         desc="The research results. The data should be sufficient to build all the features"
#     )


class FeatureGroupingSignature(dspy.Signature):
    """
    You are part of a clinical research team creating features for clinical trial machine learning models.

    You have a list of features that need to be constructed.

    Your job is to identify groups of features that rely on similar data, and can be researched together based on their plan instructions.
    The grouping should be based on the feature plans and the kind of data the features require to enable efficient research.

    You should return a list of groups, where each group is a list of feature names that can be constructed together.
    There should be a maximum of 5 features in each group.
    Each feature should only be in one group.
    If a feature cannot be grouped with any other feature, it should be in its own group.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_plans: dict[str, dict] = dspy.InputField(
        desc="A dict of feature names to their corresponding plans."
    )
    groups: list[list[str]] = dspy.OutputField(
        desc="A list of groups, where each group is a list of feature names that can be constructed together."
    )


# class FeatureBuilderResearchInstructionSignature(dspy.Signature):
#     """
#     You are part of a clinical research team creating features for clinical trial machine learning models.

#     You are investigating a particular clinical trial. You are given a dict of features and their corresponding instructions that your team needs to do research on.

#     Your job is to create a synthesized, combined set of instructions for your team to do deep research and extract the data necessary to build all the features.
#     You should think about how best to create a streamlined set of instructions that'll enable the researcher to most efficiently find the necessary data.

#     The instructions should contain details such as
#     - the aggregated set of data to look for
#     - where to look for if
#     - how to derive research results from the data

#     Your instructions should be detailed, clear and concise.
#     """

#     task: str = dspy.InputField(desc="The task for this machine learning model.")
#     feature_plans: dict[str, dict] = dspy.InputField(
#         desc="A dict of feature names to their corresponding plans."
#     )

#     research_instructions: str = dspy.OutputField(
#         desc="Your combined research instructions that can be used to build all the features."
#     )


class FeatureBuilderResearchMultiSignature(dspy.Signature):
    """
    You are part of a clinical research team creating features for clinical trial machine learning models.

    You are investigating a particular clinical trial. You are given a dict of features that your team needs to do research on.
    You should make use of the given tools to do deep research, gather information and provide the data necessary to build all the features.

    Do not focus on formatting the features correctly, instead focus on making sure you have a full and complete set of data.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    nctid: str = dspy.InputField(
        desc="The NCT ID of the clinical trial you are looking at"
    )

    # instruction: str = dspy.InputField(
    #     desc="The instruction for how to go about doing the research."
    # )
    feature_plans: dict[str, dict] = dspy.InputField(
        desc="A dict of feature names to their corresponding plans. This is the ultimate set of features we're doing research for."
    )
    research_results: str = dspy.OutputField(
        desc="Your summarized research results that can be used to build the features. You should produce enough results that are sufficient to build all the features."
    )


class FeatureBuilderConstructSignature(dspy.Signature):
    """
    You are part of a clinical research team creating features for clinical trial machine learning models.

    You are investigating a particular clinical trial. You are given a dict of features and their corresponding plans that your team needs to construct.
    A previous step has already gathered the necessary research results for these features, your job is to CORRECTLY construct these in the format prescribed by the feature plan.

    If there is
    - insufficient information
    - missing information
    - uncertainty/ambiguity
    for any of the features, you should return the value 'None' for that feature (or sub-feature) and provide explanations for the feature you can't build.

    YOU MUST HAVE AN OUTPUT FOR EACH FEATURE
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    feature_plans: dict[str, dict] = dspy.InputField(
        desc="A dict of feature names to their corresponding plans."
    )
    research_results: str = dspy.InputField(
        desc="The research results. The data should be sufficient to build all the features"
    )
    all_feature_values: dict[str, dict[str, Any]] = dspy.OutputField(
        desc="A dict containing the feature name of each feature from feature_plans, and the output for that feature as a STRING. For multi-categorical features, this should be JSON array of String"
    )
    none_feature_explanations: dict[str, str] = dspy.OutputField()


class EvaluatorSignature(dspy.Signature):
    """
    You are an experienced biomedical data scientist.

    You are supervising the construction of a machine learning model for a specific clinical trial task.
    The model must be built with features from data from ClinicalTrials.gov and from past scholarly research from PubMed.
    A version of the model has been trained, and you are provided the current performance.

    Please provide suggestions for
    - additional features
    - refinements to the existing features
    - features to remove

    Keep your suggestions concise, and limit to a maximum of 2-3 suggestions.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")

    roc_auc_score: float = dspy.InputField()
    current_features_with_plan: List[Tuple[str, str]] = dspy.InputField()
    current_feature_importances: list = dspy.InputField()

    suggestions: list[str] = dspy.OutputField()


class EvaluatorWithExampleSignature(dspy.Signature):
    """
    You are an experienced biomedical data scientist.

    You are supervising the construction of a machine learning model for a specific clinical trial task.
    The model must be built with features from data from ClinicalTrials.gov and from past scholarly research from PubMed.
    A version of the model has been trained, and you are provided the current performance.

    Based on the example, please provide a single suggestion for improving the model. Examples of this can be
    - additional features
    - refinements to the existing features
    - features to remove

    You are also given a sample incorrect prediction from the validation set. Use this to help inform your suggestion.

    Keep your suggestion concise.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")

    roc_auc_score: float = dspy.InputField()
    current_features_with_plan: List[Tuple[str, str]] = dspy.InputField()
    current_feature_importances: list = dspy.InputField()
    sample_incorrect_prediction: str = dspy.InputField()

    suggestion: str = dspy.OutputField()


class EvaluatorWithExampleSignatureV2(dspy.Signature):
    """
    You are an experienced clinical researcher.

    You are supervising the construction of a machine learning model for a specific clinical trial task.
    The model must be built with features from data from ClinicalTrials.gov and from past scholarly research from PubMed.
    A version of the model has been trained, and you are provided the current performance, and an example of an incorrect prediction from the current model.

    Based on the example and using the tools provided to help with further research, please conduct some analysis on why the model made the incorrect prediction.

    You should consider
    - features that were missed, and could have helped with the prediction
    - features that were not useful
    - misconstructed features
    = feature plans that are not properly set up (e.g. missing instructions / missing categories)

    Your analysis should be generalizable to other trials where possible
    Keep your analysis concise.
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")

    roc_auc_score: float = dspy.InputField()
    current_features_with_plan: List[Tuple[str, str]] = dspy.InputField()
    example: str = dspy.InputField()

    analysis: str = dspy.OutputField()


class EvaluatorSummarizerSignature(dspy.Signature):
    """
    You are an experienced clinical researcher.

    You are supervising the construction of a machine learning model for a specific clinical trial task.
    The model must be built with features from data from ClinicalTrials.gov and from past scholarly research from PubMed.

    You are given ideas and analyses from various experts that have analyzed the model results.

    Your job is to summarize and generalize these into a list of a maximum of 4-5 suggestions that can be used to improve the model.

    Each suggestion should either
    - add a new feature
    - refine an existing feature
    - remove an existing feature
    """

    task: str = dspy.InputField(desc="The task for this machine learning model.")
    analyses: list[str] = dspy.InputField(desc="The suggestions from other experts")

    suggestions: list[str] = dspy.OutputField()


class FeatureBuilderV2(dspy.Module):
    def __init__(self):
        super().__init__()
        self.feature_builder_final = dspy.ChainOfThought(
            FeatureBuilderFinalStepSignature
        )

    def forward(
        self,
        feature_name,
        feature_type: dict[str, FeatureType],
        data_sources,
        example_values,
        possible_values,
        plan_steps,
        nctid,
        task: Task,
    ):
        nct_info = get_clinical_trial_info_from_clinical_trials_gov_dict(nctid)
        current_trial_context = get_detailed_nct_info(nctid)

        builderstep = dspy.ReAct(
            FeatureBuilderStepSignature,
            tools=[
                make_pubmed_search(nct_info),
                make_nct_search(nct_info),
                get_detailed_nct_info,
            ],
            max_iters=5,
        )

        results = []
        previous_steps = []
        last_step_outputs = []
        for step in plan_steps[:-1]:
            step_result = builderstep(
                task=task.value,
                current_clinical_trial_context=current_trial_context,
                feature_name=feature_name,
                your_step=step,
                data_sources=data_sources,
                previous_steps=previous_steps,
                last_step_outputs=last_step_outputs,
            )
            results.append(step_result)
            previous_steps.append(step)
            last_step_outputs.append(step_result.output_value)

        builder_result = self.feature_builder_final(
            task=task.value,
            current_clinical_trial_context=current_trial_context,
            feature_name=feature_name,
            feature_type=feature_type,
            example_values=example_values,
            possible_values=possible_values,
            your_step=plan_steps[-1],
            previous_steps=previous_steps,
            last_step_outputs=last_step_outputs,
        )

        print(f"[{feature_name}] {nctid} Got result {builder_result.feature_value}")
        feature_value = builder_result.feature_value

        if len(feature_value) == 0:
            print(f"Failed to generate a feature output for {feature_name} {nctid}")
            print(builder_result)
            print(results)
            return builder_result, {
                "feature_value": {k: None for k in feature_type.keys()},
                "intermediate_results": results,
            }

        for key, value in feature_value.items():
            dspy.Assert(
                key in feature_type.keys(),
                f"Unexpected sub-feature key: {key}. Expected one of {feature_type.keys()}",
            )

            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]

            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            if value == "None":
                print(
                    f"Failed to generate a feature output for {feature_name}/{key} {nctid}"
                )
                print(builder_result)
                print(results)
                value = None

            ft = feature_type[key]

            if ft.value == FeatureType.BOOLEAN.value:
                if value is not None:
                    dspy.Assert(
                        value in ("True", "False"),
                        f"'boolean' features must have values 'True' or 'False'. Got {value} in {key}.",
                    )
                    value = value == "True"
                else:
                    value = np.nan
            elif ft.value == FeatureType.CATEGORICAL.value:
                possible_value_for_key = possible_values[key]
                if value is not None:
                    dspy.Assert(
                        value in possible_value_for_key,
                        f"'categorical' feature must be a single value in {possible_value_for_key}. Got {value} in {key}.",
                    )
            elif ft.value == FeatureType.MULTICATEGORICAL.value:
                possible_value_for_key = possible_values[key]
                possible_value_for_key_as_set = set(possible_value_for_key)
                if value is not None:
                    try:
                        as_list = json.loads(value)
                    except ValueError:
                        as_list = None

                    dspy.Assert(
                        as_list is not None and isinstance(as_list, list),
                        f"'multi-categorical' feature must be returned as a JSON array of values that are from {possible_value_for_key} e.g. `[\"{possible_value_for_key[0]}\"]`. Got {value} in {key}",
                    )

                    dspy.Assert(
                        set(as_list).issubset(possible_value_for_key_as_set),  # type: ignore
                        f"'multi-categorical' feature must be values from {possible_value_for_key}. The following values are invalid: {set(as_list) - possible_value_for_key_as_set} in {key}",  # type: ignore
                    )
            elif ft.value == FeatureType.INTEGER.value:
                if value is not None:
                    dspy.Assert(
                        value.isdigit(),
                        f"'integer' feature must return an integer value. Got {value} in {key}",
                    )
                    value = int(value)
                else:
                    value = np.nan
            elif ft.value == FeatureType.FLOAT.value:
                if value is not None:
                    dspy.Assert(
                        re.match(r"^-?\d+(?:\.\d+)$", value) is not None,
                        f"'float' feature must return a floating point value. Got {value} in {key}",
                    )
                    value = float(value)
                else:
                    value = np.nan

            feature_value[key] = value

        # print(f"[{threading.get_ident()}][{nctid}] Done building {feature_name} - {builder_result.feature_value}")
        return builder_result, {
            "feature_value": feature_value,
            "intermediate_results": results,
        }


class FeatureGrouper(dspy.Module):
    def __init__(self):
        super().__init__()
        self.feature_grouper = dspy.ChainOfThought(FeatureGroupingSignature)

    def forward(
        self,
        task: Task,
        feature_plans: dict[str, FeaturePlanV2],
    ):
        serialized_feature_plans = {
            feature_name: json.loads(dump_as_json(plan))
            for feature_name, plan in feature_plans.items()
        }

        grouped = self.feature_grouper(
            task=task.value, feature_plans=serialized_feature_plans
        )

        dspy.Assert(
            len(grouped.groups) > 0,
            "No groups were generated. Expected at least one group.",
        )

        dspy.Assert(
            sum(len(group) for group in grouped.groups)
            == len(serialized_feature_plans),
            f"The total number of features in the groups does not match the number of features in the feature plans. Got {sum(len(group) for group in grouped.groups)}. Expected {len(serialized_feature_plans)}",
        )

        grouped_features: list[list[str]] = grouped.groups
        grouped_feature_all = set(fn for fns in grouped_features for fn in fns)
        missing = set(feature_plans.keys()) - grouped_feature_all

        dspy.Assert(
            len(missing) == 0,
            f"The following features are missing from the groups: {', '.join(missing)}",
        )

        grouped_feature_plans = []
        for group in grouped.groups:
            grouped_feature_plan = {
                feature_name: feature_plans[feature_name] for feature_name in group
            }
            grouped_feature_plans.append(grouped_feature_plan)

        return grouped_feature_plans


class FeatureBuilderV3(dspy.Module):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.feature_grouper = dspy.ChainOfThought(FeatureGroupingSignature)
        self.constructor = dspy.ChainOfThought(FeatureBuilderConstructSignature)
        # self.research_instruction = dspy.ChainOfThought(
        #     FeatureBuilderResearchInstructionSignature
        # )

    def forward(
        self,
        nctid: str,
        feature_plan_group: dict[str, FeaturePlanV2],
    ):
        nct_info = get_clinical_trial_info_from_clinical_trials_gov_dict(nctid)
        # current_trial_context = get_detailed_nct_info(nctid)
        serialized_feature_plan_group = {
            feature_name: json.loads(dump_as_json(plan))
            for feature_name, plan in feature_plan_group.items()
        }

        simplified_feature_plan_group = {
            fn: {
                k: v
                for k, v in d.items()
                if k not in ["data_sources", "example_values", "possible_values"]
            }
            for fn, d in serialized_feature_plan_group.items()
        }

        # TODO: maybe create an instruction for the whole group

        # if len(feature_plan_group) == 1:
        #     instruction = feature_plan_group[
        #         list(feature_plan_group.keys())[0]
        #     ].feature_instructions
        # else:
        #     instruction = self.research_instruction(
        #         task=self.task.value,
        #         feature_plans=simplified_feature_plan_group,
        #     ).research_instructions

        research = dspy.ReAct(
            FeatureBuilderResearchMultiSignature,
            tools=[
                make_pubmed_search(nct_info),
                make_nct_search(nct_info),
                get_detailed_nct_info,
            ],
            max_iters=5,
        )

        values = {}
        research_result = research(
            task=self.task.value,
            nctid=nctid,
            feature_plans=simplified_feature_plan_group,
            # instruction=instruction,
        )

        builder_result = self.constructor(
            task=self.task.value,
            feature_plans=serialized_feature_plan_group,
            research_results=research_result.research_results,
        )

        print(f"[{nctid}] Got result {builder_result.all_feature_values}")
        feature_values = deepcopy(builder_result.all_feature_values)

        missing_features = set(feature_plan_group.keys()) - set(feature_values.keys())
        dspy.Assert(
            len(missing_features) == 0,
            f"Some features were not generated: {missing_features}. Expected all features to be generated.",
        )

        for feature_name, feature_value in feature_values.items():
            if feature_name not in feature_plan_group:
                continue

            feature_type = feature_plan_group[feature_name].feature_type
            possible_values = feature_plan_group[feature_name].possible_values

            if len(feature_value) == 0:
                print(
                    f"Completely failed to generate a feature output for {feature_name} {nctid}"
                )
                print(builder_result)
                values[feature_name] = {k: None for k in feature_type.keys()}
                continue

            for key, value in feature_value.items():
                value = soft_assert(
                    value,
                    key in feature_type.keys(),
                    f"Unexpected sub-feature key: {key} for {feature_name}. Expected one of {feature_type.keys()}",
                )

                if isinstance(value, list):
                    value = json.dumps(value)

                if not isinstance(value, str):
                    value = str(value)

                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]

                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                if value == "None":
                    print(
                        f"Failed to generate a feature output for {feature_name}/{key} {nctid}"
                    )
                    print(builder_result)
                    value = None

                ft = feature_type[key]

                if ft.value == FeatureType.BOOLEAN.value:
                    if value is not None:
                        value = value.lower()

                        value = soft_assert(
                            value,
                            value in ("true", "false"),
                            f"'boolean' features must have values 'True' or 'False'. Got '{value}' in {feature_name}/{key}.",
                        )
                        value = np.nan if value is None else value == "true"
                    else:
                        value = np.nan
                elif ft.value == FeatureType.CATEGORICAL.value:
                    possible_value_for_key = possible_values[key]
                    if value is not None:
                        value = soft_assert(
                            value,
                            value in possible_value_for_key,
                            f"'categorical' feature must be a single value in {possible_value_for_key}. Got '{value}' in {feature_name}/{key}.",
                        )
                elif ft.value == FeatureType.MULTICATEGORICAL.value:
                    possible_value_for_key = possible_values[key]
                    possible_value_for_key_as_set = set(possible_value_for_key)
                    if value is not None:
                        try:
                            as_list = json.loads(value)
                        except ValueError:
                            as_list = None

                        as_list = soft_assert(
                            as_list,
                            as_list is not None and isinstance(as_list, list),
                            f"'multi-categorical' feature must be returned as a JSON array of values that are from {possible_value_for_key} e.g. `[\"{possible_value_for_key[0]}\"]`. Got {value} in {feature_name}/{key}",
                        )

                        if as_list is None:
                            value = None
                        else:
                            if len(as_list) == 1 and (
                                as_list[0] == "None" or as_list[0] is None
                            ):
                                value = None
                            else:
                                as_list = soft_assert(
                                    as_list,
                                    set(as_list).issubset(
                                        possible_value_for_key_as_set
                                    ),  # type: ignore
                                    f"'multi-categorical' feature must be values from {possible_value_for_key}. The following values are invalid: {set(as_list) - possible_value_for_key_as_set} in {feature_name}/{key}",  # type: ignore
                                )

                                value = as_list
                elif ft.value == FeatureType.INTEGER.value:
                    if value is not None:
                        value = soft_assert(
                            value,
                            value.isdigit(),
                            f"'integer' feature must return an integer value. Got {value} in {feature_name}/{key}",
                        )
                        value = np.nan if value is None else float(int(value))
                    else:
                        value = np.nan
                elif ft.value == FeatureType.FLOAT.value:
                    if value is not None:
                        value = soft_assert(
                            value,
                            re.match(r"^-?\d+(?:\.\d+)$", value) is not None,
                            f"'float' feature must return a floating point value. Got {value} in {feature_name}/{key}",
                        )
                        value = np.nan if value is None else float(value)
                    else:
                        value = np.nan

                feature_value[key] = value
            values[feature_name] = feature_value
        return values, {
            "research_results": research_result.research_results,
            "research_result_reasoning": research_result.reasoning,
            "builder_reasoning": builder_result.reasoning,
            "none_feature_explanations": builder_result.none_feature_explanations,
            # "instruction": instruction,
        }


class FeatureProposer(dspy.Module):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.proposer = dspy.ChainOfThought(FeatureProposerSingleOutputSignature)

    def forward(self, previous_output: Output):
        fns = previous_output.planner_output.feature_plans.keys()

        proposer_result = self.proposer(
            task=self.task.value,
            current_model_features=[
                fp.pretty_print()
                for fp in previous_output.planner_output.feature_plans.values()
            ],
            suggestion=previous_output.get_next_suggestion(),
        )

        if proposer_result.operation.value == FeatureOp.ADD.value:
            dspy.Assert(
                proposer_result.feature_name is None,
                "When the operation is ADD, the feature_name should be None",
            )
        else:
            dspy.Assert(
                proposer_result.feature_name in fns,
                f"When the operation is REMOVE or REFINE, the feature_name should be an existing feature in {fns}, got {proposer_result.feature_name}",
            )

        return proposer_result


class FeatureProposerV2(dspy.Module):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.proposer = dspy.ChainOfThought(FeatureProposerSingleOutputV2Signature)

    def forward(self, previous_output: OutputV2):
        fns = previous_output.feature_plans.keys()

        current_features_with_plan = [
            (fp.feature_name, dump_as_json(fp, pretty=False))
            for fp in previous_output.feature_plans.values()
        ]

        proposer_result = self.proposer(
            task=self.task.value,
            current_features_with_plan=current_features_with_plan,
            suggestion=previous_output.get_next_suggestion(),
        )

        if proposer_result.operation.value == FeatureOp.ADD.value:
            dspy.Assert(
                proposer_result.feature_name not in fns,
                f"The feature_name {proposer_result.feature_name} is already in use. Pick a slightly different name",
            )
        else:
            dspy.Assert(
                proposer_result.feature_name in fns,
                f"When the operation is REMOVE or REFINE, the feature_name should be an existing feature in {fns}, got {proposer_result.feature_name}",
            )

        return ProposerOutput(
            feature_name=proposer_result.feature_name,
            feature_explanation=proposer_result.operation_description,
            feature_operation=proposer_result.operation,
        )


class FeaturePlanner(dspy.Module):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.planner = dspy.ChainOfThought(FeaturePlannerSignature)

    def run_planner(self, feature_idea: str):
        planner_result = self.planner(task=self.task.value, feature_idea=feature_idea)

        dspy.Assert(
            set(planner_result.possible_values.keys()).issubset(
                set(planner_result.feature_type.keys())
            ),
            f"The possible_values keys must be a subset of the feature_type keys, got {planner_result.possible_values.keys()} and {planner_result.feature_type.keys()}",
        )

        for key, keytype in planner_result.feature_type.items():
            if (
                keytype.value == FeatureType.MULTICATEGORICAL.value
                or keytype.value == FeatureType.CATEGORICAL.value
            ):
                dspy.Assert(
                    key in planner_result.possible_values.keys(),
                    f"If the feature type is 'categorical' or 'multi-categorical', the possible_values must be defined for that key. {key} is not in {planner_result.possible_values.keys()}",
                )

        return planner_result

    def forward(
        self, proposer_result: ProposerOutput, previous_output: Optional[Output] = None
    ):
        plans = (
            {}
            if previous_output is None
            else deepcopy(previous_output.planner_output.feature_plans)
        )

        if proposer_result.feature_operation.value == FeatureOp.ADD.value:
            plan = self.run_planner(feature_idea=proposer_result.feature_explanation)
            fp = FeaturePlan(
                feature_name=plan.feature_name,
                feature_type=plan.feature_type,
                data_sources=plan.data_sources,
                possible_values=plan.possible_values,
                example_values=plan.example_values,
                plan_steps=plan.plan_steps,
                proposer_outputs=[proposer_result],
            )
            plans[fp.feature_name] = fp
        elif proposer_result.feature_operation.value == FeatureOp.REMOVE.value:
            del plans[proposer_result.feature_name]  # type: ignore
            plan = None
        else:  # REFINE
            current = plans[proposer_result.feature_name]  # type: ignore
            current.proposer_outputs.append(proposer_result)
            explanations = "\n---\n".join(
                po.feature_explanation for po in current.proposer_outputs
            )
            plan = self.run_planner(feature_idea=explanations)
            fp = FeaturePlan(
                feature_name=proposer_result.feature_name,  # type: ignore
                feature_type=plan.feature_type,
                data_sources=plan.data_sources,
                possible_values=plan.possible_values,
                example_values=plan.example_values,
                plan_steps=plan.plan_steps,
                proposer_outputs=current.proposer_outputs,
            )
            plans[fp.feature_name] = fp

        return plan, plans


class Evaluator(dspy.Module):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.evaluator_no_example = dspy.ChainOfThought(EvaluatorSignature)
        self.evaluator = dspy.ChainOfThought(EvaluatorWithExampleSignature)
        self.rng = np.random.default_rng(42)

    def forward(
        self, planner_output: PlannerOutput, model_eval_result: ModelEvalResult
    ):
        # run the no example evaluator
        eval_result = self.evaluator_no_example(
            task=self.task.value,
            roc_auc_score=model_eval_result.roc_auc,
            current_feature_importances=model_eval_result.feature_importance,
            current_features_with_plan=[
                (fp.feature_name, orjson.dumps(fp, default=orjson_default))
                for fp in planner_output.feature_plans.values()
            ],
        )
        no_example_suggestions = eval_result.suggestions
        example_suggestions = []

        picks = self.rng.choice(
            model_eval_result.wrong_idxs,
            size=min(3, len(model_eval_result.wrong_idxs)),
            replace=False,
        )

        wrong_idx_to_preds = {
            i: p
            for (i, p) in zip(
                model_eval_result.wrong_idxs, model_eval_result.wrong_preds
            )
        }

        for pick in picks:
            entry = model_eval_result.wrong_df.loc[pick]
            pred = wrong_idx_to_preds[pick]

            correct = 1 if pred == 0 else 0

            example_error = f"""
        ## Predicted {pred}, should be {correct}

        ### Features
        ```
        {yaml.dump(entry.to_dict())}
        ```

        ### Trial Summary
        {yaml.dump(get_clinical_trial_info_from_clinical_trials_gov_dict(entry["id"]))}
            """
            eval_result = self.evaluator(
                task=self.task.value,
                roc_auc_score=model_eval_result.roc_auc,
                current_feature_importances=model_eval_result.feature_importance,
                current_features_with_plan=[
                    (fp.feature_name, orjson.dumps(fp, default=orjson_default))
                    for fp in planner_output.feature_plans.values()
                ],
                sample_incorrect_prediction=example_error,
            )
            example_suggestions.append(eval_result.suggestion)
        return EvalOutput(
            model_eval_result=model_eval_result,
            suggestions=no_example_suggestions[:3] + example_suggestions,
        )


class EvaluatorV2(dspy.Module):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.evaluator_no_example = dspy.ChainOfThought(EvaluatorSignature)
        self.summarizer = dspy.ChainOfThought(EvaluatorSummarizerSignature)
        self.rng = np.random.default_rng(42)

    def forward(
        self,
        feature_plans: dict[str, FeaturePlanV2],
        model_eval_result: ModelEvalResult,
        none_explanations: dict[str, dict[str, str]],
    ):
        current_features_with_plan = [
            (fp.feature_name, dump_as_json(fp, pretty=False))
            for fp in feature_plans.values()
        ]
        # run the no example evaluator
        eval_result = self.evaluator_no_example(
            task=self.task.value,
            roc_auc_score=model_eval_result.roc_auc,
            current_feature_importances=model_eval_result.feature_importance,
            current_features_with_plan=current_features_with_plan,
        )
        no_example_suggestions = eval_result.suggestions
        example_suggestions = []

        picks = self.rng.choice(
            model_eval_result.wrong_idxs,
            size=min(3, len(model_eval_result.wrong_idxs)),
            replace=False,
        )

        wrong_idx_to_preds = {
            i: p
            for (i, p) in zip(
                model_eval_result.wrong_idxs, model_eval_result.wrong_preds
            )
        }

        for pick in picks:
            entry = model_eval_result.wrong_df.loc[pick]
            pick_nct_id = entry["id"]
            nct_info = get_clinical_trial_info_from_clinical_trials_gov_dict(
                pick_nct_id
            )
            pred = wrong_idx_to_preds[pick]

            correct = 1 if pred == 0 else 0
            example_error = f"""
        ## {pick_nct_id} Predicted {pred}, should be {correct}

        ### Features
        ```
        {yaml.dump(entry.to_dict())}
        ```

        ### Reasons for features that are None
        {yaml.dump(none_explanations[pick_nct_id])}
            """
            try:
                example_evaluator = dspy.ReAct(
                    EvaluatorWithExampleSignatureV2,
                    tools=[
                        get_detailed_nct_info,
                        make_pubmed_search(nct_info),
                        make_nct_search(nct_info),
                    ],
                )

                eval_result = example_evaluator(
                    task=self.task.value,
                    roc_auc_score=model_eval_result.roc_auc,
                    current_features_with_plan=current_features_with_plan,
                    example=example_error,
                )
                example_suggestions.append(eval_result.analysis)
            except Exception as e:
                print("Failed to run evaluator", nct_info)
                print(e)
                continue


        return EvalOutput(
            model_eval_result=model_eval_result,
            # suggestions=summarizer_result.suggestions,
            suggestions=example_suggestions + no_example_suggestions,
        )


class _WrappedMPBuilder:
    def __init__(self, klass, task: Task):
        self.feature_builder_klass = klass
        self.task = task

    def _compute_fp_hash(self, fp: FeaturePlan):
        serialized = dill.dumps(fp)
        file_hash = hashlib.sha256()
        file_hash.update(serialized)
        return file_hash.hexdigest()

    def __call__(self, fp: FeaturePlan, nctid: str):
        fp_hash = self._compute_fp_hash(fp)

        prefix = f"{GLOBAL_FB_CACHE_DIR}{fp.feature_name}/{self.task.name}--{fp_hash}"
        plan_path = f"{prefix}/feature_plan.json"
        cached_path = f"{prefix}/{nctid}.json"

        if os.path.exists(cached_path):
            with open(cached_path, "r") as f:
                return dill.loads(bytes.fromhex(json.load(f)["feature_value_b"]))

        os.makedirs(prefix, exist_ok=True)

        try:
            fb = self.feature_builder_klass()
            fb = fb.activate_assertions(max_backtracks=5)
            assert fb._assert_transformed
            builder_result, meta = fb(
                feature_name=fp.feature_name,
                feature_type=fp.feature_type,
                data_sources=fp.data_sources,
                possible_values=fp.possible_values,
                example_values=fp.example_values,
                plan_steps=fp.plan_steps,
                nctid=nctid,
                task=self.task,
            )

            with open(cached_path, "w") as f:
                print(f"Caching {cached_path}")
                json.dump(
                    {
                        "feature_value": meta["feature_value"],
                        "feature_value_b": dill.dumps(meta["feature_value"]).hex(),
                        "builder_result": builder_result.feature_value,
                        "intermediate_results": [
                            r.output_value for r in meta["intermediate_results"]
                        ],
                    },
                    f,
                )

            if not os.path.exists(plan_path):
                with open(plan_path, "wb") as f:
                    f.write(orjson.dumps(fp, default=orjson_default))
            return meta["feature_value"]
        except Exception as e:
            print(f"Failed to run {nctid}", e)
            traceback.print_exc()
            return {k: None for k in fp.feature_type.keys()}


class WrappedFeatureBuilderV3:
    def __init__(self, task: Task):
        self.task = task

    def _compute_fp_hash(self, plans: dict[str, FeaturePlanV2]):
        file_hash = hashlib.sha256()
        for fp in plans.values():
            serialized = orjson.dumps(fp, default=orjson_default)
            file_hash.update(serialized)
        return file_hash.hexdigest()

    def __call__(self, arg: Tuple[str, dict[str, FeaturePlanV2]]):
        nctid, plans = arg

        fp_hash = self._compute_fp_hash(plans)

        prefix = f"{GLOBAL_FB_CACHE_DIR}/{self.task.name}--{fp_hash}"
        plans_path = f"{prefix}/feature_plans.json"
        cached_path = f"{prefix}/{nctid}.json"

        if os.path.exists(cached_path):
            with open(cached_path, "r") as f:
                loaded_f = json.load(f)
                return (
                    nctid,
                    dill.loads(bytes.fromhex(loaded_f["feature_values_b"])),
                    loaded_f["meta"],
                )

        os.makedirs(prefix, exist_ok=True)

        try:
            fb = FeatureBuilderV3(task=self.task)
            fb = fb.activate_assertions(max_backtracks=5)
            values, meta = fb(
                nctid=nctid,
                feature_plan_group=plans,
            )

            with open(cached_path, "w") as f:
                print(f"Caching {cached_path}")
                json.dump(
                    {
                        "feature_values": values,
                        "feature_values_b": dill.dumps(values).hex(),
                        "meta": meta,
                    },
                    f,
                    indent=2,
                )

            if not os.path.exists(plans_path):
                with open(plans_path, "wb") as f:
                    f.write(dump_as_json(plans).encode("utf-8"))

            return (nctid, values, meta)
        except Exception as e:
            print(f"Failed to run {nctid}", e)
            traceback.print_exc()
            return (
                nctid,
                {
                    k: {kk: None for kk in plans[k].feature_type.keys()}
                    for k in plans.keys()
                },
                {},
            )


def compute_features_v3(
    grouper: dspy.Module, nctids: list[str], task, plans: dict[str, FeaturePlanV2]
) -> tuple[dict[str, dict], dict[str, dict[str, str]]]:
    if len(plans) == 1:
        grouped_feature_plans = [plans]
    else:
        grouped_feature_plans: list[dict[str, FeaturePlanV2]] = grouper(
            task=task, feature_plans=plans
        )
    product = itertools.product(nctids, grouped_feature_plans)
    features = tqdm(
        get_global_pool().map(WrappedFeatureBuilderV3(task), product),
        total=len(grouped_feature_plans) * len(nctids),
    )

    agged = defaultdict(dict)
    none_feature_reasons = defaultdict(dict[str, str])
    for nctid, feature_values, metadata in features:
        agged[nctid] = agged[nctid] | feature_values  # type: ignore
        none_feature_reasons[nctid] = none_feature_reasons[nctid] | metadata.get(
            "none_feature_explanations", {}
        )

    return agged, none_feature_reasons


class FeaturePlannerV2(dspy.Module):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.planner = dspy.ChainOfThought(FeaturePlannerV2Signature)

    def forward(self, feature_name: str, feature_idea: str):
        planner_result = self.planner(
            task=self.task.value, feature_name=feature_name, feature_idea=feature_idea
        )

        dspy.Assert(
            set(planner_result.possible_values.keys()).issubset(
                set(planner_result.feature_type.keys())
            ),
            f"The possible_values keys must be a subset of the feature_type keys, got {planner_result.possible_values.keys()} and {planner_result.feature_type.keys()}",
        )

        possible_values = {}
        for key, keytype in planner_result.feature_type.items():
            if (
                keytype.value == FeatureType.MULTICATEGORICAL.value
                or keytype.value == FeatureType.CATEGORICAL.value
            ):
                dspy.Assert(
                    key in planner_result.possible_values.keys(),
                    f"If the feature type is 'categorical' or 'multi-categorical', the possible_values must be defined for that key. {key} is not in {planner_result.possible_values.keys()}",
                )
                possible_values[key] = planner_result.possible_values[key]

        plan = FeaturePlanV2(
            feature_name=feature_name,
            feature_idea=feature_idea,
            feature_type=planner_result.feature_type,
            data_sources=planner_result.data_sources,
            possible_values=possible_values,
            example_values=planner_result.example_values,
            feature_instructions=planner_result.feature_instructions,
        )

        return plan, planner_result


class Initializer(dspy.Module):
    def __init__(
        self,
        task: Task,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
        seed=42,
        num_examples=3,
    ):
        super().__init__()
        self.feature_initializer_zero_shot = dspy.ChainOfThought(
            FeatureInitializerZeroShotSignature
        )
        self.feature_initializer_from_factors = dspy.ChainOfThought(
            FeatureInitializerWithFactorsSignature
        )
        self.feature_initalizer_combined = dspy.ChainOfThought(
            FeatureInitializerCombinedSignature
        )
        self.feature_planner = FeaturePlannerV2(task=task)
        self.feature_planner.activate_assertions(max_backtracks=5)

        self.task = task
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.rng = np.random.default_rng(seed)
        self.num_examples = num_examples

    def forward(self):
        zero_shot_initializer_result = self.feature_initializer_zero_shot(
            task=self.task.value
        )

        success_mask = self.y_train == 1

        successes = self.X_train[success_mask]
        failures = self.X_train[~success_mask]

        example_factors = []
        for _ in range(self.num_examples):
            success_eg = successes.sample(n=1, random_state=self.rng).iloc[0]
            failure_eg = failures.sample(n=1, random_state=self.rng).iloc[0]

            nct_info_success = get_clinical_trial_info_from_clinical_trials_gov_dict(
                success_eg
            )
            nct_info_failure = get_clinical_trial_info_from_clinical_trials_gov_dict(
                failure_eg
            )
            current_trial_context_success = get_detailed_nct_info(success_eg)
            current_trial_context_failure = get_detailed_nct_info(failure_eg)

            factor_analyst_success = dspy.ReAct(
                FactorAnalystSignature,
                tools=[
                    make_pubmed_search(nct_info_success),
                    make_nct_search(nct_info_success),
                ],
            )

            factor_analyst_failure = dspy.ReAct(
                FactorAnalystSignature,
                tools=[
                    make_pubmed_search(nct_info_failure),
                    make_nct_search(nct_info_failure),
                ],
            )

            factor_analyst_result_success = factor_analyst_success(
                task=self.task,
                sample_clinical_trial=current_trial_context_success,
                trial_task_label=1,
            )

            factor_analyst_result_failure = factor_analyst_failure(
                task=self.task,
                sample_clinical_trial=current_trial_context_failure,
                trial_task_label=0,
            )

            print(f"chosen: success {success_eg} failure {failure_eg}")
            print(factor_analyst_result_success)
            print(factor_analyst_result_failure)

            example_factors += factor_analyst_result_success.factors
            example_factors += factor_analyst_result_failure.factors

        with_factors_initializer_result = self.feature_initializer_from_factors(
            task=self.task.value, factors=example_factors
        )
        combined_feature_ideas = (
            zero_shot_initializer_result.feature_ideas
            + with_factors_initializer_result.feature_ideas
        )

        initializer_result_combined = self.feature_initalizer_combined(
            task=self.task.value,
            combined_feature_ideas=combined_feature_ideas,
        )

        feature_plans = {}
        for (
            feature_name,
            feature_idea,
        ) in initializer_result_combined.feature_ideas.items():
            plan, _ = self.feature_planner(
                feature_name=feature_name, feature_idea=feature_idea
            )
            feature_plans[feature_name] = plan

        return feature_plans


class AgentV2(dspy.Module):
    def __init__(self, task: Task, X_train, X_val, y_train, y_val, X_test, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.task = task
        self.initializer = Initializer(
            task=task,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
        )
        self.proposer = FeatureProposerV2(self.task)
        self.proposer.activate_assertions(max_backtracks=5)
        self.planner = FeaturePlannerV2(self.task)
        self.planner.activate_assertions(max_backtracks=5)
        self.evaluator = EvaluatorV2(self.task)
        self.evaluator.activate_assertions(max_backtracks=5)
        self.grouper = FeatureGrouper()
        self.grouper.activate_assertions(max_backtracks=5)

    def forward(self, previous_output: Optional[OutputV2] = None):
        if previous_output is None:
            current_feature_plans = self.initializer()
            current_feature_values, _ = compute_features_v3(
                self.grouper, self.X_train, self.task, current_feature_plans
            )
            current_val_feature_values, none_explanations = compute_features_v3(
                self.grouper, self.X_val, self.task, current_feature_plans
            )
            current_test_feature_values, _ = compute_features_v3(
                self.grouper, self.X_test, self.task, current_feature_plans
            )

            proposer_result: Optional[ProposerOutput] = None

        else:
            current_feature_values = deepcopy(previous_output.raw_features)
            current_val_feature_values = deepcopy(previous_output.raw_val_features)
            current_test_feature_values = deepcopy(previous_output.raw_test_features)

            current_feature_plans = deepcopy(previous_output.feature_plans)
            current_none_explanations = deepcopy(previous_output.none_explanations)

            proposer_result: Optional[ProposerOutput] = self.proposer(
                previous_output=previous_output
            )

            assert proposer_result is not None

            if (
                proposer_result.feature_operation.value == FeatureOp.ADD.value
                or proposer_result.feature_operation.value == FeatureOp.REFINE.value
            ):
                feature_idea = proposer_result.feature_explanation
                current_plan = current_feature_plans.get(proposer_result.feature_name)
                if current_plan is not None:
                    feature_idea = f"""
                    {current_plan.feature_idea}
                    ---
                    {proposer_result.feature_explanation}
                    """

                plan, _ = self.planner(
                    feature_name=proposer_result.feature_name,
                    feature_idea=feature_idea,
                )

                current_feature_plans[plan.feature_name] = plan

                new_features, _ = compute_features_v3(
                    self.grouper, self.X_train, self.task, {plan.feature_name: plan}
                )
                new_val_features, new_none_explanations = compute_features_v3(
                    self.grouper, self.X_val, self.task, {plan.feature_name: plan}
                )

                new_test_features, _ = compute_features_v3(
                    self.grouper, self.X_test, self.task, {plan.feature_name: plan}
                )

                for nctid, nf in new_features.items():
                    current_feature_values[nctid] = current_feature_values[nctid] | nf

                for nctid, nf in new_val_features.items():
                    current_val_feature_values[nctid] = (
                        current_val_feature_values[nctid] | nf
                    )

                for nctid, nf in new_test_features.items():
                    current_test_feature_values[nctid] = (
                        current_test_feature_values[nctid] | nf
                    )

                for nctid, ne in new_none_explanations.items():
                    if nctid not in current_none_explanations:
                        current_none_explanations[nctid] = {}
                    current_none_explanations[nctid] = (
                        current_none_explanations[nctid] | ne
                    )
                none_explanations = current_none_explanations

            else:
                assert proposer_result.feature_operation.value == FeatureOp.REMOVE.value
                current_feature_values = {
                    nctid: {
                        k: v
                        for k, v in features.items()
                        if k != proposer_result.feature_name
                    }
                    for nctid, features in current_feature_values.items()
                }

                current_val_feature_values = {
                    nctid: {
                        k: v
                        for k, v in features.items()
                        if k != proposer_result.feature_name
                    }
                    for nctid, features in current_val_feature_values.items()
                }

                current_test_feature_values = {
                    nctid: {
                        k: v
                        for k, v in features.items()
                        if k != proposer_result.feature_name
                    }
                    for nctid, features in current_test_feature_values.items()
                }

                del current_feature_plans[proposer_result.feature_name]  # type: ignore

                for nctid, ne in current_none_explanations.items():
                    if proposer_result.feature_name in ne:
                        del ne[proposer_result.feature_name]

                none_explanations = current_none_explanations

        df = features_to_df_v2(current_feature_values)
        val_df = features_to_df_v2(current_val_feature_values)
        test_df = features_to_df_v2(current_test_feature_values)

        xgb_model, xgb_meta = train_simple_model_v2(
            current_feature_plans, df, self.y_train, model_type="xgb"
        )
        lr_model, lr_meta = train_simple_model_v2(
            current_feature_plans, df, self.y_train, model_type="logistic"
        )
        rf_model, rf_meta = train_simple_model_v2(
            current_feature_plans, df, self.y_train, model_type="rf"
        )

        print(f"Done training -- xgb: {xgb_meta}, lr: {lr_meta}, rf: {rf_meta}")

        xgb_model_eval_out = eval(xgb_model, val_df, self.y_val, False)
        lr_model_eval_out = eval(lr_model, val_df, self.y_val, False)
        rf_model_eval_out = eval(rf_model, val_df, self.y_val, False)

        print(
            f"Done val -- xgb: {xgb_model_eval_out.roc_auc}, lr: {lr_model_eval_out.roc_auc}, rf: {rf_model_eval_out.roc_auc}"
        )

        test_xgb_model_eval_out = eval(xgb_model, test_df, self.y_test, False)
        test_lr_model_eval_out = eval(lr_model, test_df, self.y_test, False)
        test_rf_model_eval_out = eval(rf_model, test_df, self.y_test, False)

        print(
            f"Done test -- xgb: {test_xgb_model_eval_out.roc_auc}, lr: {test_lr_model_eval_out.roc_auc}, rf: {test_rf_model_eval_out.roc_auc}"
        )

        print("Running eval")

        xgb_eval_out = self.evaluator(
            feature_plans=current_feature_plans,
            model_eval_result=xgb_model_eval_out,
            none_explanations=none_explanations,
        )
        lr_eval_out = self.evaluator(
            feature_plans=current_feature_plans,
            model_eval_result=lr_model_eval_out,
            none_explanations=none_explanations,
        )
        rf_eval_out = self.evaluator(
            feature_plans=current_feature_plans,
            model_eval_result=rf_model_eval_out,
            none_explanations=none_explanations,
        )

        return OutputV2(
            xgb_eval_output=xgb_eval_out,
            lr_eval_output=lr_eval_out,
            rf_eval_output=rf_eval_out,
            test_xgb_eval_output=test_xgb_model_eval_out,
            test_lr_eval_output=test_lr_model_eval_out,
            test_rf_eval_output=test_rf_model_eval_out,
            operation=proposer_result,
            feature_plans=current_feature_plans,
            df=df,
            val_df=val_df,
            suggestion_index=0,
            raw_features=current_feature_values,
            raw_val_features=current_val_feature_values,
            raw_test_features=current_test_feature_values,
            none_explanations=none_explanations,
        )


class AgentBase(dspy.Module):
    feature_proposer_zero_shot: dspy.Module
    feature_proposer: dspy.Module
    feature_planner: dspy.Module
    feature_builder_klass: type[dspy.Module]
    evaluator: dspy.Module
    task: Task

    def __init__(self, task: Task, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.task = task

    def compute_features(self, nctids, planner_results: PlannerOutput) -> dict:
        features = {}
        for feature_name, fp in planner_results.feature_plans.items():
            features[feature_name] = tqdm(
                get_global_pool().map(
                    _WrappedMPBuilder(self.feature_builder_klass, self.task),
                    itertools.repeat(fp),
                    nctids,
                ),
                desc=fp.feature_name,
                total=len(nctids),
            )

        for k in features.keys():
            features[k] = list(v for v in features[k])

        return features

    def run_proposer(self, previous_output: Optional[Output]) -> ProposerOutput:
        if previous_output is None:
            proposer = self.feature_proposer_zero_shot
            proposer_result = proposer(task=self.task.value)
            return ProposerOutput(
                feature_operation=FeatureOp.ADD,
                feature_name=None,
                feature_explanation=proposer_result.feature_idea,
            )
        else:
            proposer = self.feature_proposer
            proposer = proposer.activate_assertions(max_backtracks=3)
            proposer_result = proposer(previous_output=previous_output)
            return ProposerOutput(
                feature_operation=proposer_result.operation,
                feature_name=proposer_result.feature_name,
                feature_explanation=proposer_result.feature_explanation,
            )

    def run_planner(
        self, proposer_result: ProposerOutput, previous_output: Optional[Output]
    ) -> PlannerOutput:
        planner = self.feature_planner
        planner = planner.activate_assertions(max_backtracks=3)

        _, plans = planner(
            proposer_result=proposer_result, previous_output=previous_output
        )

        return PlannerOutput(feature_plans=plans)

    def run_evaluator(
        self, planner_output: PlannerOutput, model_eval_result: ModelEvalResult
    ) -> EvalOutput:
        eval_result = self.evaluator(
            planner_output=planner_output,
            model_eval_result=model_eval_result,
        )
        return eval_result

    def forward(self, previous_output: Optional[Output] = None):
        proposer_result = None
        planner_result = None
        df = None
        try:
            proposer_result = self.run_proposer(previous_output=previous_output)
            print("Proposer Result", proposer_result)

            planner_result = self.run_planner(
                proposer_result=proposer_result, previous_output=previous_output
            )
            print("Planner Result", planner_result)

            features = self.compute_features(self.X_train, planner_result)
            val_features = self.compute_features(self.X_val, planner_result)

            df = features_to_df(self.X_train, features)
            val_df = features_to_df(self.X_val, val_features)

            model = train_simple_model(
                planner_result, df, self.y_train, model_type="xgb"
            )
            lr_model = train_simple_model(
                planner_result, df, self.y_train, model_type="logistic"
            )

            model_eval_out = eval(model, val_df, self.y_val, True)
            lr_model_eval_out = eval(model, val_df, self.y_val, False)

            eval_out = self.run_evaluator(
                planner_output=planner_result, model_eval_result=model_eval_out
            )
            lr_eval_out = self.run_evaluator(
                planner_output=planner_result, model_eval_result=lr_model_eval_out
            )

            return Output(
                eval_output=eval_out,
                lr_eval_output=lr_eval_out,
                proposer_output=proposer_result,
                planner_output=planner_result,
                pipeline=model,
                lr_pipeline=lr_model,
                df=df,
                val_df=val_df,
                suggestion_index=0,
                raw_features=features,
                raw_val_features=val_features,
            )
        except Exception as e:
            traceback.print_exception(e)
            traceback.print_stack()
            print("Error running agent")
            print(
                "Previous Planner",
                previous_output.planner_output if previous_output is not None else None,
            )
            print("Proposer Result", proposer_result)
            print("Planner Result", planner_result)
            if df is not None:
                df.to_csv(".logs/last_df.csv")
            raise e


class AgentV0(AgentBase):
    def __init__(self, task: Task, X_train, X_val, y_train, y_val):
        super().__init__(
            task=task, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val
        )
        self.feature_proposer_zero_shot = dspy.ChainOfThought(
            FeatureProposerZeroShotSingleOutputSignature
        )
        self.feature_proposer = FeatureProposer(task)
        self.feature_planner = FeaturePlanner(task)
        self.feature_builder_klass = FeatureBuilderV2
        self.evaluator = Evaluator(task)


def features_to_df(nctids, features_dict: dict):
    values = defaultdict(list)
    for top_feature_name, rows in features_dict.items():
        for feature_inner_dict in rows:
            for sub_feature_name, value in feature_inner_dict.items():
                values[f"{top_feature_name}--{sub_feature_name}"].append(value)
    values["id"] = list(nctids)
    return pd.DataFrame(values)


def features_to_df_v2(features: dict):
    return pd.DataFrame(
        [
            {
                "id": k,
                **(
                    {
                        f"{kk}--{kkk}": vvv
                        for kk, vv in v.items()
                        for kkk, vvv in vv.items()
                    }
                ),
            }
            for k, v in features.items()
        ]
    )


def make_bool_val_transform(fn):
    fn = deepcopy(fn)

    def transform_bool_val(input: pd.DataFrame):
        input[fn] = input[fn].apply(lambda xx: 1 if xx else 0)
        return input

    return transform_bool_val


def train_simple_model_v2(
    plans: dict[str, FeaturePlanV2],
    input_X_df,
    y_train,
    model_type: Literal["rf", "logistic", "xgb"],
    skip=list(),
):
    transformers = []
    for fp in plans.values():
        feature_name = fp.feature_name

        if feature_name in skip:
            continue

        for sub_feature_name, sub_feature_type in fp.feature_type.items():
            full_feature_name = f"{feature_name}--{sub_feature_name}"

            assert full_feature_name in input_X_df.columns

            if sub_feature_type.value == FeatureType.CATEGORICAL.value:
                transformers.append(
                    (
                        f"{full_feature_name}--cat_onehot",
                        OneHotEncoder(
                            categories=[
                                fp.possible_values[sub_feature_name],
                            ],
                            handle_unknown="ignore",
                        ),
                        [
                            full_feature_name,
                        ],
                    )
                )
            elif sub_feature_type.value == FeatureType.MULTICATEGORICAL.value:
                transformers.append(
                    (
                        f"{full_feature_name}--multicat",
                        make_pipeline(
                            FunctionTransformer(
                                lambda x: x.apply(lambda c: [] if c is None else c),
                                feature_names_out="one-to-one",
                            ),
                            CountVectorizer(
                                analyzer=lambda lst: lst,
                                vocabulary=fp.possible_values[sub_feature_name],
                                lowercase=False,
                            ),
                        ),
                        full_feature_name,
                    )
                )
            elif sub_feature_type.value == FeatureType.INTEGER.value:
                transformers.append(
                    (
                        f"{full_feature_name}--int",
                        "passthrough"
                        if model_type == "xgb"
                        else SimpleImputer(strategy="constant", fill_value=0),
                        [full_feature_name],
                    )
                )
            elif sub_feature_type.value == FeatureType.BOOLEAN.value:
                transform = make_bool_val_transform(full_feature_name)
                transformers.append(
                    (
                        f"{full_feature_name}--bool",
                        "passthrough"
                        if model_type == "xgb"
                        else make_pipeline(
                            FunctionTransformer(
                                transform,
                                feature_names_out="one-to-one",
                            ),
                            SimpleImputer(strategy="constant", fill_value=0),
                        ),
                        [full_feature_name],
                    )
                )
            else:
                transformers.append(
                    (
                        f"{full_feature_name}--passthrough",
                        "passthrough"
                        if model_type == "xgb"
                        else SimpleImputer(strategy="constant", fill_value=0.0),
                        [
                            full_feature_name,
                        ],
                    )
                )

    if model_type == "logistic":
        model = LogisticRegression()
    elif model_type == "rf":
        model = RandomForestClassifier(random_state=42)
    else:
        model = XGBClassifier()
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    try:
        pipeline.fit(input_X_df, y_train)
        return pipeline, {
            "train_roc_auc": roc_auc_score(
                y_train, pipeline.predict_proba(input_X_df)[:, 1]
            )
        }
    except Exception as e:
        print("Failed to train model", e)
        print("Input X", input_X_df)
        print("Input y", y_train)

        input_X_df.to_csv(".logs/last_df.csv")
        pipeline[0].fit_transform(input_X_df, y_train).to_csv(
            f".logs/{model_type}_last_df.csv"
        )
        raise e


def train_simple_model(
    planner_results: PlannerOutput,
    input_X_df,
    y_train,
    model_type: Literal["logistic", "xgb"],
    skip=list(),
):
    transformers = []
    for fp in planner_results.feature_plans.values():
        feature_name = fp.feature_name

        if feature_name in skip:
            continue

        for sub_feature_name, sub_feature_type in fp.feature_type.items():
            full_feature_name = f"{feature_name}--{sub_feature_name}"

            assert full_feature_name in input_X_df.columns

            if sub_feature_type.value == FeatureType.CATEGORICAL.value:
                transformers.append(
                    (
                        f"{full_feature_name}--cat_onehot",
                        OneHotEncoder(
                            categories=[
                                fp.possible_values[sub_feature_name],
                            ],
                            handle_unknown="ignore",
                        ),
                        [
                            full_feature_name,
                        ],
                    )
                )
            elif sub_feature_type.value == FeatureType.MULTICATEGORICAL.value:
                transformers.append(
                    (
                        f"{full_feature_name}--multicat",
                        make_pipeline(
                            FunctionTransformer(
                                lambda x: x.apply(
                                    lambda c: [] if c is None else json.loads(c)
                                ),
                                feature_names_out="one-to-one",
                            ),
                            CountVectorizer(
                                analyzer=lambda lst: lst,
                                vocabulary=fp.possible_values[sub_feature_name],
                                lowercase=False,
                            ),
                        ),
                        full_feature_name,
                    )
                )
            else:
                transformers.append(
                    (
                        f"{full_feature_name}--passthrough",
                        "passthrough"
                        if model_type == "xgb"
                        else SimpleImputer(strategy="constant", fill_value=0),
                        [
                            full_feature_name,
                        ],
                    )
                )

    if model_type == "logistic":
        model = LogisticRegression()
    else:
        model = XGBClassifier()
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    try:
        pipeline.fit(input_X_df, y_train)
        return pipeline
    except Exception as e:
        print("Failed to train model", e)
        print("Input X", input_X_df)
        print(preprocessor.fit_transform(input_X_df))
        raise e


def plot_multicat_feature(df, planner_result, rotate_xlabels=False):
    assert planner_result.feature_type.value == FeatureType.MULTICATEGORICAL.value

    processor = make_pipeline(
        FunctionTransformer(
            lambda x: x.apply(json.loads),
            feature_names_out="one-to-one",
        ),
        CountVectorizer(
            analyzer=lambda lst: lst,
            vocabulary=planner_result.possible_values,
            lowercase=False,
        ),
    )
    result = processor.fit_transform(df[planner_result.feature_name])
    df = (
        pd.DataFrame(
            pd.DataFrame(
                result.todense(),  # type: ignore
                columns=processor.get_feature_names_out(),
            ).sum(axis=0)
        )
        .reset_index()
        .rename(columns={"index": f"{planner_result.feature_name}", 0: "count"})
    )
    df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
    sns.barplot(data=df, y="count", x=f"{planner_result.feature_name}")
    if rotate_xlabels:
        plt.xticks(rotation=90)

    return df


def eval(pipeline, val_df, y_val, plot=False) -> ModelEvalResult:
    predictions_prob = pipeline.predict_proba(val_df)[:, 1]
    predictions = pipeline.predict(val_df)

    wrong_mask = predictions != y_val
    wrong_preds = predictions[wrong_mask].tolist()
    wrong_idxs = val_df.index[wrong_mask].to_list()
    wrong_df = val_df[wrong_mask]

    roc_auc = roc_auc_score(y_val, predictions_prob)
    f1 = f1_score(y_val, predictions)
    pr_auc = average_precision_score(y_val, predictions_prob)

    fpr, tpr, thresholds = metrics.roc_curve(y_val, predictions_prob)

    model = pipeline[-1]
    if isinstance(model, XGBClassifier) or isinstance(model, RandomForestClassifier):
        coefs = model.feature_importances_
    else:
        coefs = model.coef_[0]

    feats = pipeline[:-1].get_feature_names_out()
    assert len(coefs) == len(feats)

    sorted_feature_importances = sorted(
        zip(coefs, feats), key=lambda x: x[0], reverse=True
    )
    coefs, feats = zip(*sorted_feature_importances)

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
        sns.barplot(x=coefs, y=feats, ax=ax[0], orient="y")
        # ax[0].tick_params(axis="x", labelrotation=90)
        display = metrics.RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="model"
        )
        display.plot(ax=ax[1])

    return ModelEvalResult(
        roc_auc,  # type: ignore
        f1,  # type: ignore
        pr_auc,  # type: ignore
        sorted_feature_importances,
        wrong_idxs,
        wrong_preds,
        wrong_df,
        pipeline,
    )


def explain_feature(output: Output, feature_name: str, nctid: str):
    fp = output.planner_output.feature_plans[feature_name]

    fb = FeatureBuilderV2()
    fb = fb.activate_assertions()
    result = fb(
        feature_name=fp.feature_name,
        feature_type=fp.feature_type,
        data_sources=fp.data_sources,
        possible_values=fp.possible_values,
        plan_steps=fp.plan_steps,
        nctid=nctid,
    )
    return result


def run_agent_as_script(id, task: Task, input: Optional[Output]):
    with tempfile.NamedTemporaryFile(
        delete_on_close=False
    ) as input_file, tempfile.NamedTemporaryFile(
        delete_on_close=False
    ) as output_file, open(logfile_stderr, "w") as stderr_logfile, open(
        logfile_stdout, "w"
    ) as stdout_logfile:
        input_arg = " "
        if input is not None:
            dill.dump(input, input_file)
            input_arg = f" --input {input_file.name} "

        id_to_use = f"{task.name}--{id}"
        os.makedirs(cache_dir, exist_ok=True)

        cached_loc = f"{cache_dir}/{id_to_use}"
        if os.path.exists(cached_loc):
            print(f"Loading {id_to_use} from cache")
            with open(cached_loc, "rb") as cachef:
                return dill.load(cachef)

        output_file.close()

        try:
            subprocess.run(
                f"python scripts/run_agent.py --pid {os.getpid()} --task {task.name} {input_arg}{output_file.name}",
                shell=True,
                check=True,
                stdout=stdout_logfile,
                stderr=stderr_logfile,
                preexec_fn=os.setsid,
            )
        except subprocess.CalledProcessError as e:
            errpath = error_dir + "/" + id_to_use
            shutil.copytree("./.logs", errpath)
            raise Exception(
                f"Failed to run agent for {id_to_use}, dumped logs into {errpath}", e
            )

        with open(output_file.name, "rb") as of, open(cached_loc, "wb") as cachef:
            ret = dill.load(of)
            dill.dump(ret, cachef)
            return ret


def run_agent_as_script_v2(id, task: Task, input: Optional[OutputV2]):
    with tempfile.NamedTemporaryFile(
        delete_on_close=False
    ) as input_file, tempfile.NamedTemporaryFile(
        delete_on_close=False
    ) as output_file, open(logfile_stderr, "w") as stderr_logfile, open(
        logfile_stdout, "w"
    ) as stdout_logfile:
        input_arg = " "
        if input is not None:
            print(f"Dumping input to file {input_file.name}")
            dill.dump(input, input_file)
            input_file.flush()
            input_arg = f" --input {input_file.name} "

        id_to_use = f"{task.name}--{id}"
        os.makedirs(cache_dir, exist_ok=True)

        cached_loc = f"{cache_dir}/{id_to_use}"
        if os.path.exists(cached_loc):
            print(f"Loading {id_to_use} from cache")
            with open(cached_loc, "rb") as cachef:
                return dill.load(cachef)

        output_file.close()

        try:
            subprocess.run(
                f"python scripts/run_agent.py --pid {os.getpid()} --task {task.name} {input_arg}{output_file.name}",
                shell=True,
                check=True,
                stdout=stdout_logfile,
                stderr=stderr_logfile,
                preexec_fn=os.setsid,
            )
        except subprocess.CalledProcessError as e:
            errpath = error_dir + "/" + id_to_use
            shutil.copytree("./.logs", errpath)
            raise Exception(
                f"Failed to run agent for {id_to_use}, dumped logs into {errpath}", e
            )

        with open(output_file.name, "rb") as of, open(cached_loc, "wb") as cachef:
            ret = dill.load(of)
            dill.dump(ret, cachef)
            return ret


def load_output_v2(path: str) -> OutputV2:
    with open(path, "rb") as f:
        return dill.load(f)
