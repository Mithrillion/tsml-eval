# %%
import numpy as np
import pandas as pd
from tsml_eval._wip.mulsigclf.mulsig_classifier import MulSigClassifier
from aeon.benchmarking.results_loaders import get_estimator_results
from aeon.classification import DummyClassifier
from aeon.datasets import load_classification
from aeon.visualisation import plot_critical_difference
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tsml_eval.evaluation.storage import load_classifier_results, ClassifierResults
from tsml_eval.experiments import (
    experiments,
    get_classifier_by_name,
    run_classification_experiment,
    load_and_run_classification_experiment,
)

# %%
# NOTE: univar:
# config = dict(
#     depth=4,
#     use_logsig=True,
#     sig_mode="brackets",
#     do_aug=True,
#     do_time_aug=True,
#     random_state=7777,
#     add_c22=False,
#     add_rocket=True,
#     do_rescale=False,
#     wt_levels=5,
#     dim_limit=10,
#     classifier="ensemble",
#     ensemble_params=dict(
#         use_rf2=True,
#         rf=dict(n_estimators=100),
#         logreg=dict(penalty="elasticnet", C=0.1, max_iter=1000, l1_ratio=0.15),
#         logreg2=dict(penalty="l2", C=100, max_iter=1000),
#         rf2=dict(
#             n_estimators=100,
#             max_features="log2",
#             criterion="entropy",
#             max_depth=5,
#         ),
#     ),
# )

# NOTE: multivar
config = dict(
    depth=4,
    use_logsig=True,
    sig_mode="brackets",
    do_aug=True,
    # window_alphas=[0],
    do_time_aug=True,
    random_state=7777,
    add_c22=False,
    add_rocket=True,
    do_rescale=False,
    wt_levels=4,
    dim_limit=10,
    use_kPCA=False,
    classifier="ensemble",
    ensemble_params=dict(
        use_rf2=True,
        rf=dict(n_estimators=100),
        logreg=dict(penalty="elasticnet", C=0.1, max_iter=1000, l1_ratio=0.15),
        logreg2=dict(penalty="l2", C=100, max_iter=1000),
        rf2=dict(
            n_estimators=100,
            max_features="log2",
            criterion="entropy",
            max_depth=5,
        ),
    ),
)
msc = MulSigClassifier(**config)

# %%
datasets = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "ERing",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "LSST",
    "Libras",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
]
extract_path = "/mnt/Nova/source_repos/tsml-eval/test_datasets/"
for resample in range(30):
    for dataset in tqdm(datasets):
        X_train, y_train = load_classification(
            dataset, split="train", extract_path=extract_path
        )
        X_test, y_test = load_classification(
            dataset, split="test", extract_path=extract_path
        )

        run_classification_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
            msc,
            "./generated_results/",
            dataset_name=dataset,
            resample_id=resample,
        )