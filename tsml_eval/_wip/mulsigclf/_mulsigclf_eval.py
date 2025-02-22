# %%
import pickle

import matplotlib.pyplot as plt
import os

from tsml_eval.evaluation import (
    evaluate_classifiers_by_problem,
    evaluate_classifiers_from_file,
    evaluate_classifiers
)

classifiers = ["MulSigClassifier"]
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
# %%
evaluate_classifiers_by_problem(
    "./generated_results/",
    classifiers,
    datasets,
    "./generated_evals/",
    resamples=20,
    eval_name="ExampleEval",
)

# %%
