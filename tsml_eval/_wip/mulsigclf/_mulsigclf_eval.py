# %%
import pickle

import matplotlib.pyplot as plt
import os

from tsml_eval.evaluation import (
    evaluate_classifiers_by_problem,
    evaluate_classifiers_from_file,
    evaluate_classifiers,
)

classifiers = ["MulSigClassifier", "HydraClassifier", "MiniRocketClassifier"]
datasets = [
    "ArticularyWordRecognition",
    "AsphaltObstacles",
    "AsphaltPavementTypeCoordinates",
    "AsphaltRegularityCoordinates",
    "AtrialFibrillation",
    "BasicMotions",
    "Blink",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "EMOPain",
    "Epilepsy",
    "EthanolConcentration",
    "EyesOpenShut",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    # "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MindReading",
    "MotorImagery",
    "MotionSenseHAR",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "Siemens",
    "SpokenArabicDigits",
    "StandWalkJump",
    # "Tiselac",
    "UWaveGestureLibrary",
]
# %%
evaluate_classifiers_by_problem(
    "./generated_results/",
    classifiers,
    datasets,
    "./generated_evals/",
    resamples=30,
    eval_name="ExampleEval",
)

# %%
