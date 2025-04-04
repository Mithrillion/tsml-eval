# %%
import numpy as np
from aeon.datasets import load_classification
from aeon.classification.convolution_based import (
    MiniRocketClassifier,
    MultiRocketHydraClassifier,
    HydraClassifier,
    MultiRocketClassifier,
)
from aeon.classification.hybrid import RISTClassifier
from aeon.transformations.collection import Padder
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import DrCIFClassifier
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import warnings
import os

from tsml_eval.evaluation.storage import load_classifier_results
from tsml_eval.experiments import (
    run_classification_experiment,
)


# %%
def pad_if_short(x, min_len=9):
    if x.shape[1] < min_len:
        return np.pad(x, ((0, 0), (0, min_len - x.shape[1]), (0, 0)), mode="constant")
    else:
        return x


# %%
MODE = "multivar"
# %%
# NOTE: univar:
if MODE == "univar":
    datasets = [
        "ACSF1",
        "Adiac",
        "ArrowHead",
        "Beef",
        "BeetleFly",
        "BirdChicken",
        "BME",
        "Car",
        "CBF",
        "Chinatown",
        "ChlorineConcentration",
        "CinCECGTorso",
        "Coffee",
        "Computers",
        "CricketX",
        "CricketY",
        "CricketZ",
        "Crop",
        "DiatomSizeReduction",
        "DistalPhalanxOutlineAgeGroup",
        "DistalPhalanxOutlineCorrect",
        "DistalPhalanxTW",
        "Earthquakes",
        "ECG200",
        "ECG5000",
        "ECGFiveDays",
        "ElectricDevices",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "EthanolLevel",
        "FaceAll",
        "FaceFour",
        "FacesUCR",
        "FiftyWords",
        "Fish",
        "FordA",
        "FordB",
        "FreezerRegularTrain",
        "FreezerSmallTrain",
        "GunPoint",
        "GunPointAgeSpan",
        "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung",
        "Ham",
        "HandOutlines",
        "Haptics",
        "Herring",
        "HouseTwenty",
        "InlineSkate",
        "InsectEPGRegularTrain",
        "InsectEPGSmallTrain",
        "InsectWingbeatSound",
        "ItalyPowerDemand",
        "LargeKitchenAppliances",
        "Lightning2",
        "Lightning7",
        "Mallat",
        "Meat",
        "MedicalImages",
        "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxOutlineCorrect",
        "MiddlePhalanxTW",
        "MixedShapesRegularTrain",
        "MixedShapesSmallTrain",
        "MoteStrain",
        "NonInvasiveFetalECGThorax1",
        "NonInvasiveFetalECGThorax2",
        "OliveOil",
        "OSULeaf",
        "PhalangesOutlinesCorrect",
        "Phoneme",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
        "Plane",
        "PowerCons",
        "ProximalPhalanxOutlineAgeGroup",
        "ProximalPhalanxOutlineCorrect",
        "ProximalPhalanxTW",
        "RefrigerationDevices",
        "Rock",
        "ScreenType",
        "SemgHandGenderCh2",
        "SemgHandMovementCh2",
        "SemgHandSubjectCh2",
        "ShapeletSim",
        "ShapesAll",
        "SmallKitchenAppliances",
        "SmoothSubspace",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "StarLightCurves",
        "Strawberry",
        "SwedishLeaf",
        "Symbols",
        "SyntheticControl",
        "ToeSegmentation1",
        "ToeSegmentation2",
        "Trace",
        "TwoLeadECG",
        "TwoPatterns",
        "UMD",
        "UWaveGestureLibraryAll",
        "UWaveGestureLibraryX",
        "UWaveGestureLibraryY",
        "UWaveGestureLibraryZ",
        "Wafer",
        "Wine",
        "WordSynonyms",
        "Worms",
        "WormsTwoClass",
        "Yoga",
    ]
    extract_path = "/workspace/test_uni_data/"
elif MODE == "multivar":
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
    extract_path = "/workspace/test_data/"
# %%
mrc = MiniRocketClassifier(n_jobs=16, random_state=7777)
mrhc = MultiRocketHydraClassifier(n_jobs=16, random_state=7777)
hrc = HydraClassifier(n_jobs=16, random_state=7777)
ristc = RISTClassifier(n_jobs=16, random_state=7777)
drcif = DrCIFClassifier(n_jobs=16, random_state=7777)
c22 = Catch22Classifier(n_jobs=16, random_state=7777)
# %%
for resample in range(30):
    for dataset in tqdm(datasets):
        X_train, y_train = load_classification(
            dataset,
            split="train",
            extract_path=extract_path,
            load_equal_length=True,
            load_no_missing=True,
        )
        X_test, y_test = load_classification(
            dataset,
            split="test",
            extract_path=extract_path,
            load_equal_length=True,
            load_no_missing=True,
        )
        X_train = np.transpose(
            pad_if_short(np.transpose(X_train, (0, 2, 1))), (0, 2, 1)
        )
        X_test = np.transpose(pad_if_short(np.transpose(X_test, (0, 2, 1))), (0, 2, 1))
        with warnings.catch_warnings(action="ignore"):
            run_classification_experiment(
                X_train,
                y_train,
                X_test,
                y_test,
                c22,
                "./generated_results/",
                dataset_name=dataset,
                resample_id=resample,
            )
# %%
