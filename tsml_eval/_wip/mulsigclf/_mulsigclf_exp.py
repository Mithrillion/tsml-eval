# %%
import numpy as np
from tsml_eval._wip.mulsigclf.mulsig_classifier import MulSigClassifier
from aeon.datasets import load_classification
from tqdm import tqdm
import warnings
import os

from tsml_eval.evaluation.storage import load_classifier_results
from tsml_eval.experiments import (
    run_classification_experiment,
)

# %%
MODE = "multivar"
# %%
# NOTE: univar:
if MODE == "univar":
    config = dict(
        depth=4,
        use_logsig=True,
        sig_mode="brackets",
        do_aug=True,
        do_time_aug=True,
        do_fourier_aug=True,
        random_state=7777,
        add_c22=False,
        add_rocket=True,
        do_rescale=False,
        scale_features=True,
        wt_levels=5,
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
    # NOTE: multivar
    config = dict(
        depth=4,
        use_logsig=True,
        sig_mode="brackets",
        do_aug=True,
        # window_alphas=[0],
        do_time_aug=True,
        do_fourier_aug=False,
        random_state=7777,
        scaler_quantile=0.05,
        add_c22=False,
        add_rocket=True,
        do_rescale=True,
        scale_features=True,
        wt_levels=4,
        dim_limit=10,
        use_kPCA=False,
        classifier="ensemble",
        # classifier="ridgecv",
        regression_params={
            "ridge": {"alphas": np.logspace(-3, 3, 10)},
        },
        ensemble_params=dict(
            use_rf2=True,
            use_ridge=False,
            rf=dict(n_estimators=100),
            logreg=dict(penalty="elasticnet", C=0.1, max_iter=1000, l1_ratio=0.15),
            logreg2=dict(penalty="l2", C=100, max_iter=1000),
            rf2=dict(
                n_estimators=100,
                max_features="log2",
                criterion="entropy",
                max_depth=5,
            ),
            # ridge=dict(
            #     alphas=np.logspace(-3, 3, 10),
            # )
        ),
    )
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
msc = MulSigClassifier(**config)
# %%
for resample in range(25, 30):
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
        with warnings.catch_warnings(action="ignore"):
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
# %%
cr = load_classifier_results(
    "./generated_results/MulSigClassifier/Predictions/ERing/testResample0.csv"
)
print(cr.predictions)
print(cr.accuracy)
print(cr.balanced_accuracy)
print(cr.auroc_score)
print(cr.log_loss)
# %%
