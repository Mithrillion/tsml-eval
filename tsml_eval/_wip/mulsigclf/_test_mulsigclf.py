from tsml_eval._wip.mulsigclf.mulsig_classifier import MulSigClassifier
from aeon.datasets import tsc_datasets, load_italy_power_demand
from sklearn.metrics import classification_report

IPD_X_train, IPD_y_train = load_italy_power_demand(split="train")
IPD_X_test, IPD_y_test = load_italy_power_demand(split="test")
aeon_file_name = ""

config = dict(
    depth=4,
    use_logsig=True,
    sig_mode="brackets",
    do_aug=True,
    do_time_aug=True,
    random_state=7777,
    add_c22=False,
    add_rocket=True,
    do_rescale=False,
    wt_levels=5,
    dim_limit=10,
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
IPD_y_train_pred = msc.fit_predict(IPD_X_train, IPD_y_train)
IPD_y_test_pred = msc.predict(IPD_X_test)

print(classification_report(IPD_y_train, IPD_y_train_pred))
print(classification_report(IPD_y_test, IPD_y_test_pred))