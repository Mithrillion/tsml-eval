import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from tsml_eval._wip.mulsigclf.utils import *
from functools import partial
from aeon.transformations.collection.feature_based import Catch22
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch.nn.functional as F
from scipy.signal import windows
import sklearn.metrics as skm


def multisigclassifier_run(
    X_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    depth: int,
    use_logsig: bool,
    sig_mode: str,
    do_aug: bool,
    do_time_aug: bool,
    random_seed: int,
    aug_type: str = "swt",
    return_raw: bool = False,
    aug_levels: int = 3,
    dim_limit: int = 10,
    use_kPCA: bool = False,
    do_rescale: bool = False,
    add_c22: bool = False,
    add_rocket: bool = False,
    window_alphas: list = [1 / 3, 2 / 3, 1],
    classifier: str = "logreg",
    use_cuda: bool = True,
    ensemble_params: dict = {},
    **clf_kwargs,
):
    def get_window(length, alpha):
        return torch.tensor(windows.tukey(length, alpha=alpha))

    if do_aug:
        # print("augmenting...")
        match aug_type:
            case "leadlag":
                td_X_og_train, td_X_og_test = to_leadlag(X_train_tensor), to_leadlag(
                    X_test_tensor
                )
            case "swt":
                td_X_og_train, td_X_og_test = (
                    swt_map(X_train_tensor.numpy(), aug_levels, "haar"),
                    swt_map(X_test_tensor.numpy(), aug_levels, "haar"),
                )
                td_X_og_train, td_X_og_test = torch.cat(
                    [X_train_tensor, diff_series(X_train_tensor)]
                    + [torch.tensor(x).float() for x in td_X_og_train],
                    -1,
                ), torch.cat(
                    [X_test_tensor, diff_series(X_test_tensor)]
                    + [torch.tensor(x).float() for x in td_X_og_test],
                    -1,
                )
            case "td":
                delays = [2**i for i in range(1, aug_levels + 1)]
                td_X_og_train = torch.cat(
                    [X_train_tensor, diff_series(X_train_tensor)]
                    + [
                        F.pad(X_train_tensor, (0, 0, 0, d), "replicate").roll(d, 1)[
                            :, :-d, :
                        ]
                        for d in delays
                    ],
                    -1,
                )
                td_X_og_test = torch.cat(
                    [X_test_tensor, diff_series(X_test_tensor)]
                    + [
                        F.pad(X_test_tensor, (0, 0, 0, d), "replicate").roll(d, 1)[
                            :, :-d, :
                        ]
                        for d in delays
                    ],
                    -1,
                )
            case _:
                raise NotImplementedError(
                    f"Augmentation type {aug_type} not implemented"
                )

    else:
        td_X_og_train, td_X_og_test = X_train_tensor, X_test_tensor

    if td_X_og_train.shape[-1] > dim_limit:
        if not use_kPCA:
            td_X_og_train, pca = to_td_pca(
                td_X_og_train, dim_limit, 1, 1, random_state=random_seed
            )
            td_X_og_test, _ = to_td_pca(
                td_X_og_test,
                dim_limit,
                1,
                1,
                pretrained_pca=pca,
                random_state=random_seed,
            )
        else:
            kpca = KernelPCA(
                dim_limit,
                kernel="rbf",
                random_state=random_seed,
                fit_inverse_transform=False,
                n_jobs=-1,
            )
            td_X_og_train = torch.tensor(
                kpca.fit_transform(td_X_og_train.flatten(0, 1))
            ).view(*td_X_og_train.shape[:2], -1)
            td_X_og_test = torch.tensor(
                kpca.transform(td_X_og_test.flatten(0, 1))
            ).view(*td_X_og_test.shape[:2], -1)

    multi_series_train = [td_X_og_train] + [
        get_window(td_X_og_train.shape[1], alpha)[None, :, None].float() * td_X_og_train
        for alpha in window_alphas
    ]
    multi_series_test = [td_X_og_test] + [
        get_window(td_X_og_test.shape[1], alpha)[None, :, None].float() * td_X_og_test
        for alpha in window_alphas
    ]

    if use_logsig:
        sig_map = partial(signatory.logsignature, mode=sig_mode, depth=depth)
        base_sig_map = partial(
            signatory.logsignature, mode=sig_mode, basepoint=True, depth=depth
        )
    else:
        sig_map = partial(signatory.signature, depth=depth)
        base_sig_map = partial(signatory.signature, basepoint=True, depth=depth)

    td_X_trains, td_X_tests = [], []
    og_iter = True
    # print("computing sigs...")
    for td_X_train, td_X_test in zip(multi_series_train, multi_series_test):
        if do_time_aug:
            td_X_train, td_X_test = time_aug(td_X_train, (0, 1)), time_aug(
                td_X_test, (0, 1)
            )

        if og_iter:
            sigs_train_base = base_sig_map(
                rescale_path(td_X_train, depth if do_rescale else 1)
            )
            sigs_test_base = base_sig_map(
                rescale_path(td_X_test, depth if do_rescale else 1)
            )
            sigs_train_base_normed, sigs_test_base_normed = unorm(
                sigs_train_base
            ), unorm(sigs_test_base)
            td_X_trains.append(sigs_train_base_normed)
            td_X_tests.append(sigs_test_base_normed)
            og_iter = False

        sigs_train = sig_map(rescale_path(td_X_train, depth if do_rescale else 1))
        sigs_test = sig_map(rescale_path(td_X_test, depth if do_rescale else 1))

        sigs_train_normed, sigs_test_normed = unorm(sigs_train), unorm(sigs_test)
        td_X_trains.append(sigs_train_normed)
        td_X_tests.append(sigs_test_normed)

    if add_c22:
        c22 = Catch22(n_jobs=-1, replace_nans=True)
        c22.fit(X_train_tensor)
        c22_train = c22.transform(X_train_tensor.permute(0, 2, 1).numpy())
        c22_test = c22.transform(X_test_tensor.permute(0, 2, 1).numpy())
        td_X_trains.append(torch.tensor(c22_train.values).float())
        td_X_tests.append(torch.tensor(c22_test.values).float())

    if add_rocket:
        rocket = MiniRocket(
            n_kernels=10000, random_state=random_seed, n_jobs=-1
        )
        rocket.fit(pad_if_short(X_train_tensor).permute(0, 2, 1).numpy())
        rocket_train = rocket.transform(
            pad_if_short(X_train_tensor).permute(0, 2, 1).numpy()
        )
        rocket_test = rocket.transform(
            pad_if_short(X_test_tensor).permute(0, 2, 1).numpy()
        )
        td_X_trains.append(torch.tensor(rocket_train.values).float())
        td_X_tests.append(torch.tensor(rocket_test.values).float())

    feats_train = torch.cat(td_X_trains, -1)
    feats_test = torch.cat(td_X_tests, -1)

    # print(f"fitting model of feature size {feats_train.shape[-1]}...")
    match classifier:
        case "rf":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="For reproducible results in Random Forest Classifier",
                )
                clf = RandomForestClassifier(random_state=random_seed, **clf_kwargs)
                clf.fit(feats_train.numpy(), y_train)
                y_train_pred = clf.predict(feats_train.numpy())
                y_pred = clf.predict(feats_test.numpy())
                y_train_pred_proba = clf.predict_proba(feats_train.numpy())
                y_pred_proba = clf.predict_proba(feats_test.numpy())
        case "logreg":
            clf = LogisticRegression(**clf_kwargs)
            scaler = StandardScaler()
            feats_train = scaler.fit_transform(feats_train)
            feats_test = scaler.transform(feats_test)
            clf.fit(feats_train, y_train)
            y_train_pred = clf.predict(feats_train)
            y_pred = clf.predict(feats_test)
            y_train_pred_proba = clf.predict_proba(feats_train)
            y_pred_proba = clf.predict_proba(feats_test)
        case "calibratedridgecv":
            min_samples_per_class = np.bincount(y_train).min()
            if min_samples_per_class > 2:
                clf = CalibratedClassifierCV(
                    estimator=RidgeClassifierCV(
                        **clf_kwargs, cv=min(5, min_samples_per_class - 1)
                    ),
                    method="sigmoid",
                    cv=min(5, min_samples_per_class),
                    n_jobs=-1,
                    ensemble=False,
                )
            else:
                # fall back to RF if not enough samples for RidgeCV (having single instance in a class)
                clf = RandomForestClassifier(random_state=random_seed, n_estimators=100)
            scaler = StandardScaler()
            feats_train = scaler.fit_transform(feats_train)
            feats_test = scaler.transform(feats_test)
            clf.fit(feats_train, y_train)
            y_train_pred = clf.predict(feats_train)
            y_pred = clf.predict(feats_test)
            y_train_pred_proba = clf.predict_proba(feats_train)
            y_pred_proba = clf.predict_proba(feats_test)
        case "small_ensemble":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="For reproducible results in Random Forest Classifier",
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(feats_train)
                X_test = scaler.transform(feats_test)
                clf_1 = RandomForestClassifier(
                    random_state=random_seed, n_jobs=-1, **ensemble_params["rf"]
                )
                clf_2 = LogisticRegression(**ensemble_params["logreg"])
                clf_meta = LogisticRegression(C=1, max_iter=100)
                min_samples_per_class = np.bincount(y_train).min()
                if min_samples_per_class > 1:
                    ens_clfs = [("rf", clf_1), ("lr", clf_2)]
                    clf = StackingClassifier(
                        ens_clfs,
                        final_estimator=clf_meta,
                        cv=min(5, min_samples_per_class),
                    )
                else:
                    clf = VotingClassifier(
                        [("rf", clf_1), ("lr", clf_2)], voting="soft"
                    )
                clf.fit(X_train, y_train)
                y_train_pred = clf.predict(X_train)
                y_pred = clf.predict(X_test)
                y_train_pred_proba = clf.predict_proba(X_train)
                y_pred_proba = clf.predict_proba(X_test)
        case "ensemble":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="For reproducible results in Random Forest Classifier",
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(feats_train)
                X_test = scaler.transform(feats_test)
                if ensemble_params.get("use_rf2", True):
                    clf_4 = RandomForestClassifier(
                        random_state=random_seed, n_jobs=-1, **ensemble_params["rf2"]
                    )
                clf_1 = RandomForestClassifier(
                    random_state=random_seed, n_jobs=-1, **ensemble_params["rf"]
                )
                clf_2 = LogisticRegression(**ensemble_params["logreg"])
                clf_3 = LogisticRegression(**ensemble_params["logreg2"])

                clf_meta = LogisticRegression(C=1, max_iter=100)
                min_samples_per_class = np.bincount(y_train).min()
                if min_samples_per_class > 1:
                    ens_clfs = [("rf", clf_1), ("lr", clf_2), ("lr2", clf_3)]
                    if ensemble_params.get("use_rf2", True):
                        ens_clfs.append(("rf2", clf_4))
                    clf = StackingClassifier(
                        ens_clfs,
                        final_estimator=clf_meta,
                        cv=min(5, min_samples_per_class),
                    )
                else:
                    clf = VotingClassifier(
                        [("rf", clf_1), ("lr", clf_2)], voting="soft"
                    )
                clf.fit(X_train, y_train)
                y_train_pred = clf.predict(X_train)
                y_pred = clf.predict(X_test)
                y_train_pred_proba = clf.predict_proba(X_train)
                y_pred_proba = clf.predict_proba(X_test)
        case _:
            raise NotImplementedError(f"Classifier {classifier} not implemented")

    if not return_raw:
        train_acc = skm.accuracy_score(y_train, y_train_pred)
        acc = skm.accuracy_score(y_test, y_pred)
        f1 = skm.f1_score(y_test, y_pred, average="macro")
        return train_acc, acc, f1
    else:
        return y_train_pred, y_pred, y_train_pred_proba, y_pred_proba


def minirocket_run(
    X_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_seed: int,
    return_raw: bool = False,
    **clf_kwargs,
):
    rocket = MiniRocket(
        n_kernels=10000, random_state=random_seed, n_jobs=-1
    )
    rocket.fit(pad_if_short(X_train_tensor).permute(0, 2, 1).numpy())
    rocket_train = rocket.transform(
        pad_if_short(X_train_tensor).permute(0, 2, 1).numpy()
    )
    rocket_test = rocket.transform(pad_if_short(X_test_tensor).permute(0, 2, 1).numpy())

    feats_train = torch.tensor(rocket_train.values).float()
    feats_test = torch.tensor(rocket_test.values).float()

    clf = LogisticRegression(**clf_kwargs)
    scaler = StandardScaler()
    feats_train = scaler.fit_transform(feats_train)
    feats_test = scaler.transform(feats_test)

    clf.fit(feats_train, y_train)
    y_train_pred = clf.predict(feats_train)
    y_pred = clf.predict(feats_test)

    if not return_raw:
        train_acc = skm.accuracy_score(y_train, y_train_pred)
        acc = skm.accuracy_score(y_test, y_pred)
        f1 = skm.f1_score(y_test, y_pred, average="macro")
        return train_acc, acc, f1
    else:
        return y_pred


def multisigclassifier_features(
    X_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    depth: int,
    use_logsig: bool,
    sig_mode: str,
    do_aug: bool,
    do_time_aug: bool,
    random_seed: int,
    wt_levels: int = 3,
    dim_limit: int = 10,
    add_c22: bool = False,
    add_rocket: bool = False,
):
    def get_window(length, alpha):
        return torch.tensor(windows.tukey(length, alpha=alpha))

    if do_aug:
        # print("augmenting...")
        # td_X_og_train, td_X_og_test = to_leadlag(X_train_tensor), to_leadlag(
        #     X_test_tensor
        # )

        td_X_og_train, td_X_og_test = (
            swt_map(X_train_tensor.numpy(), wt_levels, "haar"),
            swt_map(X_test_tensor.numpy(), wt_levels, "haar"),
        )
        td_X_og_train, td_X_og_test = torch.cat(
            [X_train_tensor, diff_series(X_train_tensor)]
            + [torch.tensor(x).float() for x in td_X_og_train],
            -1,
        ), torch.cat(
            [X_test_tensor, diff_series(X_test_tensor)]
            + [torch.tensor(x).float() for x in td_X_og_test],
            -1,
        )
    else:
        td_X_og_train, td_X_og_test = X_train_tensor, X_test_tensor

    if td_X_og_train.shape[-1] > dim_limit:
        td_X_og_train, pca = to_td_pca(
            td_X_og_train, dim_limit, 1, 1, random_state=random_seed
        )
        td_X_og_test, _ = to_td_pca(
            td_X_og_test, dim_limit, 1, 1, pretrained_pca=pca, random_state=random_seed
        )

    multi_series_train = [td_X_og_train] + [
        get_window(td_X_og_train.shape[1], alpha)[None, :, None].float() * td_X_og_train
        for alpha in [1 / 3, 2 / 3, 1]
    ]
    multi_series_test = [td_X_og_test] + [
        get_window(td_X_og_test.shape[1], alpha)[None, :, None].float() * td_X_og_test
        for alpha in [1 / 3, 2 / 3, 1]
    ]

    if use_logsig:
        sig_map = partial(signatory.logsignature, mode=sig_mode, depth=depth)
        base_sig_map = partial(
            signatory.logsignature, mode=sig_mode, basepoint=True, depth=depth
        )
    else:
        sig_map = partial(signatory.signature, depth=depth)
        base_sig_map = partial(signatory.signature, basepoint=True, depth=depth)

    td_X_trains, td_X_tests = [], []
    og_iter = True
    # print("computing sigs...")
    for td_X_train, td_X_test in zip(multi_series_train, multi_series_test):
        if do_time_aug:
            td_X_train, td_X_test = time_aug(td_X_train, (0, 1)), time_aug(
                td_X_test, (0, 1)
            )

        if og_iter:
            sigs_train_base = base_sig_map(rescale_path(td_X_train, depth))
            sigs_test_base = base_sig_map(rescale_path(td_X_test, depth))
            td_X_trains.append(sigs_train_base)
            td_X_tests.append(sigs_test_base)
            og_iter = False

        sigs_train = sig_map(rescale_path(td_X_train, depth))
        sigs_test = sig_map(rescale_path(td_X_test, depth))

        sigs_train_normed, sigs_test_normed = unorm(sigs_train), unorm(sigs_test)
        td_X_trains.append(sigs_train_normed)
        td_X_tests.append(sigs_test_normed)

    if add_c22:
        c22 = Catch22(n_jobs=-1, replace_nans=True)
        c22.fit(X_train_tensor)
        c22_train = c22.transform(X_train_tensor.permute(0, 2, 1).numpy())
        c22_test = c22.transform(X_test_tensor.permute(0, 2, 1).numpy())
        td_X_trains.append(torch.tensor(c22_train.values).float())
        td_X_tests.append(torch.tensor(c22_test.values).float())

    if add_rocket:
        if X_train_tensor.shape[-1] == 1:
            rocket = MiniRocket(num_kernels=10000, random_state=random_seed, n_jobs=-1)
        else:
            rocket = MiniRocket(
                n_kernels=10000, random_state=random_seed, n_jobs=-1
            )
        rocket.fit(pad_if_short(X_train_tensor).permute(0, 2, 1).numpy())
        rocket_train = rocket.transform(
            pad_if_short(X_train_tensor).permute(0, 2, 1).numpy()
            if X_train_tensor.shape[-1] != 1
            else pad_if_short(X_train_tensor).numpy()[..., 0]
        )
        rocket_test = rocket.transform(
            pad_if_short(X_test_tensor).permute(0, 2, 1).numpy()
            if X_test_tensor.shape[-1] != 1
            else pad_if_short(X_test_tensor).numpy()[..., 0]
        )
        td_X_trains.append(torch.tensor(rocket_train.values).float())
        td_X_tests.append(torch.tensor(rocket_test.values).float())

    feats_train = torch.cat(td_X_trains, -1)
    feats_test = torch.cat(td_X_tests, -1)

    return feats_train, feats_test


def multisigclassifier_eval(
    feats_train: torch.Tensor,
    feats_test: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    return_raw: bool = False,
    **clf_kwargs,
):
    # print(f"fitting model of feature size {feats_train.shape[-1]}...")
    # clf = cuRF(
    #     random_state=random_seed, n_estimators=n_estimators, **rf_kwargs
    # )
    clf = LogisticRegression(**clf_kwargs)
    scaler = StandardScaler()
    feats_train = scaler.fit_transform(feats_train)
    feats_test = scaler.transform(feats_test)
    # TODO: move scaling to the feature outputs

    clf.fit(feats_train, y_train)
    y_train_pred = clf.predict(feats_train)
    y_pred = clf.predict(feats_test)

    if not return_raw:
        train_acc = skm.accuracy_score(y_train, y_train_pred)
        acc = skm.accuracy_score(y_test, y_pred)
        f1 = skm.f1_score(y_test, y_pred, average="macro")
        return train_acc, acc, f1
    else:
        return y_pred


def msc_objective_fn(
    trial,
    feats_train,
    feats_test,
    y_train,
    y_test,
    clf_kwargs,
):
    acc, f1, _ = multisigclassifier_eval(
        feats_train,
        feats_test,
        y_train,
        y_test,
        **(trial.params if len(trial.params) > 0 else trial._params),
        **clf_kwargs,
    )

    return f1, acc