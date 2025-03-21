__maintainer__ = ["mithrillion"]
__all__ = ["MulSigClassifier"]

import warnings
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from aeon.classification.base import BaseClassifier
from cuml import LogisticRegression
from aeon.transformations.collection.feature_based import Catch22
from aeon.transformations.collection.convolution_based import MiniRocket
from tsml_eval._wip.mulsigclf.utils import *
from tsml_eval._wip.mulsigclf.mulsig_features import MulSigTransformer


class RidgeClassifierCVExt(RidgeClassifierCV):
    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        store_cv_values=False,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring,
            cv=cv,
            store_cv_values=store_cv_values,
            class_weight=class_weight,
        )

    def predict_proba(self, X) -> np.ndarray:
        dists = np.zeros((X.shape[0], len(self.classes_)))
        preds = self.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1
        return dists


class MulSigClassifier(BaseClassifier):
    _tags = {
        "capability:multithreading": True,
        "capability:multivariate": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        depth: int = 3,
        use_logsig: bool = True,
        sig_mode: str = "words",
        do_aug: bool = True,
        do_time_aug: bool = True,
        do_fourier_aug: bool = True,
        random_state: int = None,
        scaler_quantile: float = 0.05,
        wt_levels: int = 3,
        dim_limit: int = 8,
        use_kPCA: bool = False,
        window_alphas: tuple = (1 / 3, 2 / 3, 1),
        do_rescale: bool = False,
        scale_features: bool = True,
        add_c22: bool = False,
        add_rocket: bool = False,
        classifier: str = "logreg",
        n_jobs: int = -1,
        ensemble_params: dict = {},
        regression_params: dict = {},
    ):
        super(MulSigClassifier, self).__init__()
        self.depth = depth
        self.use_logsig = use_logsig
        self.sig_mode = sig_mode
        self.do_aug = do_aug
        self.do_fourier_aug = do_fourier_aug
        self.do_time_aug = do_time_aug
        self.random_state = random_state
        self.scaler_quantile = scaler_quantile
        self.wt_levels = wt_levels
        self.dim_limit = dim_limit
        self.use_kPCA = use_kPCA
        self.window_alphas = window_alphas
        self.do_rescale = do_rescale
        self.scale_features = scale_features
        self.add_c22 = add_c22
        self.add_rocket = add_rocket
        self.classifier = classifier
        self.n_jobs = n_jobs
        self.ensemble_params = ensemble_params if ensemble_params is not None else {}
        self.regression_params = (
            regression_params if regression_params is not None else {}
        )
        self._transformer = MulSigTransformer(
            depth=depth,
            use_logsig=use_logsig,
            sig_mode=sig_mode,
            do_aug=do_aug,
            do_time_aug=do_time_aug,
            do_fourier_aug=do_fourier_aug,
            random_state=random_state,
            wt_levels=wt_levels,
            dim_limit=dim_limit,
            use_kPCA=use_kPCA,
            window_alphas=window_alphas,
            do_rescale=do_rescale,
            scale_features=scale_features,
            n_jobs=n_jobs,
        )
        self._c22 = None
        self._rocket = None
        self._clf = None
        self._scaler_lower = None
        self._scaler_upper = None

    def _transform_data(self, X: np.ndarray, random_state: np.int32):
        # transform features
        self._scaler_lower = np.nanquantile(X, self.scaler_quantile)
        self._scaler_upper = np.nanquantile(X, 1 - self.scaler_quantile)
        X = (X - self._scaler_lower) / (self._scaler_upper - self._scaler_lower)
        X_feats = self._transformer.fit_transform(X)

        X_aeon = np.transpose(pad_if_short(np.transpose(X, (0, 2, 1))), (0, 2, 1))

        if self.add_c22:
            self._c22 = Catch22(n_jobs=self.n_jobs, replace_nans=True)
            c22_feats = self._c22.fit_transform(X_aeon).astype(np.float32)
            X_feats = np.concatenate([X_feats, c22_feats], axis=-1)

        if self.add_rocket:
            self._rocket = MiniRocket(
                n_kernels=10000, random_state=random_state, n_jobs=self.n_jobs
            )
            rocket_feats = self._rocket.fit_transform(X_aeon)
            X_feats = np.concatenate([X_feats, rocket_feats], axis=-1)
        return X_feats

    def _apply_transform(self, X: np.ndarray):
        X = (X - self._scaler_lower) / (self._scaler_upper - self._scaler_lower)
        X_feats = self._transformer.transform(X)
        X_aeon = np.transpose(pad_if_short(np.transpose(X, (0, 2, 1))), (0, 2, 1))
        if self.add_c22:
            c22_feats = self._c22.transform(X_aeon).astype(np.float32)
            X_feats = np.concatenate([X_feats, c22_feats], axis=-1)

        if self.add_rocket:
            rocket_feats = self._rocket.transform(X_aeon)
            X_feats = np.concatenate([X_feats, rocket_feats], axis=-1)
        return X_feats

    def _do_fit(self, X: np.ndarray, y: np.ndarray, return_preds: bool = False):
        random_state = (
            np.int32(self.random_state) if isinstance(self.random_state, int) else None
        )

        # transform features
        X_feats = self._transform_data(X, random_state)

        if "rf" in self.regression_params:
            rf_kwargs = self.regression_params["rf"]
        if "logreg" in self.regression_params:
            logreg_kwargs = self.regression_params["logreg"]
        if "ridge" in self.regression_params:
            ridge_kwargs = self.regression_params["ridge"]

        match self.classifier:
            case "rf":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="For reproducible results in Random Forest Classifier",
                    )
                    self._clf = RandomForestClassifier(
                        random_state=random_state, **rf_kwargs
                    )
                    self._clf.fit(X_feats, y)
            case "logreg":
                clf = LogisticRegression(**logreg_kwargs, verbose=2)
                self._clf = clf
                self._clf.fit(X_feats, y)
            case "ridgecv":
                min_samples_per_class = np.min(
                    [np.sum(y == label) for label in np.unique(y)]
                )
                if min_samples_per_class > 2:
                    clf = RidgeClassifierCVExt(
                        **ridge_kwargs, cv=min(5, min_samples_per_class - 1)
                    )
                else:
                    # fall back to RF if not enough samples for RidgeCV (having single instance in a class)
                    clf = RandomForestClassifier(
                        random_state=random_state, n_estimators=100
                    )
                self._clf = clf
                self._clf.fit(X_feats, y)
            case "ensemble":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="For reproducible results in Random Forest Classifier",
                    )
                    clf_1 = RandomForestClassifier(
                        random_state=random_state,
                        n_jobs=self.n_jobs,
                        **self.ensemble_params["rf"],
                    )
                    clf_2 = LogisticRegression(
                        **self.ensemble_params["logreg"], verbose=2
                    )
                    clf_3 = LogisticRegression(
                        **self.ensemble_params["logreg2"], verbose=2
                    )
                    if self.ensemble_params.get("use_rf2", True):
                        clf_4 = RandomForestClassifier(
                            random_state=random_state,
                            n_jobs=self.n_jobs,
                            **self.ensemble_params["rf2"],
                        )
                    if self.ensemble_params.get("use_ridge", True):
                        clf_5 = RidgeClassifierCV(**self.ensemble_params["ridge"])
                    clf_meta = LogisticRegression(C=1, max_iter=100, verbose=2)
                    min_samples_per_class = np.min(
                        [np.sum(y == label) for label in np.unique(y)]
                    )
                    if min_samples_per_class > 1:
                        ens_clfs = [("rf", clf_1), ("lr", clf_2), ("lr2", clf_3)]
                        if self.ensemble_params.get("use_rf2", True):
                            ens_clfs.append(("rf2", clf_4))
                        if self.ensemble_params.get("use_ridge", True):
                            ens_clfs.append(("ridge", clf_5))
                        clf = StackingClassifier(
                            ens_clfs,
                            final_estimator=clf_meta,
                            cv=min(5, min_samples_per_class),
                        )
                    else:
                        clf = VotingClassifier(
                            [("rf", clf_1), ("lr", clf_2)], voting="soft"
                        )
                    self._clf = clf
                    self._clf.fit(X_feats, y)
            case _:
                raise NotImplementedError(
                    f"Classifier {self.classifier} not implemented"
                )
        if return_preds:
            return X_feats
        else:
            return self

    def _fit(self, X: np.ndarray, y: np.ndarray):
        return self._do_fit(X, y, return_preds=False)

    def _predict(self, X):
        X_feats = self._apply_transform(X)
        return self._clf.predict(X_feats)

    def _predict_proba(self, X):
        X_feats = self._apply_transform(X)
        return self._clf.predict_proba(X_feats)

    def _fit_predict(self, X, y, **kwargs):
        X_feats = self._do_fit(X, y, return_preds=True)
        return self._clf.predict(X_feats)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        if parameter_set == "results_comparison":
            return dict(
                depth=3,
                use_logsig=True,
                sig_mode="brackets",
                do_aug=True,
                do_time_aug=True,
                do_fourier_aug=True,
                return_raw=False,
                add_c22=False,
                add_rocket=True,
                do_rescale=False,
                aug_levels=3,
                dim_limit=8,
                use_kpca=False,
                classifier="ensemble",
                ensemble_params=dict(
                    use_rf2=True,
                    rf=dict(n_estimators=5),
                    logreg=dict(
                        penalty="elasticnet", C=0.1, max_iter=10, l1_ratio=0.15
                    ),
                    logreg2=dict(penalty="l2", C=100, max_iter=10),
                    rf2=dict(
                        n_estimators=5,
                        max_features="log2",
                        criterion="entropy",
                        max_depth=5,
                    ),
                ),
            )
        else:
            return dict(
                depth=2,
                use_logsig=True,
                sig_mode="brackets",
                do_aug=True,
                do_time_aug=True,
                do_fourier_aug=True,
                return_raw=False,
                add_c22=False,
                add_rocket=True,
                do_rescale=False,
                aug_levels=2,
                dim_limit=8,
                use_kpca=False,
                classifier="ensemble",
                ensemble_params=dict(
                    use_rf2=True,
                    rf=dict(n_estimators=2),
                    logreg=dict(
                        penalty="elasticnet", C=0.1, max_iter=10, l1_ratio=0.15
                    ),
                    logreg2=dict(penalty="l2", C=100, max_iter=10),
                    rf2=dict(
                        n_estimators=2,
                        max_features="log2",
                        criterion="entropy",
                        max_depth=5,
                    ),
                ),
            )
