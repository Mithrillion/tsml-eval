__maintainer__ = ["mithrillion"]
__all__ = ["MulSigClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from tsml_eval._wip.mulsigclf.mulsig_features import MulSigTransformer


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
        random_state: int = None,
        wt_levels: int = 3,
        dim_limit: int = 10,
        use_kPCA: bool = False,
        window_alphas: tuple = (1 / 3, 2 / 3, 1),
        do_rescale: bool = False,
        add_c22: bool = False,
        add_rocket: bool = False,
        classifier: str = "logreg",
        ensemble_params: dict = {},
        **clf_kwargs,
    ):
        super(MulSigClassifier, self).__init__()
        self.depth = depth
        self.use_logsig = use_logsig
        self.sig_mode = sig_mode
        self.do_aug = do_aug
        self.do_time_aug = do_time_aug
        self.random_state = random_state
        self.wt_levels = wt_levels
        self.dim_limit = dim_limit
        self.use_kPCA = use_kPCA
        self.window_alphas = window_alphas
        self.do_rescale = do_rescale
        self.add_c22 = add_c22
        self.add_rocket = add_rocket
        self.classifier = classifier
        self.ensemble_params = ensemble_params
        self._transformer = MulSigTransformer(
            depth=depth,
            use_logsig=use_logsig,
            sig_mode=sig_mode,
            do_aug=do_aug,
            do_time_aug=do_time_aug,
            random_state=random_state,
            wt_levels=wt_levels,
            dim_limit=dim_limit,
            use_kPCA=use_kPCA,
            window_alphas=window_alphas,
            do_rescale=do_rescale,
        )
