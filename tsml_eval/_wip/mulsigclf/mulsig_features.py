"""Symbolic Fourier Approximation (SFA) Transformer.
Configurable SFA transform for discretising time series into words.
"""

__maintainer__ = ["mithrillion"]
__all__ = ["MulSigTransformer"]

from sklearn.decomposition import KernelPCA
from tsml_eval._wip.mulsigclf.utils import *
from functools import partial
import numpy as np
from scipy.signal import windows
from aeon.transformations.collection import BaseCollectionTransformer
from einops import rearrange


class MulSigTransformer(BaseCollectionTransformer):

    _tags = {
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    @staticmethod
    def get_window(length, alpha):
        return torch.tensor(windows.tukey(length, alpha=alpha))

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
    ):
        super(MulSigTransformer, self).__init__()
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
        self._pca = None

    def _fit(self, X, y=None):
        # Fit method implementation
        random_state = (
            np.int32(self.random_state) if isinstance(self.random_state, int) else None
        )

        td_X_og = self._preprocess_data(X)

        if td_X_og.shape[-1] > self.dim_limit:
            if not self.use_kPCA:
                td_X_og, pca = to_td_pca(
                    td_X_og, self.dim_limit, 1, 1, random_state=random_state
                )
            else:
                pca = KernelPCA(
                    self.dim_limit,
                    kernel="rbf",
                    random_state=random_state,
                    fit_inverse_transform=False,
                    n_jobs=-1,
                )
                td_X_og = torch.tensor(pca.fit_transform(td_X_og.flatten(0, 1))).view(
                    *td_X_og.shape[:2], -1
                )
        self._pca = pca
        self._pca_rs = random_state

        return self

    def _preprocess_data(self, X):
        X_numpy = rearrange(X, "b c t -> b t c").astype(np.float32)
        X_tensor = torch.tensor(X_numpy)
        td_X_og = swt_map(X_numpy, self.wt_levels, "haar")
        td_X_og = torch.cat(
            [X_tensor, diff_series(X_tensor)]
            + [torch.tensor(x).float() for x in td_X_og],
            -1,
        )
        return td_X_og

    def _compute_sig_feats(self, td_X_og):
        multi_series = [td_X_og] + [
            self.get_window(td_X_og.shape[1], alpha)[None, :, None].float() * td_X_og
            for alpha in self.window_alphas
        ]

        if self.use_logsig:
            sig_map = partial(
                signatory.logsignature, mode=self.sig_mode, depth=self.depth
            )
            base_sig_map = partial(
                signatory.logsignature,
                mode=self.sig_mode,
                basepoint=True,
                depth=self.depth,
            )
        else:
            sig_map = partial(signatory.signature, depth=self.depth)
            base_sig_map = partial(
                signatory.signature, basepoint=True, depth=self.depth
            )

        td_X_list = []
        og_iter = True
        # print("computing sigs...")
        for td_X_part in multi_series:
            if self.do_time_aug:
                td_X_part = time_aug(td_X_part, (0, 1))

            if og_iter:
                sigs_base = base_sig_map(
                    rescale_path(td_X_part, self.depth if self.do_rescale else 1)
                )
                td_X_list.append(sigs_base)
                og_iter = False

            sigs = sig_map(
                rescale_path(td_X_part, self.depth if self.do_rescale else 1)
            )

            sigs_normed = unorm(sigs)
            td_X_list.append(sigs_normed)

        feats = torch.cat(td_X_list, -1)

        return feats

    def _transform(self, X, y=None):
        td_X_og = self._preprocess_data(X)
        if td_X_og.shape[-1] > self.dim_limit:
            if not self.use_kPCA:
                td_X_og, _ = to_td_pca(
                    td_X_og,
                    self.dim_limit,
                    1,
                    1,
                    pretrained_pca=self._pca,
                    random_state=self._pca_rs,
                )
            else:
                td_X_og = torch.tensor(
                    self._pca.fit_transform(td_X_og.flatten(0, 1))
                ).view(*td_X_og.shape[:2], -1)

        return self._compute_sig_feats(td_X_og)

    def _fit_transform(self, X, y=None):
        random_state = (
            np.int32(self.random_state) if isinstance(self.random_state, int) else None
        )

        td_X_og = self._preprocess_data(X)

        if td_X_og.shape[-1] > self.dim_limit:
            if not self.use_kPCA:
                td_X_og, pca = to_td_pca(
                    td_X_og, self.dim_limit, 1, 1, random_state=random_state
                )
            else:
                pca = KernelPCA(
                    self.dim_limit,
                    kernel="rbf",
                    random_state=random_state,
                    fit_inverse_transform=False,
                    n_jobs=-1,
                )
                td_X_og = torch.tensor(pca.fit_transform(td_X_og.flatten(0, 1))).view(
                    *td_X_og.shape[:2], -1
                )
        self._pca = pca
        self._pca_rs = random_state
        return self._compute_sig_feats(td_X_og)
