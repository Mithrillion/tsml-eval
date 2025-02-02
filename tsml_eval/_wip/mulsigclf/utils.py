import math
import numpy as np
import signatory
import torch
from sklearn.decomposition import PCA
import torch.nn.functional as F
from pywt import swt


def endpoint_insensitive_signature(X, depth, basepoint, endpoints=5):
    full_sig = signatory.signature(X, depth=depth, basepoint=False)
    # first, add the full signature itself
    if basepoint == False:
        X_padded = torch.nn.functional.pad(X, (0, 0, 1, 1), mode="replicate")
    else:
        X_padded = torch.nn.functional.pad(X, (0, 0, 0, 1), mode="replicate")
    prefix = signatory.signature(
        X_padded[:, : (endpoints + int(not basepoint)), :],
        depth,
        True,
        basepoint,
        inverse=True,
    )
    suffix = signatory.signature(
        X_padded[:, -(endpoints + 1) :, :],
        depth,
        True,
        False,
        inverse=True,
    )
    sigs = signatory.multi_signature_combine(
        [
            prefix.flatten(0, 1),
            full_sig[:, None, :].expand(-1, endpoints, -1).flatten(0, 1),
            suffix.flatten(0, 1),
        ],
        X.shape[-1],
        depth,
    ).view(X.shape[0], endpoints, -1)

    sig = sigs.mean(1)
    return sig


def to_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, (channel)]
    X_repeat = X.repeat_interleave(2, dim=1)

    # Split out lead and lag
    lead = X_repeat[:, 1:, :]
    lag = X_repeat[:, :-1, :]

    # Combine
    X_leadlag = torch.cat((lead, lag), 2)

    return X_leadlag


def to_nd_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, channel]
    n_chs = X.shape[-1]
    X_repeat = X.repeat_interleave(n_chs * 2, dim=1)
    streams = []
    for i in range(2):
        for j in range(n_chs):
            streams += [
                X_repeat[
                    :, 2 * n_chs - 2 * i - j - 1 : (X_repeat.shape[1] - 2 * i - j), j
                ]
            ]
    X_leadlag = torch.stack(streams, 2)
    return X_leadlag


def time_aug(X, t_range=(0, 1)):
    # assume (batch, time, channel)
    t_len = X.shape[1]
    time_axis = torch.linspace(*t_range, t_len)
    time_component = torch.repeat_interleave(time_axis[None, :, None], len(X), dim=0)
    return torch.cat([time_component, X], dim=-1)


def to_td(X, tau=1, d=2, pad=None, time_last=True):
    # last dimension assumed to be time
    if not time_last:
        # then it must be (batch, time, channel)
        X = X.transpose(-2, -1)
    if d < 2:
        raise ValueError("TD dimension must be at least 2!")
    if pad == "last":
        X_padded = torch.cat(
            [X, torch.repeat_interleave(X[..., [-1]], tau * d, -1)], dim=-1
        )
    elif pad == "first":
        X_padded = torch.cat(
            [torch.repeat_interleave(X[..., [0]], tau * d, -1), X], dim=-1
        )
    else:
        X_padded = X
    return torch.stack(
        [
            X_padded[
                ...,
                i * tau : -(d - i - 1) * tau if d - i - 1 != 0 else X_padded.shape[-1],
            ]
            for i in range(d)
        ],
        dim=-1,
    ).float()


def to_td_pca(
    X,
    dims=2,
    tau=1,
    d=10,
    pretrained_pca=None,
    pad=None,
    random_state=None,
    inc_pca_batch_size=None,
):
    # (batch, time, channel) -> (batch, channel, time, delays)
    # -> (batch, time, channel * delays)
    if d == 1:
        X_td = X
    else:
        X_td = to_td(X, tau, d, pad, time_last=False).transpose(1, 2).flatten(2, 3)
    batch_dims = X_td.shape[:-1]
    if dims is not None and dims != 0:
        if pretrained_pca:
            pca = pretrained_pca
            Z = pca.transform(X_td.reshape(-1, X_td.shape[-1]))
        else:
            pca = PCA(n_components=dims, random_state=random_state)
            Z = pca.fit_transform(X_td.reshape(-1, X_td.shape[-1]))
        return torch.tensor(Z).float().view(*batch_dims, -1), pca
    else:
        # dims is None or 0 -> do not perform pca
        return X_td, None


def rescale_path(path, depth):
    coeff = math.factorial(depth) ** (1 / depth)
    return coeff * path


def znorm(x, dim=-1, eps=1e-8):
    if torch.is_tensor(x):
        return (x - x.mean(dim, keepdim=True)) / (eps + x.std(dim, keepdim=True))
    else:
        return (x - x.mean(dim, keepdims=True)) / (eps + x.std(dim, keepdims=True))


def minmaxnorm(x, dim=-1, eps=1e-8):
    if torch.is_tensor(x):
        return (x - x.amin(dim, keepdim=True)) / (
            eps + x.amax(dim, keepdim=True) - x.amin(dim, keepdim=True)
        )
    else:
        return (x - x.amin(dim, keepdims=True)) / (
            eps + x.amax(dim, keepdims=True) - x.amin(dim, keepdims=True)
        )


def unorm(x, dim=-1, eps=1e-8):
    return x / (eps + torch.norm(x, p=2, dim=dim, keepdim=True))


def swt_map(signal, levels, wavelet="db4"):
    data_len = signal.shape[1]
    required_levels = int(np.ceil(np.log2(data_len)))
    signal = np.pad(
        signal,
        ((0, 0), (0, 2 ** max(levels, required_levels) - data_len), (0, 0)),
        mode="edge",
    )
    coeffs = swt(signal, wavelet, level=levels, trim_approx=True, axis=1)
    return [x[:, :data_len, :] for x in coeffs]


def diff_series(signal):
    padded = F.pad(signal, (0, 0, 1, 1), mode="replicate")
    return ((padded[:, 2:] - padded[:, 1:-1]) + (padded[:, 1:-1] - padded[:, :-2])) / 2


def pad_if_short(x, min_len=9):
    if x.shape[1] < min_len:
        if torch.is_tensor(x):
            return F.pad(x, (0, 0, 0, min_len - x.shape[1]), mode="constant")
        else:
            # Assuming x is a numpy array
            return np.pad(
                x, ((0, 0), (0, min_len - x.shape[1]), (0, 0)), mode="constant"
            )
    else:
        return x
