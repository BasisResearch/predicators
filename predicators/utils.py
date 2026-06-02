"""General utility methods (full version, including ML-dependent helpers).

This is the kitchen-sink utils module. It re-exports the env-safe
subset from :mod:`predicators.utils_lite` (so existing callers that
use ``utils.X`` keep working unchanged) and then defines helpers that
depend on heavy libraries (torch, scipy, imageio, the pretrained-model
SDKs).

Browser / Pyodide-targeted code should import
:mod:`predicators.utils_lite` directly instead of this module.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any, List, Sequence, Tuple, Union

import imageio
import numpy as np
import torch
from scipy.stats import beta as BetaRV

# isort: off
# The wildcard re-export of utils_lite has to come before the explicit
# imports of `predicators.structs` and
# `predicators.pretrained_model_interface` below: utils_lite imports
# structs itself, and bringing utils_lite up first avoids a half-loaded
# structs module when utils.py is the first thing imported in a CPython
# process.
# pylint: disable=wildcard-import,unused-wildcard-import
from predicators.utils_lite import *  # noqa: F401, F403
# pylint: enable=wildcard-import,unused-wildcard-import

# Underscore-prefixed names aren't re-exported by `import *` per
# Python's defaults. Surface the private utils symbols that external
# callers (planning.py, planning_with_processes.py, tests/test_utils.py)
# rely on explicitly.
from predicators.utils_lite import (  # noqa: F401
    _abstract_with_derived_predicates,
    _Geom2D,
    _PyperplanHeuristicWrapper,
    _TaskPlanningHeuristic,
)

from predicators.pretrained_model_interface import GoogleGeminiLLM, \
    GoogleGeminiVLM, LargeLanguageModel, OpenAILLM, OpenAIVLM, \
    OpenRouterLLM, OpenRouterVLM, VisionLanguageModel
from predicators.settings import CFG
from predicators.structs import DelayDistribution, Video
# isort: on

def create_llm_by_name(
        model_name: str) -> LargeLanguageModel:  # pragma: no cover
    """Create particular llm using a provided name."""
    if CFG.pretrained_model_service_provider == "openai":
        return OpenAILLM(model_name)
    if CFG.pretrained_model_service_provider == "google":
        return GoogleGeminiLLM(model_name)
    if CFG.pretrained_model_service_provider == "openrouter":
        return OpenRouterLLM(model_name)
    raise ValueError(f"Unknown pretrained model service provider: "
                     f"{CFG.pretrained_model_service_provider}")


def create_vlm_by_name(
        model_name: str) -> VisionLanguageModel:  # pragma: no cover
    """Create particular vlm using a provided name."""
    if CFG.pretrained_model_service_provider == "openai":
        return OpenAIVLM(model_name)
    if CFG.pretrained_model_service_provider == "google":
        return GoogleGeminiVLM(model_name)
    if CFG.pretrained_model_service_provider == "openrouter":
        return OpenRouterVLM(model_name)
    raise ValueError(f"Unknown pretrained model service provider: "
                     f"{CFG.pretrained_model_service_provider}")

def save_video(outfile: str, video: Video) -> None:
    """Save the video to video_dir/outfile."""
    outdir = CFG.video_dir
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile)
    video_uint8 = [np.array(frame).astype(np.uint8) for frame in video]
    imageio.mimwrite(outpath, video_uint8, fps=CFG.video_fps)  # type: ignore
    logging.info(f"Wrote out to {outpath}")


def save_images_parallel(outfile_prefix: str, video: Video) -> None:
    """Save the video as individual images in parallel."""
    outdir = CFG.image_dir
    outdir = os.path.join(outdir, os.path.dirname(outfile_prefix))
    outfile_prefix = os.path.basename(outfile_prefix)

    os.makedirs(outdir, exist_ok=True)
    width = len(str(len(video)))

    def _write_frame(i: int, image: Any) -> None:
        image_number = str(i).zfill(width)
        outfile = outfile_prefix + f"_image_{image_number}.png"
        outpath = os.path.join(outdir, outfile)
        image_array = np.array(image)
        imageio.imwrite(outpath, image_array.astype(np.uint8))
        logging.info(f"Wrote out to {outpath}")

    with ThreadPoolExecutor() as executor:
        for i, frame in enumerate(video):
            executor.submit(_write_frame, i, frame)


def save_images(outfile_prefix: str, video: Video) -> None:
    """Save the video as individual images to image_dir."""
    return save_images_parallel(outfile_prefix, video)

class ConstantDelay(DelayDistribution):
    """ConstantDelay class."""

    def __init__(self, delay: Union[int, float, torch.Tensor]):
        # keep dtype consistent with the rest of the model
        self.delay = torch.as_tensor(delay, dtype=torch.get_default_dtype())
        # reusable – matches self.delay’s dtype/device
        self._neg_inf = torch.tensor(float("-inf"),
                                     dtype=self.delay.dtype,
                                     device=self.delay.device)

    def copy(self) -> ConstantDelay:
        """Return a copy of this distribution."""
        return ConstantDelay(self.delay.clone())

    def sample(self) -> int:
        return int(self.delay.item())

    def set_parameters(self, parameters: Sequence[torch.Tensor],
                       **kwargs: Any) -> None:
        self.delay = parameters[0]
        # Invalidate cached properties
        self.__dict__.pop("_str", None)
        self.__dict__.pop("_hash", None)

    def get_parameters(self) -> Sequence[float]:
        """Return the parameters of the distribution."""
        return [self.delay.item()]

    def probability(self, k: int) -> float:
        return 1.0 if k == int(self.delay.item()) else 0.0

    def log_prob(self, k: Union[int, torch.Tensor]) -> torch.Tensor:
        """Vectorised log-prob; differentiable w.r.t.

        self.delay.
        """
        if not isinstance(k, torch.Tensor):
            k_tensor = torch.tensor(k,
                                    dtype=torch.long,
                                    device=self.delay.device)
        else:
            k_tensor = k.long().to(self.delay.device)

        zeros = torch.zeros_like(k_tensor, dtype=torch.get_default_dtype())
        neg_inf = torch.full_like(k_tensor,
                                  float("-inf"),
                                  dtype=torch.get_default_dtype())
        return torch.where(k_tensor == self.delay.long(), zeros, neg_inf)

    @cached_property
    def _str(self) -> str:
        return f"ConstantDelay({self.delay:.4f})"


class DiscreteGaussianDelay(DelayDistribution):
    r"""Truncated discrete Gaussian distribution  (a.k.a. Discrete Normal).

    Parameters
    ----------
    mu : float or Tensor
        Location parameter (can be any real number).
    sigma : float or Tensor
        Scale (> 0).  Smaller values → tighter mass around ``mu``.
    max_k : int, optional
        Build / cache the PMF on the support  k = 0 … max_k-1  (default 300).
    """

    def __init__(self,
                 mu: torch.Tensor,
                 sigma: torch.Tensor,
                 max_k: int = 300) -> None:
        if not torch.all(sigma > 0):
            raise ValueError("Initial sigma must be positive.")

        self.log_mu = torch.log(mu)
        self.log_sigma = torch.log(sigma)
        self._max_k = max_k
        self._update_cache()

    def copy(self) -> DiscreteGaussianDelay:
        """Return a copy of this distribution."""
        return DiscreteGaussianDelay(self.mu.clone(), self.sigma.clone(),
                                     self._max_k)

    @property
    def sigma(self) -> torch.Tensor:
        """The actual standard deviation, derived from the optimized
        log_sigma."""
        return torch.exp(self.log_sigma)

    @property
    def mu(self) -> torch.Tensor:
        """The mean of the discrete Gaussian."""
        return torch.exp(self.log_mu)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _update_cache(self) -> None:
        """Rebuild cached log-PMF / PMF / CDF using safe numerics."""
        EPS = 1e-8

        mu = self.mu
        sigma_val = self.sigma
        sigma = torch.clamp(sigma_val, min=EPS)  # ensure positivity
        if not torch.all(sigma > 0):
            raise ValueError("Initial sigma must be positive.")

        assert isinstance(self._max_k, int)
        ks = torch.arange(self._max_k, dtype=mu.dtype,
                          device=mu.device)  # k = 0 … max_k-1

        # Unnormalised log-probability of a discrete Gaussian
        #   p̃(k) = exp( −(k−μ)² / (2σ²) )
        # Work in log-space for stability:
        log_p_unnorm = -0.5 * ((ks - mu)**2) / (sigma**2)

        # Remove any accidental NaNs / ±Inf
        log_p_unnorm = torch.nan_to_num(log_p_unnorm,
                                        nan=-torch.inf,
                                        posinf=-torch.inf,
                                        neginf=-torch.inf)

        # Normalise on the bounded support 0 … max_k-1
        log_norm = torch.logsumexp(log_p_unnorm, dim=0)
        self._log_pmf = log_p_unnorm - log_norm

        self._pmf = self._log_pmf.exp()
        self._cdf = torch.cumsum(self._pmf, dim=0)

    # ------------------------------------------------------------------ #
    # Public interface (identical to DoublePoissonDelay)
    # ------------------------------------------------------------------ #
    def set_parameters(self, parameters: Sequence[torch.Tensor],
                       **kwargs: Any) -> None:
        self.log_mu, self.log_sigma = parameters
        if "max_k" in kwargs and kwargs["max_k"] is not None:
            self._max_k = kwargs["max_k"]
        self._update_cache()
        # Invalidate cached repr/hash if present
        self.__dict__.pop('_str', None)
        self.__dict__.pop('_hash', None)

    def get_parameters(self) -> Sequence[float]:
        """Return the parameters of the distribution."""
        return [self.mu.item(), self.sigma.item()]

    def probability(self, k: int) -> float:
        if 0 <= k < self._max_k:
            return float(self._pmf[k])
        return 0.0

    def log_prob(self, k: Union[int, torch.Tensor]) -> torch.Tensor:
        if not isinstance(k, torch.Tensor):
            k_tensor = torch.tensor(k, dtype=torch.long)
        else:
            k_tensor = k.long()

        k_flat = k_tensor.flatten()
        log_probs_flat = torch.full_like(k_flat,
                                         float('-inf'),
                                         dtype=self._log_pmf.dtype)

        mask = (k_flat >= 0) & (k_flat < self._max_k)
        if mask.any():
            log_probs_flat[mask] = self._log_pmf[k_flat[mask]]

        return log_probs_flat.reshape(k_tensor.shape)

    def sample(self, sample_mode: bool = True) -> int:
        if sample_mode:
            return int(self.mu.item())
        u = torch.rand(1).item()
        return int(torch.searchsorted(self._cdf, torch.tensor(u)))

    @cached_property
    def _str(self) -> str:
        return f"DiscreteGaussianDelay({self.mu:.4f}, {self.sigma:.4f})"


def _beta_bernoulli_posterior_alpha_beta(
        success_history: List[bool],
        alpha: float = 1.0,
        beta: float = 1.0) -> Tuple[float, float]:
    """See https://gregorygundersen.com/blog/2020/08/19/bernoulli-beta/"""
    n = len(success_history)
    s = sum(success_history)
    alpha_n = alpha + s
    beta_n = n - s + beta
    return (alpha_n, beta_n)


def beta_bernoulli_posterior(success_history: List[bool],
                             alpha: float = 1.0,
                             beta: float = 1.0) -> BetaRV:
    """Returns the RV."""
    alpha_n, beta_n = _beta_bernoulli_posterior_alpha_beta(
        success_history, alpha, beta)
    return BetaRV(alpha_n, beta_n)


def beta_bernoulli_posterior_mean(success_history: List[bool],
                                  alpha: float = 1.0,
                                  beta: float = 1.0) -> float:
    """Faster computation to avoid instantiating BetaRV when not needed."""
    alpha_n, beta_n = _beta_bernoulli_posterior_alpha_beta(
        success_history, alpha, beta)
    return alpha_n / (alpha_n + beta_n)


def beta_from_mean_and_variance(mean: float,
                                variance: float,
                                variance_lower_pad: float = 1e-6,
                                variance_upper_pad: float = 1e-3) -> BetaRV:
    """Recover a beta distribution given a mean and a variance.

    See https://stats.stackexchange.com/questions/12232/ for derivation.
    """
    # Clip variance.
    variance = max(min(variance,
                       mean * (1 - mean) - variance_upper_pad),
                   variance_lower_pad)
    alpha = ((1 - mean) / variance - 1 / mean) * (mean**2)
    beta = alpha * (1 / mean - 1)
    assert alpha > 0
    assert beta > 0
    rv = BetaRV(alpha, beta)
    assert abs(rv.mean() - mean) < 1e-6
    return rv
