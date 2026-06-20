from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias, runtime_checkable

import numpy as np

Float64_1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
FrictionFactorResult: TypeAlias = tuple[
    Float64_1D,
    Float64_1D,
]


@runtime_checkable
class FrictionFactorModel(Protocol):
    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult: ...


@dataclass(slots=True)
class SwameeJain(FrictionFactorModel):
    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        inv_re_09 = 1 / re**0.9
        inner_log_term = k_over_D / 3.7 + 5.74 * inv_re_09
        log_term = np.log(inner_log_term)
        log_squared = log_term * log_term
        log_cubed = log_squared * log_term

        # a = 0.25 * ln(10)**2
        a = 1.325474527619599502640416597148504422899
        lambda_ = a / log_squared

        # a = 0.25 * ln(10)**2 * (-2) * 5.74 * (-0.9)
        b = 13.69480281936570206128078428173834769740
        dlambda_dm = b * inv_re_09 / (log_cubed * inner_log_term * m)
        return lambda_, dlambda_dm


@dataclass(slots=True)
class Nikuradse(FrictionFactorModel):
    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        laminar = 64 / re
        nikuradse = 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2
        lambda_ = laminar + nikuradse

        # FIXME?: mathematically, dlambda / dm should be an odd function,
        # but with m**2 the function is even
        # return -64 / (re * m)
        dlambda_dm = -64 / (re * np.abs(m))
        return lambda_, dlambda_dm


LambdaEstimator: TypeAlias = Callable[[Float64_1D, Float64_1D], Float64_1D]


def _default_initial_estimator(k_over_D: Float64_1D, re: Float64_1D) -> Float64_1D:
    return 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2


@dataclass(slots=True)
class Colebrook(FrictionFactorModel):
    initial_estimator: LambdaEstimator | None = None
    tolerance: float = 1e-4
    max_iter: int = 100

    def __post_init__(self):
        if self.initial_estimator is None:
            self.initial_estimator = _default_initial_estimator
        if not self.max_iter > 0:
            msg = "'max_iter' should be > 0"
            raise ValueError(msg)
        if not self.tolerance > 0:
            msg = "'tolerance' should be > 0"
            raise ValueError(msg)

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged

        lambda_prev = self.initial_estimator(k_over_D, re)
        lambda_curr = lambda_prev

        a = k_over_D / 3.71
        b = 2.51 / re
        # 1 / ln(10)
        inv_ln10 = 0.4342944819032518276511289189166050822944
        for _ in range(self.max_iter):
            inv_lambda_sqrt = 1 / np.sqrt(lambda_curr)
            inner_log_term = a + b * inv_lambda_sqrt
            cubed_inv_lambda_sqrt = inv_lambda_sqrt**3

            f = inv_lambda_sqrt + 2 * np.log10(inner_log_term)
            df = (
                -0.5 * cubed_inv_lambda_sqrt
                - b * cubed_inv_lambda_sqrt * inv_ln10 / inner_log_term
            )

            lambda_curr = lambda_prev - f / df
            if np.all(np.abs(lambda_curr - lambda_prev) < self.tolerance):
                break
            lambda_prev = lambda_curr
        else:
            msg = (
                "The Colebrook-White algorithm did not converge. "
                "There might be model inconsistencies. The maximum iterations "
                "can be given as 'max_iter_colebrook' argument to the pipeflow."
            )
            raise PipeflowNotConverged(msg)

        ln10 = 2.302585092994045684017991454684364207601
        dlambda_dm = -10.04 * lambda_curr / ((ln10 * inner_log_term * re + 5.02) * m)
        return lambda_curr, dlambda_dm


@dataclass(slots=True)
class RegimeAwareFrictionFactorModel(FrictionFactorModel):
    laminar: FrictionFactorModel
    transient: FrictionFactorModel
    turbulent: FrictionFactorModel
    re_laminar: float = 2300
    re_turbulent: float = 4000

    def __post_init__(self):
        if not (0 < self.re_laminar < self.re_turbulent):
            msg = "Must have 0 < re_laminar < re_turbulent"
            raise ValueError(msg)

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        lam = re <= self.re_laminar
        turb = re > self.re_turbulent
        trans = ~lam & ~turb
        ranges = [
            (lam, self.laminar),
            (trans, self.transient),
            (turb, self.turbulent),
        ]

        lambda_ = np.empty_like(re, dtype=np.float64)
        dlambda_dm = np.empty_like(lambda_)
        for mask, model in ranges:
            if mask.any():
                lambda_[mask], dlambda_dm[mask] = model.compute_lambda_and_dlambda_dm(
                    k_over_D[mask],
                    re[mask],
                    m[mask],
                )

        return lambda_, dlambda_dm
