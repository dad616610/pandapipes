from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


def get_friction_model(opts):
    model_key = opts["friction_model"]
    if isinstance(model_key, FrictionFactorModel):
        return model_key

    if model_key == "colebrook":
        friction_factor_model = Colebrook(
            tolerance=opts.get("tolerance_colebrook", 1e-4),
            max_iter=opts.get("max_iter_colebrook", 100),
        )
    elif model_key == "swamee-jain":
        friction_factor_model = SwameeJain()
    else:
        friction_factor_model = Nikuradse()
    return friction_factor_model


@runtime_checkable
class FrictionFactorModel(Protocol):
    def compute_lambda_and_dlambda_dm(self, k_over_D, re, m) -> tuple[npt.NDArray]: ...


@dataclass(slots=True)
class SwameeJain(FrictionFactorModel):
    def compute_lambda_and_dlambda_dm(self, k_over_D, re, m):
        inv_re_09 = 1 / re**0.9
        inner_log_term = k_over_D / 3.7 + 5.74 * inv_re_09
        log_term = np.log(inner_log_term)
        log_squared = log_term * log_term
        log_cubed = log_squared * log_term

        # a = 0.25 * ln(10)
        a = 0.5756462732485114210044978636710910519003
        lambda_ = a / log_squared

        # a = 0.25 * ln(10)**2 * (-2) * 5.74 * (-0.9)
        b = 13.69480281936570206128078428173834769740
        dlambda_dm = b * inv_re_09 / (log_cubed * inner_log_term * m)
        return lambda_, dlambda_dm


@dataclass(slots=True)
class Nikuradse(FrictionFactorModel):
    def compute_lambda_and_dlambda_dm(self, k_over_D, re, m):
        laminar = 64 / re
        nikuradse = 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2
        lambda_ = laminar + nikuradse

        # FIXME?: mathematically, dlambda / dm should be an odd function,
        # but with m**2 the function is even
        # return -64 / (re * m)
        dlambda_dm = -64 / (re * np.abs(m))
        return lambda_, dlambda_dm


@dataclass(slots=True)
class Colebrook(FrictionFactorModel):
    tolerance: float = 1e-4
    max_iter: int = 100

    def __post_init__(self):
        if not self.max_iter > 0:
            msg = "'max_iter' should be > 0"
            raise ValueError(msg)
        if not self.tolerance > 0:
            msg = "'tolerance' should be > 0"
            raise ValueError(msg)

    def compute_lambda_and_dlambda_dm(self, k_over_D, re, m):
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged

        lambda_prev = 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2
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
