r"""
.. _friction-derivations:

Friction factor models for pipe flow
------------------------------------

This module provides implementations of the Darcy‑Weisbach friction factor,
including explicit models (Swamee‑Jain, Nikuradse), an iterative model
(Colebrook‑White), and a regime‑aware model that switches between them
based on the Reynolds number. All models conform to the
:class:`FrictionFactorModel` protocol.

Each model also computes the derivative :math:`\mathrm{d}\lambda/\mathrm{d}m`
of the friction factor with respect to mass flow. This derivative is essential
for building the Jacobian matrix in the Newton‑Raphson pipe flow solver. The
derivations of the implemented expressions are given below.

Preliminaries
~~~~~~~~~~~~~

For a circular pipe of diameter :math:`D` and dynamic viscosity :math:`\mu`,
the Reynolds number is

.. math::
   Re = \frac{4|\dot{m}|}{\pi D \mu}.

When the mass flow is positive, :math:`\mathrm{d}Re/\mathrm{d}\dot{m} = Re/\dot{m}`.
For negative flows the absolute value introduces a sign change; in the code we use
:math:`m` directly in the derivative expressions with the understanding that
:math:`\mathrm{d}Re/\mathrm{d}m = Re/m` (valid when :math:`m \neq 0`). The
calling code is expected to guard against zero mass flow.

Swamee‑Jain model
~~~~~~~~~~~~~~~~~

The explicit Swamee‑Jain formula is

.. math::
   \lambda = \frac{0.25}{\left[\log_{10}\!\left(\frac{k/D}{3.7} + 5.74\,Re^{-0.9}\right)\right]^2}
   = \frac{0.25\,(\ln 10)^2}{(\ln X)^2},

where :math:`X = \frac{k/D}{3.7} + 5.74\,Re^{-0.9}`.

Differentiating:

.. math::
   \frac{\mathrm{d}\lambda}{\mathrm{d}Re}
   = 0.25\,(\ln 10)^2 \cdot (-2)\,(\ln X)^{-3} \cdot \frac{1}{X} \cdot \frac{\mathrm{d}X}{\mathrm{d}Re},

with :math:`\frac{\mathrm{d}X}{\mathrm{d}Re} = 5.74 \cdot (-0.9)\,Re^{-1.9}
= -5.166\,Re^{-1.9}`.

Thus

.. math::
   \frac{\mathrm{d}\lambda}{\mathrm{d}Re}
   = 0.25\,(\ln 10)^2 \cdot (-2) \cdot (-5.166)\, \frac{Re^{-1.9}}{(\ln X)^3 X}
   = 0.25\,(\ln 10)^2 \cdot 10.332\, \frac{Re^{-1.9}}{(\ln X)^3 X}.

Multiplying by :math:`\mathrm{d}Re/\mathrm{d}m = Re/m` gives

.. math::
   \frac{\mathrm{d}\lambda}{\mathrm{d}m}
   = \underbrace{0.25\,(\ln 10)^2 \cdot 10.332}_{b}\; \frac{Re^{-0.9}}{(\ln X)^3 X \, m},

where the constant :math:`b \approx 13.6948028193657`. This is the exact
expression implemented in the code.


Nikuradse model
~~~~~~~~~~~~~~~

The Nikuradse friction factor in this module is the sum of a laminar term
and the fully rough turbulent term:

.. math::
   \lambda = \frac{64}{Re} + \frac{1}{\left( -2\log_{10} \left( \frac{k/D}{3.71} \right) \right)^2}.

Only the laminar part depends on :math:`Re` (hence on :math:`m`). Its
derivative is

.. math::
   \frac{\mathrm{d}\lambda}{\mathrm{d}m}
   = \frac{\mathrm{d}}{\mathrm{d}m}\!\left(\frac{64}{Re}\right)
   = -\frac{64}{Re^2}\,\frac{\mathrm{d}Re}{\mathrm{d}m}.

Using :math:`\mathrm{d}Re/\mathrm{d}m = Re/m` yields

.. math::
   \frac{\mathrm{d}\lambda}{\mathrm{d}m} = -\frac{64}{Re\,m}.

This is an **odd** function of :math:`m` (because :math:`Re` is even in
:math:`m`). **The current implementation**, however, computes

.. math::
   \frac{\mathrm{d}\lambda}{\mathrm{d}m} = -\frac{64}{Re\,|m|},

which is even and mathematically incorrect. This is a known limitation
(see the class docstring and the ``FIXME`` comment in the source).

Colebrook model
~~~~~~~~~~~~~~~

The Colebrook‑White equation is solved iteratively for :math:`\lambda`.
After convergence, the derivative is obtained from a closed‑form expression
derived by implicit differentiation. Starting from

.. math::
   \frac{1}{\sqrt{\lambda}} = -2\log_{10}\!\left(
       \frac{k/D}{3.71} + \frac{2.51}{Re\sqrt{\lambda}}
   \right),

define :math:`A = k/D / 3.71` and :math:`B = 2.51 / Re`. The equation can be
written as

.. math::
   F(\lambda, Re) = \frac{1}{\sqrt{\lambda}} + 2\log_{10}\!\left(A + \frac{B}{\sqrt{\lambda}}\right) = 0.

Differentiate with respect to :math:`m`, treating :math:`Re` as a function of
:math:`m`:

.. math::
   -\frac{1}{2}\lambda^{-3/2}\frac{\mathrm{d}\lambda}{\mathrm{d}m}
   + \frac{2}{\ln 10}\,
     \frac{1}{A + B/\sqrt{\lambda}}
     \left( \frac{\mathrm{d}B}{\mathrm{d}m}\frac{1}{\sqrt{\lambda}}
            - \frac{B}{2}\lambda^{-3/2}\frac{\mathrm{d}\lambda}{\mathrm{d}m}
     \right) = 0.

Since :math:`B = 2.51/Re`,
:math:`\frac{\mathrm{d}B}{\mathrm{d}m} = -\frac{2.51}{Re^2}\frac{\mathrm{d}Re}{\mathrm{d}m}
= -\frac{B}{Re}\frac{\mathrm{d}Re}{\mathrm{d}m}`.
With :math:`\frac{\mathrm{d}Re}{\mathrm{d}m} = Re/m` this simplifies to
:math:`\frac{\mathrm{d}B}{\mathrm{d}m} = -B/m`.

Substituting and solving for :math:`\mathrm{d}\lambda/\mathrm{d}m` yields

.. math::
   \frac{\mathrm{d}\lambda}{\mathrm{d}m}
   = -\,\frac{10.04\;\lambda}
           {\bigl(\ln(10)\,(A + B/\sqrt{\lambda})\,Re \;+\; 5.02\bigr)\;m}.

In the code, ``inner_log_term`` corresponds to
:math:`A + B/\sqrt{\lambda}` at the final iteration, and the constants
:math:`10.04` and :math:`5.02` arise from combining the numerical
coefficients in the derivation.

Regime‑aware model
~~~~~~~~~~~~~~~~~~

:class:`RegimeAwareFrictionFactorModel` does not have its own derivative
formula. It delegates the computation to the laminar, transient, or
turbulent sub‑model according to the local Reynolds number.
"""

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
    """Protocol to compute both lambda (friction factor) and
    dlambda / dm (friction factor derivative w.r.t. mass flow)
    """

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        """Computes both lambda and dlambda / dm in one go.

        lambda is a friction factor for Darcy-Weisbach equation. Should be
        and even function: f(-m) = f(m).

        dlambda / dm is a derivative of friction factor with respect to mass flow.
        Should be an odd function: f(-m) = -f(m).

        :param k_over_D: Relative roughness of pipe:
            height_of_pipe_roughness / pipe_inner_diameter
        :param re: Reynolds number. Should be > 0.
        :param m: Mass flow. Used only for dlambda / dm calculation.
        :return: friction factor (lambda) and dlambda / dm.
        """
        ...


@dataclass(slots=True)
class SwameeJain(FrictionFactorModel):
    """Implementation of the Swamee‑Jain explicit friction factor equation.

    This is a direct (non‑iterative) approximation valid for turbulent flow.
    """

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
    r"""Implementation of the Nikuradse friction factor equation.

    The model computes :math:`\lambda` as the sum of a laminar term and the fully
    rough Nikuradse term:

    .. math::
       \lambda = \frac{64}{Re} \;+\; \frac{1}{\bigl(-2\log_{10}(\frac{k/D}{3.71})\bigr)^2}

    The derivative :math:`\mathrm{d}\lambda/\mathrm{d}m` is currently computed as

    .. math::
       \frac{\mathrm{d}\lambda}{\mathrm{d}m} = -\frac{64}{Re\,|m|}\,,

    which makes it an **even** function of :math:`m`.  The mathematically correct
    odd derivative would be :math:`-64/(Re\,m)`.
    """

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
    """Default lambda estimator used for Colebrook first iteration."""
    return 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2


@dataclass(slots=True)
class Colebrook(FrictionFactorModel):
    r"""Implementation of the Colebrook‑White friction factor equation.

    The Colebrook equation is solved iteratively with the Newton‑Raphson method.
    Convergence is controlled by `tolerance` and `max_iter`.  An initial estimator
    (by default `_default_initial_estimator`) provides the starting value.

    .. math::
       \frac{1}{\sqrt{\lambda}} = -2\log_{10}\!\left(
           \frac{k/D}{3.71} + \frac{2.51}{Re\sqrt{\lambda}}
       \right)

    Attributes:
       initial_estimator: Callable returning a starting guess for :math:`\lambda`.
       tolerance: Absolute change in :math:`\lambda` below which iteration stops.
       max_iter: Maximum number of Newton steps before raising
                 `PipeflowNotConverged`.
    """

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
    r"""Friction factor that respects flow regimes.

    Uses appropriate friction factor model for a specified flow regime.
    Laminar, transient and turbulent flow regimes are supported.

    - laminar:   :math:`0 < Re \le \text{re\_laminar}`
    - transient: :math:`\text{re\_laminar} < Re \le \text{re\_turbulent}`
    - turbulent: :math:`\text{re\_turbulent} < Re`
    """

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
