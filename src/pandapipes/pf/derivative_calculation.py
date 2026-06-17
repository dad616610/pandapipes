from typing import Protocol
import numpy as np
from dataclasses import dataclass
import numpy.typing as npt
from pandapipes.constants import NORMAL_TEMPERATURE
from pandapipes.idx_branch import (LENGTH, D, K, RE, LAMBDA, LOAD_VEC_BRANCHES, JAC_DERIV_DM, JAC_DERIV_DP,
                                   JAC_DERIV_DP1, JAC_DERIV_DM_NODE, FROM_NODE, TO_NODE, TOUTINIT, AREA,
                                   LOAD_VEC_BRANCHES_T, JAC_DERIV_DT, LOAD_VEC_NODES_TO_T,
                                   LOAD_VEC_NODES_FROM, LOAD_VEC_NODES_TO, JAC_DERIV_DT_NODE, JAC_DERIV_DTOUT_NODE,
                                   JAC_DERIV_DTOUT, MDOTINIT, DP_FRICT_LOSS)
from pandapipes.idx_node import TINIT as TINIT_NODE, INFEED, LOAD_T, JAC_DERIV_DT_N
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_net_option, get_lookup
from pandapipes.properties.fluids import get_fluid
from pandapipes.properties.properties_toolbox import get_branch_real_density, get_branch_real_eta, get_branch_cp
from scipy.optimize import newton


def calculate_derivatives_hydraulic(net,
                                    branch_pit, node_pit,
                                    branch_pit_old, node_pit_old,
                                    options):
    """
    Function which creates derivatives.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param branch_pit:
    :type branch_pit:
    :param node_pit:
    :type node_pit:
    :param options:
    :type options:
    :return: No Output.
    """
    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import (
            derivatives_hydraulic_incomp_numba as derivatives_hydraulic_incomp,
            derivatives_hydraulic_comp_numba as derivatives_hydraulic_comp,
            calc_medium_pressure_with_derivative_numba as calc_medium_pressure_with_derivative)
    else:
        from pandapipes.pf.derivative_toolbox import (derivatives_hydraulic_incomp_np as derivatives_hydraulic_incomp,
                                                      derivatives_hydraulic_comp_np as derivatives_hydraulic_comp,
                                                      calc_medium_pressure_with_derivative_np as calc_medium_pressure_with_derivative)
    fluid = get_fluid(net)
    gas_mode = fluid.is_gas
    friction_model = options["friction_model"]

    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
    to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
    tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs = get_derived_values(node_pit, from_nodes, to_nodes,
                                                                                      options["use_numba"])

    if gas_mode:
        p_m, der_p_m, der_p_m1 = calc_medium_pressure_with_derivative(p_init_i_abs, p_init_i1_abs)
    else:
        p_m, der_p_m, der_p_m1 = (p_init_i_abs + p_init_i1_abs) / 2, None, None

    rho = get_branch_real_density(fluid, node_pit, branch_pit)
    eta = get_branch_real_eta(fluid, node_pit, branch_pit, p_m)

    # Darcy Friction factor: lambda
    re = np.abs(branch_pit[:,MDOTINIT]) * branch_pit[:,D] / (eta * branch_pit[:, AREA])
    mask = ~np.isclose(re, 0) & ~np.isclose(branch_pit[:, LENGTH], 0, rtol=1e-10, atol=1e-11)
    k_over_D = branch_pit[mask, K] / branch_pit[mask, D]
    lambda_ = np.zeros_like(re)
    der_lambda = np.zeros_like(re)

    if friction_model == "colebrook":
        friction_factor_model = Colebrook(
            tolerance=options.get("tolerance_colebrook", 1e-4),
            max_iter=options.get("max_iter_colebrook", 100),
        )
    elif friction_model == "swamee-jain":
        friction_factor_model = SwameeJain()
    else:
        friction_factor_model = Nikuradse()

    lambda_[mask], der_lambda[mask] = friction_factor_model.compute_lambda_and_dlambda_dm(
        k_over_D,
        re[mask],
        branch_pit[mask, MDOTINIT],
    )

    # lambda_[mask] = calc_lambda(
    #     k_over_D,
    #     re[mask],
    #     friction_model,
    #     options,
    # )
    # der_lambda[mask] = calc_der_lambda(
    #     k_over_D,
    #     re[mask],
    #     branch_pit[mask, MDOTINIT],
    #     lambda_[mask],
    #     friction_model,
    # )

    branch_pit[:, RE] = re
    branch_pit[:, LAMBDA] = lambda_

    if not gas_mode:
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss = (
            derivatives_hydraulic_incomp(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference, rho))
    else:
        rho_n = np.full(len(branch_pit), fluid.get_density(NORMAL_TEMPERATURE))
        comp_fact = fluid.get_compressibility(p_m, tinit_branch)
        dc = fluid.get_der_compressibility()
        # TODO: this might not be required
        der_comp = dc * der_p_m
        der_comp1 = dc * der_p_m1
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss = (
            derivatives_hydraulic_comp(node_pit, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                height_difference, comp_fact, der_comp, der_comp1, rho, rho_n))

    branch_pit[:, LOAD_VEC_BRANCHES] = load_vec
    branch_pit[:, JAC_DERIV_DM] = df_dm
    branch_pit[:, JAC_DERIV_DP] = df_dp
    branch_pit[:, JAC_DERIV_DP1] = df_dp1
    branch_pit[:, LOAD_VEC_NODES_FROM] = load_vec_nodes_from
    branch_pit[:, LOAD_VEC_NODES_TO] = load_vec_nodes_to
    branch_pit[:, JAC_DERIV_DM_NODE] = df_dm_nodes
    branch_pit[:, DP_FRICT_LOSS] = dp_frict_loss


def calculate_derivatives_thermal(net,
                                  branch_pit, node_pit,
                                  branch_pit_old, node_pit_old,
                                  options):
    node_pit_old_lookup = get_lookup(net, "node", "old_pit_cols")
    branch_pit_old_lookup = get_lookup(net, "branch", "old_pit_cols")

    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import derivatives_thermal_numba as derivatives_termal
    else:
        from pandapipes.pf.derivative_toolbox import derivatives_thermal_np as derivatives_termal
    fluid = get_fluid(net)
    cp_b = get_branch_cp(fluid, node_pit, branch_pit)
    # this is not required currently, but useful when implementing leakages
    # m_init_i = np.abs(branch_pit[:, MDOTINIT])
    # m_init_i1 = np.abs(branch_pit[:, MDOTINIT])
    from_nodes = get_from_nodes_corrected(branch_pit)
    to_nodes = get_to_nodes_corrected(branch_pit)
    t_init_i = node_pit[from_nodes, TINIT_NODE]
    t_init_i1 = branch_pit[:, TOUTINIT]
    t_init_nt = node_pit[to_nodes, TINIT_NODE]
    t_init_n = node_pit[:, TINIT_NODE]
    cp_i1 = fluid.get_heat_capacity(t_init_i1)
    cp_nt = fluid.get_heat_capacity(t_init_nt)
    cp_n = fluid.get_heat_capacity((cp_i1 + cp_nt) / 2)
    transient = get_net_option(net, "transient")
    dt = get_net_option(net, "dt")
    rho = get_branch_real_density(fluid, node_pit, branch_pit)
    amb = get_net_option(net, 'ambient_temperature')

    fn, dfn_dt, fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout, infeed = (
        derivatives_termal(node_pit, branch_pit,
                           node_pit_old, node_pit_old_lookup,
                           branch_pit_old, branch_pit_old_lookup,
                           from_nodes, to_nodes,
                           t_init_i, t_init_i1, t_init_nt, t_init_n,
                           cp_n, cp_b,
                           rho, dt, transient, amb))

    node_pit[:, LOAD_T] = fn
    node_pit[:, JAC_DERIV_DT_N] = dfn_dt

    branch_pit[:, LOAD_VEC_BRANCHES_T] = fb
    branch_pit[:, JAC_DERIV_DT] = dfb_dt
    branch_pit[:, JAC_DERIV_DTOUT] = dfb_dtout

    branch_pit[:, LOAD_VEC_NODES_TO_T] = fnt
    branch_pit[:, JAC_DERIV_DT_NODE] = dfnt_dt
    branch_pit[:, JAC_DERIV_DTOUT_NODE] = dfnt_dtout

    node_pit[:, INFEED] = False
    node_pit[infeed, INFEED] = True


def get_derived_values(node_pit, from_nodes, to_nodes, use_numba):
    if use_numba:
        from pandapipes.pf.derivative_toolbox_numba import calc_derived_values_numba
        return calc_derived_values_numba(node_pit, from_nodes, to_nodes)
    from pandapipes.pf.derivative_toolbox import calc_derived_values_np
    return calc_derived_values_np(node_pit, from_nodes, to_nodes)


class FrictionFactorModel(Protocol):

    def compute_lambda_and_dlambda_dm(self, k_over_D, re, m) -> tuple[npt.NDArray]: ...

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
        dlambda_dm = b  * inv_re_09 / (log_cubed * inner_log_term * m)
        return lambda_, dlambda_dm

class Nikuradse(FrictionFactorModel):

    def compute_lambda_and_dlambda_dm(self, k_over_D, re, m):
        laminar = 64 / re
        nikuradse = 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2
        lambda_ = laminar + nikuradse

        # FIXME?: mathematically, der_lambda should be an odd function
        # with m**2 the function is even
        # return -64 / (re * m)
        dlambda_dm = -64 / (re * np.abs(m))
        return lambda_, dlambda_dm

@dataclass
class Colebrook(FrictionFactorModel):
    tolerance: float = 1e-4
    max_iter: int = 100

    def compute_lambda_and_dlambda_dm(self, k_over_D, re, m):
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged

        lambda_ = 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2

        def colebrook_white_implicit(lambda_cb, k_over_D, re):
            inv_lambda_sqrt = 1 / np.sqrt(lambda_cb)
            return inv_lambda_sqrt + 2 * np.log10(2.51 / re * inv_lambda_sqrt + k_over_D / 3.71)

        def cw_derivative(lambda_cb, k_over_D, re):
            inv_lambda_sqrt = 1 / np.sqrt(lambda_cb)
            inv_lambda_sqrt_cubed = inv_lambda_sqrt ** 3
            return -0.5 * inv_lambda_sqrt_cubed - (2.51 / re) * inv_lambda_sqrt_cubed / (
                        np.log(10) * (2.51 / re * inv_lambda_sqrt + k_over_D / 3.71))

        res = newton(colebrook_white_implicit, lambda_, maxiter=self.max_iter, args=(k_over_D, re),
                     tol=self.tolerance, full_output=True, fprime=cw_derivative)

        if lambda_.size == 1:
            lambda_ = res[0]
            converged = res[1].converged
        else:
            lambda_ = res.root
            converged = np.all(res.converged)

        if not converged:
            msg = "The Colebrook-White algorithm did not converge. There might be model inconsistencies. The maximum iterations can be given as 'max_iter_colebrook' argument to the pipeflow."
            raise PipeflowNotConverged(msg)

        ln10 = 2.302585092994045684017991454684364207601
        u = k_over_D / 3.71 + 2.51 / (re * np.sqrt(lambda_))
        dlambda_dm = -10.04 * lambda_ / ((ln10 * u * re + 5.02) * m)
        return lambda_, dlambda_dm


def calc_lambda(k_over_D, re, friction_model, options):
    """
    Function calculates the friction factor of a pipe. Turbulence is calculated based on
    Nikuradse. If v equals 0, a value of 0.001 is used in order to avoid division by zero.
    This should not be a problem as the pressure loss term will equal zero (lambda * u^2).

    :param d:
    :type d:
    :param k:
    :type k:
    :param friction_model:
    :type friction_model:
    :param options:
    :type options:
    :return:
    :rtype:
    """
    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import calc_lambda_nikuradse_numba as calc_lambda_nikuradse
    else:
        from pandapipes.pf.derivative_toolbox import calc_lambda_nikuradse_np as calc_lambda_nikuradse

    lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse(k_over_D, re)

    if friction_model == "colebrook":
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged
        max_iter = options.get("max_iter_colebrook", 100)
        tolerance = options.get("tolerance_colebrook", 1e-4)
        converged, lambda_ = colebrook_white(k_over_D, re, lambda_nikuradse, max_iter, tolerance)
        if not converged:
            raise PipeflowNotConverged("The Colebrook-White algorithm did not converge. There might be model "
                                       "inconsistencies. The maximum iterations can be given as 'max_iter_colebrook' "
                                       "argument to the pipeflow.")
    elif friction_model == "swamee-jain":
        lambda_ = 0.25 / np.log10(k_over_D / 3.7 + 5.74 / (re ** 0.9)) ** 2
    else:
        # lambda_tot = np.where(re > 2300, lambda_laminar + lambda_nikuradse, lambda_laminar)
        lambda_ = lambda_laminar + lambda_nikuradse
    return lambda_


def calc_der_lambda(k_over_D, re, m, lambda_pipe, friction_model):
    """
    Function calculates the derivative of lambda with respect to v. Turbulence is calculated based
    on Nikuradse. This should not be a problem as the pressure loss term will equal zero
    (lambda * u^2).

    :param m:
    :type m:
    :param eta:
    :type eta:
    :param d:
    :type d:
    :param k:
    :type k:
    :param friction_model:
    :type friction_model:
    :param lambda_pipe:
    :type lambda_pipe:
    :param area:
    :type area:
    :return:
    :rtype:
    """
    if friction_model == "colebrook":
        ln10 = 2.302585092994045684017991454684364207601
        u = k_over_D / 3.71 + 2.51 / (re * np.sqrt(lambda_pipe))
        return -10.04 * lambda_pipe / ((ln10 * u * re + 5.02) * m)
    elif friction_model == "swamee-jain":
        inv_re_09 = 1 / re**0.9
        log_term = k_over_D / 3.7 + 5.74 * inv_re_09
        # a = 0.25 * ln(10)**2 * (-2) * 5.74 * (-0.9)
        a = 13.69480281936570206128078428173834769740
        return a * np.log(log_term)**-3 / log_term * inv_re_09 / m
    else:
        # FIXME?: mathematically, der_lambda should be an odd function
        # with m**2 the function is even
        # return -64 / (re * m)
        return -64 / (re * np.abs(m))


def colebrook_white(k_over_D, re, lambda_nikuradse, max_iter, tolerance=1e-4):
    """
    Function calculates the friction factor of a pipe using the Colebrook-White equation. It is an
    implicit equation which is solved using the Newton-Raphson method. For pipes with zero flow or
    zero length, the initial guess is returned. This should be uncritical, as the pressure loss
    term will equal zero (lambda * u^2 * l / d).

    :param re: Reynolds number [dimensionless]
    :type re: np.array
    :param d: Diameter [m]
    :type d: np.array
    :param k: Roughness [m]
    :type k: np.array
    :param lambda_nikuradse: Initial guess for lambda (from Nikuradse)
    :type lambda_nikuradse: np.array
    :param max_iter: Maximum number of iterations for the Colebrook-White calculation
    :type max_iter: int
    :param tolerance: Tolerance for the Colebrook-White calculation
    :type tolerance: float
    :return: lambda_cb, converged
    1. lambda_cb: Friction factor according to Colebrook-White
    2. converged: True, if the Colebrook-White calculation converged for all pipes
    :rtype: (np.array, bool)
    """

    def colebrook_white_implicit(lambda_cb, k_over_D, re):
        inv_lambda_sqrt = 1 / np.sqrt(lambda_cb)
        return inv_lambda_sqrt + 2 * np.log10(2.51 / re * inv_lambda_sqrt + k_over_D / 3.71)

    def cw_derivative(lambda_cb, k_over_D, re):
        inv_lambda_sqrt = 1 / np.sqrt(lambda_cb)
        inv_lambda_sqrt_cubed = inv_lambda_sqrt ** 3
        return -0.5 * inv_lambda_sqrt_cubed - (2.51 / re) * inv_lambda_sqrt_cubed / (
                    np.log(10) * (2.51 / re * inv_lambda_sqrt + k_over_D / 3.71))

    lambda_res = lambda_nikuradse

    res = newton(colebrook_white_implicit, lambda_res, maxiter=max_iter, args=(k_over_D, re),
                 tol=tolerance, full_output=True, fprime=cw_derivative)  # , fprime2=cw_derivative_2)

    if lambda_res.size == 1:
        lambda_res = res[0]
        converged = res[1].converged
    else:
        lambda_res = res.root
        converged = np.all(res.converged)

    return converged, lambda_res
