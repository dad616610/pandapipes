import math
from dataclasses import dataclass
from optparse import make_option

import numpy as np
import numpy.typing as npt
import pytest

from pandapipes.pf import friction_factor_model as fm


@pytest.fixture
def model_payload():
    return {
        "k_over_D": 0.05,
        "re": 2000,
        "m": 1,
    }


@pytest.mark.parametrize(
    "model_class, expected_lambda, expected_dlambda_dm",
    (
        (fm.Nikuradse, 0.1035, -0.032),
        (fm.SwameeJain, 0.0373, -0.0123),
        (fm.Colebrook, 0.0818, -0.0094),
    ),
)
def test_compute_lambda_and_dlambda_dm(
    model_payload,
    model_class,
    expected_lambda,
    expected_dlambda_dm,
):
    model = model_class()
    lambda_, dlambda_dm = model.compute_lambda_and_dlambda_dm(**model_payload)
    assert math.isclose(lambda_, expected_lambda, abs_tol=1e-3)
    assert math.isclose(dlambda_dm, expected_dlambda_dm, abs_tol=1e-3)


@pytest.mark.parametrize(
    "model_class",
    (
        # fm.Nikuradse,
        fm.SwameeJain,
        fm.Colebrook,
    ),
)
def test_dlambda_dm_oddity(model_payload, model_class):
    model = model_class()
    _, dlambda_dm = model.compute_lambda_and_dlambda_dm(**model_payload)
    model_payload["m"] *= -1
    _, dlambda_dm2 = model.compute_lambda_and_dlambda_dm(**model_payload)
    assert math.isclose(dlambda_dm + dlambda_dm2, 0)


@dataclass(slots=True)
class MockFrictionModel(fm.FrictionFactorModel):
    res_value: float = 1

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D,
        re,
        m,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        res = np.full_like(re, self.res_value)
        return res, res


class TestOrchestrator:
    def test_default_uses_nikuradse(self):
        orch = fm.Orchestrator()
        nikuradse = fm.Nikuradse()

        re = np.array([1, 1e3, 1e6, 1e9])
        ones = np.ones_like(re)
        expected_lambda, expected_dlambda_dm = nikuradse.compute_lambda_and_dlambda_dm(
            k_over_D=ones,
            re=re,
            m=ones,
        )
        actual_lambda, actual_dlambda_dm = orch.do(k_over_D=ones, re=re, m=ones)
        assert np.allclose(expected_lambda, actual_lambda)
        assert np.allclose(expected_dlambda_dm, actual_dlambda_dm)

    def test_single_model_applies_everywhere(self):
        mock_value = 42
        mock = MockFrictionModel(mock_value)
        orch = fm.Orchestrator(mock)

        re = np.array([1, 1e3, 1e6, 1e9])
        ones = np.full_like(re, mock_value)
        expected_lambda, expected_dlambda_dm = mock.compute_lambda_and_dlambda_dm(
            k_over_D=ones,
            re=re,
            m=ones,
        )
        actual_lambda, actual_dlambda_dm = orch.do(k_over_D=ones, re=re, m=ones)
        assert np.allclose(expected_lambda, actual_lambda)
        assert np.allclose(expected_dlambda_dm, actual_dlambda_dm)

    def test_multi_model_assignment_is_correct(self, model_payload):
        lam_value = 1
        trans_value = 2
        turb_value = 3
        re_lam = 2300
        re_turb = 4000
        orch = fm.Orchestrator(
            re_laminar=re_lam,
            re_turbulent=re_turb,
            laminar=MockFrictionModel(lam_value),
            transient=MockFrictionModel(trans_value),
            turbulent=MockFrictionModel(turb_value),
        )
        model_payload.pop("re")

        def _assert_lambda_and_dlambda_dm(re, expected_val):
            lambda_, dlambda_dm = orch.do(**model_payload, re=re)
            assert np.allclose(lambda_, expected_val)
            assert np.allclose(dlambda_dm, expected_val)

        # test laminar re range: 0 < re <= re_lam
        _assert_lambda_and_dlambda_dm(re=re_lam * 0.8, expected_val=lam_value)
        _assert_lambda_and_dlambda_dm(re=re_lam, expected_val=lam_value)

        # test transient re range: re_lam < re <= re_turb
        _assert_lambda_and_dlambda_dm(re=re_lam + 1, expected_val=trans_value)
        _assert_lambda_and_dlambda_dm(re=re_turb, expected_val=trans_value)

        # test turbulent re range: re_turb < re
        _assert_lambda_and_dlambda_dm(re=re_turb + 1, expected_val=turb_value)

        re = np.array([2000, 3000, 5000])
        model_payload = {k: np.ones_like(re) for k in model_payload}
        expected_val = np.array([lam_value, trans_value, turb_value])
        _assert_lambda_and_dlambda_dm(re=re, expected_val=expected_val)

    def test_multi_model_incorrect_re_ranges(self):
        mock = MockFrictionModel(42)
        payload = {
            "laminar": mock,
            "transient": mock,
            "turbulent": mock,
        }

        with pytest.raises(ValueError, match="Must have 0 < re_laminar < re_turbulent"):
            fm.Orchestrator(re_laminar=-1, re_turbulent=4000, **payload)

        with pytest.raises(ValueError, match="Must have 0 < re_laminar < re_turbulent"):
            fm.Orchestrator(re_laminar=4000, re_turbulent=2000, **payload)

        fm.Orchestrator(model=mock, re_laminar=-1, re_turbulent=4000)

    def test_models_return_two_values(self, model_payload):
        mock = MockFrictionModel(42)
        single_model = fm.Orchestrator(mock)
        assert len(single_model.do(**model_payload)) == 2

        multi_model = fm.Orchestrator(laminar=mock, transient=mock, turbulent=mock)
        assert len(multi_model.do(**model_payload)) == 2

    def test_cannot_combine_model_with_laminar(self):
        mock = MockFrictionModel(42)
        with pytest.raises(TypeError, match="Cannot combine 'model' with"):
            fm.Orchestrator(model=mock, laminar=mock)

    def test_multi_model_requires_all_three(self):
        mock = MockFrictionModel(42)
        with pytest.raises(
            TypeError,
            match="All of 'laminar', 'transient', 'turbulent' are required",
        ):
            fm.Orchestrator(laminar=mock, transient=mock)

    @pytest.mark.parametrize(
        "model_class",
        (
            fm.Nikuradse,
            fm.SwameeJain,
            fm.Colebrook,
        ),
    )
    def test_do_returns_floats(
        self,
        model_payload,
        model_class,
    ):
        model = model_class()

        orch = fm.Orchestrator(model)
        lambda_, dlambda_dm = orch.do(**model_payload)
        assert np.issubdtype(lambda_.dtype, np.floating)
        assert np.issubdtype(dlambda_dm.dtype, np.floating)

        orch = fm.Orchestrator(laminar=model, transient=model, turbulent=model)
        lambda_, dlambda_dm = orch.do(**model_payload)
        assert np.issubdtype(lambda_.dtype, np.floating)
        assert np.issubdtype(dlambda_dm.dtype, np.floating)

    def test_do_accepts_single_values(self, model_payload):
        model = fm.SwameeJain()
        expected_lambda = 0.0373
        orch = fm.Orchestrator(model)
        lambda_, _ = orch.do(**model_payload)
        assert np.allclose(lambda_, expected_lambda, atol=1e-3)

        orch = fm.Orchestrator(laminar=model, transient=model, turbulent=model)
        lambda_, _ = orch.do(**model_payload)
        assert np.allclose(lambda_, expected_lambda, atol=1e-3)
