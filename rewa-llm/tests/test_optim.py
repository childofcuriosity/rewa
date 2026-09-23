from __future__ import annotations

import copy
import math
import unittest

import torch

from rewa_llm.optim import ReWAAdamW, signed_power


class SignedPowerTests(unittest.TestCase):
    def test_fractional_power_preserves_sign_and_zero(self):
        values = torch.tensor([-9.0, -1.0, -0.0, 0.0, 1.0, 9.0])
        actual = signed_power(values, 0.5)
        expected = torch.tensor([-3.0, -1.0, 0.0, 0.0, 1.0, 3.0])

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(torch.isfinite(actual).all().item())

    def test_zero_is_stable_for_large_power(self):
        values = torch.tensor([-2.0, 0.0, 2.0])
        actual = signed_power(values, 9.0)
        expected = torch.tensor([-512.0, 0.0, 512.0])

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertFalse(torch.isnan(actual).any().item())


class ReWAAdamWTests(unittest.TestCase):
    def test_k1_m0_zero_rewa_epsilon_matches_torch_adamw(self):
        initial = torch.tensor([0.0, -1.5, 0.25, 3.0], dtype=torch.float32)
        native_parameter = torch.nn.Parameter(initial.clone())
        rewa_parameter = torch.nn.Parameter(initial.clone())
        settings = dict(
            lr=3e-3,
            betas=(0.8, 0.93),
            eps=1e-6,
            weight_decay=0.07,
        )
        native = torch.optim.AdamW(
            [native_parameter], **settings, foreach=False
        )
        rewa = ReWAAdamW(
            [
                {
                    "params": [rewa_parameter],
                    "rewa": True,
                    "rewa_k": 1.0,
                    "rewa_m": 0.0,
                    "rewa_eps": 0.0,
                }
            ],
            **settings,
        )

        gradients = (
            torch.tensor([0.2, -0.4, 0.0, 1.2]),
            torch.tensor([-0.1, 0.3, -0.7, 0.0]),
            torch.tensor([0.0, -0.2, 0.5, -0.9]),
            torch.tensor([0.8, 0.0, -0.4, 0.1]),
        )
        for gradient in gradients:
            native_parameter.grad = gradient.clone()
            rewa_parameter.grad = gradient.clone()
            native.step()
            rewa.step()

            torch.testing.assert_close(
                rewa_parameter, native_parameter, rtol=1e-6, atol=1e-7
            )

        native_state = native.state[native_parameter]
        rewa_state = rewa.state[rewa_parameter]
        torch.testing.assert_close(
            rewa_state["exp_avg"], native_state["exp_avg"], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            rewa_state["exp_avg_sq"],
            native_state["exp_avg_sq"],
            rtol=0.0,
            atol=0.0,
        )

    def test_one_rewa_step_matches_manual_calculation(self):
        initial = torch.tensor([-8.0, 1.0, 0.0], dtype=torch.float32)
        gradient = torch.tensor([0.4, -0.6, 2.0], dtype=torch.float32)
        parameter = torch.nn.Parameter(initial.clone())
        lr = 0.05
        beta1, beta2 = 0.2, 0.3
        adam_eps = 1e-4
        weight_decay = 0.1
        k = 3.0
        m = 2.0
        rewa_eps = 0.25
        optimizer = ReWAAdamW(
            [
                {
                    "params": [parameter],
                    "rewa": True,
                    "rewa_k": k,
                    "rewa_m": m,
                    "rewa_eps": rewa_eps,
                }
            ],
            lr=lr,
            betas=(beta1, beta2),
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        y = initial.sign() * initial.abs().pow(1.0 / k)
        jacobian = k * y.abs().pow(k - 1.0)
        scale = jacobian / (jacobian + rewa_eps) * y.abs().pow(m)
        replaced_gradient = gradient * scale
        exp_avg = (1.0 - beta1) * replaced_gradient
        exp_avg_sq = (1.0 - beta2) * replaced_gradient.square()
        decayed_y = y * (1.0 - lr * weight_decay)
        denominator = (
            exp_avg_sq.sqrt() / math.sqrt(1.0 - beta2) + adam_eps
        )
        updated_y = decayed_y - (lr / (1.0 - beta1)) * exp_avg / denominator
        expected = updated_y.sign() * updated_y.abs().pow(k)

        parameter.grad = gradient.clone()
        optimizer.step()

        torch.testing.assert_close(parameter, expected, rtol=2e-6, atol=2e-6)
        state = optimizer.state[parameter]
        torch.testing.assert_close(
            state["exp_avg"], exp_avg, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            state["exp_avg_sq"], exp_avg_sq, rtol=1e-6, atol=1e-7
        )

    def test_state_dict_roundtrip_continues_identically(self):
        rewa_a = torch.nn.Parameter(torch.tensor([-0.8, 0.2, 1.4]))
        dense_a = torch.nn.Parameter(torch.tensor([0.5, -0.3]))
        optimizer_a = ReWAAdamW(
            [
                {
                    "params": [rewa_a],
                    "rewa": True,
                    "rewa_k": 3.0,
                    "rewa_m": 2.0,
                    "rewa_eps": 1e-6,
                    "weight_decay": 1e-3,
                },
                {"params": [dense_a], "rewa": False, "weight_decay": 0.1},
            ],
            lr=2e-3,
            betas=(0.7, 0.91),
            eps=1e-7,
        )

        warmup_gradients = (
            (torch.tensor([0.3, -0.2, 0.1]), torch.tensor([-0.4, 0.6])),
            (torch.tensor([-0.1, 0.5, -0.7]), torch.tensor([0.2, -0.3])),
        )
        for rewa_gradient, dense_gradient in warmup_gradients:
            rewa_a.grad = rewa_gradient.clone()
            dense_a.grad = dense_gradient.clone()
            optimizer_a.step()

        rewa_b = torch.nn.Parameter(rewa_a.detach().clone())
        dense_b = torch.nn.Parameter(dense_a.detach().clone())
        optimizer_b = ReWAAdamW(
            [
                {"params": [rewa_b], "rewa": True},
                {"params": [dense_b], "rewa": False},
            ],
            lr=1.0,
        )
        optimizer_b.load_state_dict(copy.deepcopy(optimizer_a.state_dict()))

        continuation = (
            torch.tensor([0.9, -0.4, 0.25]),
            torch.tensor([-0.15, 0.35]),
        )
        rewa_a.grad = continuation[0].clone()
        dense_a.grad = continuation[1].clone()
        rewa_b.grad = continuation[0].clone()
        dense_b.grad = continuation[1].clone()
        optimizer_a.step()
        optimizer_b.step()

        torch.testing.assert_close(rewa_b, rewa_a, rtol=0.0, atol=0.0)
        torch.testing.assert_close(dense_b, dense_a, rtol=0.0, atol=0.0)
        for parameter_a, parameter_b in ((rewa_a, rewa_b), (dense_a, dense_b)):
            state_a = optimizer_a.state[parameter_a]
            state_b = optimizer_b.state[parameter_b]
            self.assertEqual(state_a["step"], state_b["step"])
            torch.testing.assert_close(
                state_b["exp_avg"], state_a["exp_avg"], rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                state_b["exp_avg_sq"],
                state_a["exp_avg_sq"],
                rtol=0.0,
                atol=0.0,
            )

    def test_half_parameters_keep_fp32_state_before_and_after_load(self):
        parameter = torch.nn.Parameter(
            torch.tensor([-0.5, 0.0, 0.75], dtype=torch.float16)
        )
        optimizer = ReWAAdamW(
            [
                {
                    "params": [parameter],
                    "rewa": True,
                    "rewa_k": 3.0,
                    "rewa_m": 2.0,
                }
            ],
            lr=1e-3,
        )
        parameter.grad = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float16)
        optimizer.step()

        state = optimizer.state[parameter]
        self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(parameter).all().item())

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored_optimizer = ReWAAdamW(
            [{"params": [restored_parameter], "rewa": True}], lr=1e-3
        )
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored_state = restored_optimizer.state[restored_parameter]
        self.assertEqual(restored_state["exp_avg"].dtype, torch.float32)
        self.assertEqual(restored_state["exp_avg_sq"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
