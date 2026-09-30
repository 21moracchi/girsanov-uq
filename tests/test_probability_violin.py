import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from girsanov_uq.utils.utils import compute_probability_violin


class ProbabilityViolinTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_loglinear_probability_weighted_score_average(self):
        scores = np.array([[[1.0], [1.0]], [[3.0], [3.0]]])
        model_probs = np.array([0.2, 0.8])
        thetas_sample = np.array([[0.1]])

        _, _, p_theta, _ = compute_probability_violin(
            scores,
            model_probs,
            thetas_sample,
            method="loglinear",
            variance="estimator",
            n_samples=20,
            n_theta_plot=1,
            rng=0,
            score_average="probability_weighted",
        )

        projected = np.array([0.1, 0.3])
        expected = np.mean(model_probs) * np.exp(
            np.sum(model_probs * projected) / np.sum(model_probs)
        )
        np.testing.assert_allclose(p_theta, [expected])

    def test_loglinear_unweighted_score_average(self):
        scores = np.array([[[1.0], [1.0]], [[3.0], [3.0]]])
        model_probs = np.array([0.2, 0.8])
        thetas_sample = np.array([[0.1]])

        _, _, p_theta, _ = compute_probability_violin(
            scores,
            model_probs,
            thetas_sample,
            method="loglinear",
            variance="estimator",
            n_samples=20,
            n_theta_plot=1,
            rng=0,
            score_average="unweighted",
        )

        projected = np.array([0.1, 0.3])
        expected = np.mean(model_probs) * np.exp(np.mean(projected))
        np.testing.assert_allclose(p_theta, [expected])


if __name__ == "__main__":
    unittest.main()
