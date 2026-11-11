import unittest

import numpy as np

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import compare_classical_models, train_linear_regression, train_logistic_classifier


class LinearModelsTests(unittest.TestCase):
    def test_linear_single_batch_update(self):
        result = train_linear_regression(RegressionInput(
            [[1], [2]], [2, 4], [[1], [2]], [0, 0], learning_rate=0.1, epochs=1,
        ))
        np.testing.assert_allclose(result.weights, [1])
        self.assertAlmostEqual(result.bias, 0.6)
        np.testing.assert_allclose(result.predictions, [1.6, 2.6])
        np.testing.assert_allclose(result.loss_history, [1.06])

    def test_linear_converges_to_least_squares(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(200, 3))
        target = features @ np.array([2.5, -1, 0.75]) + 4
        target += rng.normal(scale=0.1, size=200)
        result = train_linear_regression(RegressionInput(features, target, features[:10], target[:10]))
        reference = np.linalg.lstsq(np.column_stack([features, np.ones(200)]), target, rcond=None)[0]
        np.testing.assert_allclose(result.weights, reference[:-1], atol=1e-8)
        self.assertAlmostEqual(result.bias, reference[-1], places=8)
        np.testing.assert_allclose(result.predictions, features[:10] @ result.weights + result.bias)
        self.assertEqual(len(result.loss_history), 800)
        self.assertTrue(np.all(np.diff(result.loss_history) <= 1e-10))

    def test_logistic_single_batch_update(self):
        result = train_logistic_classifier(ClassificationInput(
            [[1], [3]], [0, 1], [[1], [3]], [1, 0], learning_rate=0.2, epochs=1,
        ))
        np.testing.assert_allclose(result.weights, [0.1])
        self.assertAlmostEqual(result.bias, 0)
        expected = 1 / (1 + np.exp(-np.array([0.1, 0.3])))
        np.testing.assert_allclose(result.probabilities, expected)
        np.testing.assert_array_equal(result.predictions, [1, 1])
        self.assertAlmostEqual(result.loss_history[0], (-np.log(1 - expected[0]) - np.log(expected[1])) / 2)

    def test_logistic_stays_finite_at_extreme_scores(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = train_logistic_classifier(ClassificationInput(
                [[1000], [-1000]], [1, 0], [[1000], [-1000]], [1, 0],
                learning_rate=100, epochs=2,
            ))
        self.assertTrue(np.isfinite(result.loss_history).all())
        np.testing.assert_array_equal(result.predictions, [1, 0])
        np.testing.assert_array_equal(result.probabilities, [1, 0])

    def test_test_labels_do_not_affect_training(self):
        features = [[-2], [-1], [1], [2]]
        target = [0, 0, 1, 1]
        for trainer, data_type in [
            (train_linear_regression, RegressionInput),
            (train_logistic_classifier, ClassificationInput),
        ]:
            first = trainer(data_type(features, target, features, target))
            second = trainer(data_type(features, target, features, [1, 1, 0, 0]))
            np.testing.assert_array_equal(first.weights, second.weights)
            np.testing.assert_array_equal(first.predictions, second.predictions)
            self.assertEqual(first.bias, second.bias)
            self.assertEqual(first.loss_history, second.loss_history)

    def test_comparison_evaluates_test_labels_on_the_same_split(self):
        train_x = np.array([[-2], [-1]] * 10 + [[1], [2]] * 10)
        train_y = np.array([0] * 20 + [1] * 20)
        test_x = [[-1.5], [1.5]]
        correct = compare_classical_models(ClassificationInput(train_x, train_y, test_x, [0, 1]))
        reversed_labels = compare_classical_models(ClassificationInput(train_x, train_y, test_x, [1, 0]))
        for name in ["logistic_regression", "decision_tree", "knn"]:
            for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
                self.assertEqual(getattr(getattr(correct, name), metric), 1)
                self.assertEqual(getattr(getattr(reversed_labels, name), metric), 0)

    def test_logistic_converges_and_returns_consistent_probabilities(self):
        features = np.array([[-2, 1], [-1, -1], [1, 1], [2, -1]])
        result = train_logistic_classifier(ClassificationInput(features, [0, 0, 1, 1], features, [0, 0, 1, 1]))
        scores = features @ result.weights + result.bias
        np.testing.assert_allclose(result.probabilities, 1 / (1 + np.exp(-scores)))
        np.testing.assert_array_equal(result.predictions, [0, 0, 1, 1])
        self.assertEqual(len(result.loss_history), 1000)
        self.assertTrue(np.all(np.diff(result.loss_history) <= 1e-10))
        self.assertLess(result.loss_history[-1], 0.02)


if __name__ == "__main__":
    unittest.main()
