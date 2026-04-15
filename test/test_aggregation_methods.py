"""
Comprehensive unit tests for aggregation methods.
"""

import pytest
import numpy as np

from wagering.aggregation import LinearPooling, LogarithmicPooling, MajorityVote
from wagering.aggregation.factory import load_aggregation_function


class TestLinearPooling:
    """Test LinearPooling aggregation function."""
    
    def test_basic_aggregation(self):
        """Test basic aggregation with equal weights."""
        aggregator = LinearPooling()
        
        # 2 models, 3 options
        model_logits = np.array([
            [1.0, 2.0, 3.0],  # Model 1: prefers option 2
            [2.0, 1.0, 3.0],  # Model 2: prefers option 2
        ], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Check shapes
        assert log_probs.shape == (3,)
        assert probs.shape == (3,)
        
        # Check probability properties
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
        
        # Check log-probabilities
        assert np.allclose(np.exp(log_probs), probs, atol=1e-6)
        
        # Both models prefer option 2, so aggregated should also prefer option 2
        assert np.argmax(probs) == 2
    
    def test_unequal_weights(self):
        """Test aggregation with unequal weights."""
        aggregator = LinearPooling()
        
        # Model 1 strongly prefers option 0, Model 2 prefers option 1
        model_logits = np.array([
            [5.0, 0.0, 0.0],  # Model 1: very confident in option 0
            [0.0, 5.0, 0.0],  # Model 2: very confident in option 1
        ], dtype=np.float32)
        
        # Give more weight to model 1
        wagers = np.array([0.8, 0.2], dtype=np.float32)
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Should prefer option 0 due to higher weight on model 1
        assert np.argmax(probs) == 0
        assert probs[0] > probs[1]
        
        # Check normalization
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
    
    def test_single_model(self):
        """Test with single model."""
        aggregator = LinearPooling()
        
        model_logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        wagers = np.array([1.0], dtype=np.float32)
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Should just be softmax of the single model
        expected_probs = np.exp(model_logits[0] - np.max(model_logits[0]))
        expected_probs = expected_probs / expected_probs.sum()
        
        assert np.allclose(probs, expected_probs, atol=1e-6)
    
    def test_many_models(self):
        """Test with many models."""
        aggregator = LinearPooling()
        
        num_models = 5
        num_options = 4
        model_logits = np.random.randn(num_models, num_options).astype(np.float32)
        wagers = np.ones(num_models, dtype=np.float32) / num_models
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        assert probs.shape == (num_options,)
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
    
    def test_wager_normalization_and_validation(self):
        """Test internal wager normalization and wager validity checks."""
        aggregator = LinearPooling()
        
        model_logits = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
        ], dtype=np.float32)
        
        # Non-normalized but non-negative wagers should be accepted and normalized.
        normalized_wagers = np.array([0.8, 0.2], dtype=np.float32)
        _, expected_probs = aggregator.aggregate(model_logits, normalized_wagers)

        scaled_wagers = np.array([8.0, 2.0], dtype=np.float32)
        _, scaled_probs = aggregator.aggregate(model_logits, scaled_wagers)
        assert np.allclose(scaled_probs, expected_probs, atol=1e-6)

        # Zero-sum wagers are invalid.
        zero_wagers = np.array([0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="positive sum"):
            aggregator.aggregate(model_logits, zero_wagers)

        # Negative wagers are invalid.
        negative_wagers = np.array([0.5, -0.1], dtype=np.float32)
        with pytest.raises(ValueError, match="non-negative"):
            aggregator.aggregate(model_logits, negative_wagers)
    
    def test_shape_validation(self):
        """Test that shape validation works."""
        aggregator = LinearPooling()
        
        # Wrong number of models in wagers
        model_logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        with pytest.raises(ValueError, match="wagers must have shape"):
            aggregator.aggregate(model_logits, wagers)
        
        # Wrong shape for logits (1D instead of 2D)
        model_logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        wagers = np.array([1.0], dtype=np.float32)
        
        with pytest.raises(ValueError, match="model_logits must be either"):
            aggregator.aggregate(model_logits, wagers)
    
    def test_extreme_logits(self):
        """Test with extreme logit values."""
        aggregator = LinearPooling()
        
        # Very large logits
        model_logits = np.array([
            [100.0, 0.0, -100.0],
            [0.0, 100.0, -100.0],
        ], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Should still be normalized
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        
        # Very negative logits
        model_logits = np.array([
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Should be uniform (all logits equal)
        assert np.allclose(probs, 1.0 / 3.0, atol=1e-5)


class TestLogarithmicPooling:
    """Test LogarithmicPooling aggregation function."""
    
    def test_basic_aggregation(self):
        """Test basic aggregation with equal weights."""
        aggregator = LogarithmicPooling()
        
        # 2 models, 3 options
        model_logits = np.array([
            [1.0, 2.0, 3.0],  # Model 1: prefers option 2
            [2.0, 1.0, 3.0],  # Model 2: prefers option 2
        ], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Check shapes
        assert log_probs.shape == (3,)
        assert probs.shape == (3,)
        
        # Check probability properties
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
        
        # Check log-probabilities
        assert np.allclose(np.exp(log_probs), probs, atol=1e-6)
        
        # Both models prefer option 2, so aggregated should also prefer option 2
        assert np.argmax(probs) == 2
    
    def test_unequal_weights(self):
        """Test aggregation with unequal weights."""
        aggregator = LogarithmicPooling()
        
        # Model 1 strongly prefers option 0, Model 2 prefers option 1
        model_logits = np.array([
            [5.0, 0.0, 0.0],  # Model 1: very confident in option 0
            [0.0, 5.0, 0.0],  # Model 2: very confident in option 1
        ], dtype=np.float32)
        
        # Give more weight to model 1
        wagers = np.array([0.8, 0.2], dtype=np.float32)
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Should prefer option 0 due to higher weight on model 1
        assert np.argmax(probs) == 0
        assert probs[0] > probs[1]
        
        # Check normalization
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
    
    def test_single_model(self):
        """Test with single model."""
        aggregator = LogarithmicPooling()
        
        model_logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        wagers = np.array([1.0], dtype=np.float32)
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Should just be softmax of the single model
        expected_probs = np.exp(model_logits[0] - np.max(model_logits[0]))
        expected_probs = expected_probs / expected_probs.sum()
        
        assert np.allclose(probs, expected_probs, atol=1e-6)
    
    def test_many_models(self):
        """Test with many models."""
        aggregator = LogarithmicPooling()
        
        num_models = 5
        num_options = 4
        model_logits = np.random.randn(num_models, num_options).astype(np.float32)
        wagers = np.ones(num_models, dtype=np.float32) / num_models
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        assert probs.shape == (num_options,)
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
    
    def test_wager_normalization_and_validation(self):
        """Test internal wager normalization and wager validity checks."""
        aggregator = LogarithmicPooling()
        
        model_logits = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
        ], dtype=np.float32)
        
        # Non-normalized but non-negative wagers should be accepted and normalized.
        normalized_wagers = np.array([0.8, 0.2], dtype=np.float32)
        _, expected_probs = aggregator.aggregate(model_logits, normalized_wagers)

        scaled_wagers = np.array([8.0, 2.0], dtype=np.float32)
        _, scaled_probs = aggregator.aggregate(model_logits, scaled_wagers)
        assert np.allclose(scaled_probs, expected_probs, atol=1e-6)

        # Zero-sum wagers are invalid.
        zero_wagers = np.array([0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="positive sum"):
            aggregator.aggregate(model_logits, zero_wagers)

        # Negative wagers are invalid.
        negative_wagers = np.array([0.5, -0.1], dtype=np.float32)
        with pytest.raises(ValueError, match="non-negative"):
            aggregator.aggregate(model_logits, negative_wagers)
    
    def test_shape_validation(self):
        """Test that shape validation works."""
        aggregator = LogarithmicPooling()
        
        # Wrong number of models in wagers
        model_logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        with pytest.raises(ValueError, match="wagers must have shape"):
            aggregator.aggregate(model_logits, wagers)
        
        # Wrong shape for logits (1D instead of 2D)
        model_logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        wagers = np.array([1.0], dtype=np.float32)
        
        with pytest.raises(ValueError, match="model_logits must be either"):
            aggregator.aggregate(model_logits, wagers)
    
    def test_extreme_logits(self):
        """Test with extreme logit values."""
        aggregator = LogarithmicPooling()
        
        # Very large logits
        model_logits = np.array([
            [100.0, 0.0, -100.0],
            [0.0, 100.0, -100.0],
        ], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Should still be normalized
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        
        # Very negative logits
        model_logits = np.array([
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        # Should be uniform (all logits equal)
        assert np.allclose(probs, 1.0 / 3.0, atol=1e-5)
    
    def test_difference_from_linear_pooling(self):
        """Test that logarithmic pooling produces different results from linear pooling."""
        linear_agg = LinearPooling()
        log_agg = LogarithmicPooling()
        
        # Use logits where the difference should be noticeable
        model_logits = np.array([
            [2.0, 0.0, 0.0],  # Model 1: confident in option 0
            [0.0, 2.0, 0.0],  # Model 2: confident in option 1
        ], dtype=np.float32)
        wagers = np.array([0.5, 0.5], dtype=np.float32)
        
        linear_log_probs, linear_probs = linear_agg.aggregate(model_logits, wagers)
        log_log_probs, log_probs = log_agg.aggregate(model_logits, wagers)
        
        # Both should be valid probability distributions
        assert np.allclose(linear_probs.sum(), 1.0, atol=1e-6)
        assert np.allclose(log_probs.sum(), 1.0, atol=1e-6)
        
        # They should produce different results (log pooling is more "conservative")
        # In this case, both should give roughly equal weight to options 0 and 1
        # but the exact values will differ
        assert not np.allclose(linear_probs, log_probs, atol=1e-3)


class TestAggregationComparison:
    """Test comparison between linear and logarithmic pooling."""
    
    def test_consistency_properties(self):
        """Test that both methods maintain consistency properties."""
        linear_agg = LinearPooling()
        log_agg = LogarithmicPooling()
        
        model_logits = np.random.randn(3, 5).astype(np.float32)
        wagers = np.array([0.3, 0.3, 0.4], dtype=np.float32)
        
        linear_log_probs, linear_probs = linear_agg.aggregate(model_logits, wagers)
        log_log_probs, log_probs = log_agg.aggregate(model_logits, wagers)
        
        # Both should produce valid probability distributions
        assert np.allclose(linear_probs.sum(), 1.0, atol=1e-6)
        assert np.allclose(log_probs.sum(), 1.0, atol=1e-6)


class TestMajorityVote:
    """Test MajorityVote aggregation function."""

    def test_single_sample_vote_fractions(self):
        """Test that vote fractions match argmax counts in single-sample mode."""
        aggregator = MajorityVote()

        # 10 models, 4 options; option 0 gets 2 votes, option 1 gets 3,
        # option 2 gets 5, option 3 gets 0.
        model_logits = np.array([
            [10.0, 0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ], dtype=np.float32)
        wagers = np.ones(10, dtype=np.float32) / 10.0

        log_probs, probs = aggregator.aggregate(model_logits, wagers)

        expected = np.array([0.2, 0.3, 0.5, 0.0], dtype=np.float32)
        assert np.allclose(probs, expected, atol=1e-6)
        assert np.allclose(np.exp(log_probs[:3]), expected[:3], atol=1e-6)
        assert np.argmax(probs) == 2
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

    def test_batch_vote_fractions(self):
        """Test majority voting in batch mode."""
        aggregator = MajorityVote()

        # batch_size=2, num_models=4, num_options=3
        # Example 0 argmaxes: [0, 0, 1, 2] -> [0.5, 0.25, 0.25]
        # Example 1 argmaxes: [1, 1, 1, 2] -> [0.0, 0.75, 0.25]
        model_logits = np.array([
            [[3.0, 1.0, 0.0], [2.0, 0.5, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0]],
            [[0.0, 2.0, 1.0], [0.0, 3.0, 1.0], [1.0, 4.0, 0.0], [0.0, 1.0, 3.0]],
        ], dtype=np.float32)
        wagers = np.ones((2, 4), dtype=np.float32) / 4.0

        _, probs = aggregator.aggregate(model_logits, wagers)

        expected = np.array([
            [0.5, 0.25, 0.25],
            [0.0, 0.75, 0.25],
        ], dtype=np.float32)
        assert np.allclose(probs, expected, atol=1e-6)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    
    def test_uniform_wagers(self):
        """Test that uniform wagers work correctly."""
        linear_agg = LinearPooling()
        log_agg = LogarithmicPooling()
        
        num_models = 4
        num_options = 3
        model_logits = np.random.randn(num_models, num_options).astype(np.float32)
        wagers = np.ones(num_models, dtype=np.float32) / num_models
        
        linear_log_probs, linear_probs = linear_agg.aggregate(model_logits, wagers)
        log_log_probs, log_probs = log_agg.aggregate(model_logits, wagers)
        
        # Both should work with uniform wagers
        assert np.allclose(linear_probs.sum(), 1.0, atol=1e-6)
        assert np.allclose(log_probs.sum(), 1.0, atol=1e-6)


class TestFactory:
    """Test aggregation function factory."""
    
    def test_load_linear_pooling(self):
        """Test loading linear pooling."""
        aggregator = load_aggregation_function("linear_pooling")
        assert isinstance(aggregator, LinearPooling)
        
        aggregator = load_aggregation_function("weighted_linear_pooling")
        assert isinstance(aggregator, LinearPooling)
    
    def test_load_logarithmic_pooling(self):
        """Test loading logarithmic pooling."""
        aggregator = load_aggregation_function("logarithmic_pooling")
        assert isinstance(aggregator, LogarithmicPooling)
        
        aggregator = load_aggregation_function("weighted_log_pooling")
        assert isinstance(aggregator, LogarithmicPooling)
        
        aggregator = load_aggregation_function("log_pooling")
        assert isinstance(aggregator, LogarithmicPooling)

    def test_load_majority_vote(self):
        """Test loading majority vote aggregation aliases."""
        aggregator = load_aggregation_function("majority_vote")
        assert isinstance(aggregator, MajorityVote)

        aggregator = load_aggregation_function("majority_voting")
        assert isinstance(aggregator, MajorityVote)

        aggregator = load_aggregation_function("argmax_vote")
        assert isinstance(aggregator, MajorityVote)
    
    def test_load_unknown_method(self):
        """Test loading unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation function"):
            load_aggregation_function("unknown_method")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


