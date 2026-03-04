"""
Unit tests for wagering methods and aggregation functions.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from wagering.methods import WageringMethod, EqualWagers
from wagering.aggregation import AggregationFunction, LinearPooling, LogarithmicPooling
from wagering.methods.factory import load_wagering_method
from wagering.aggregation.factory import load_aggregation_function


class TestEqualWagers:
    """Test EqualWagers wagering method."""
    
    def test_init(self):
        """Test initialization."""
        method = EqualWagers(num_models=3)
        assert method.num_models == 3
        assert method.config == {}
    
    def test_compute_wagers(self):
        """Test wager computation."""
        method = EqualWagers(num_models=4)
        wagers = method.compute_wagers()
        
        assert wagers.shape == (4,)
        assert np.allclose(wagers, 0.25)
        assert np.allclose(wagers.sum(), 1.0)
    
    def test_compute_wagers_with_kwargs(self):
        """Test wager computation with optional kwargs."""
        method = EqualWagers(num_models=3)
        wagers = method.compute_wagers(
            question="test",
            models=None,
            model_logits=np.random.randn(3, 4),
        )
        
        assert wagers.shape == (3,)
        assert np.allclose(wagers, 1.0 / 3.0)
    
    def test_update(self):
        """Test update method (should be no-op for equal wagers)."""
        method = EqualWagers(num_models=2)
        result = method.update(
            aggregated_probs=np.array([0.3, 0.7]),
            aggregated_pred=1,
            gold_label=1,
            model_probs=np.random.rand(2, 2),
            model_logits=np.random.randn(2, 2),
        )
        
        assert result == {}
    
    def test_state_dict(self):
        """Test state dict (should be empty for equal wagers)."""
        method = EqualWagers(num_models=3)
        state = method.state_dict()
        assert state == {}
    
    def test_save_load(self):
        """Test save and load."""
        method = EqualWagers(num_models=5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            method.save_pretrained(tmpdir)
            
            # Create new method and load
            method2 = EqualWagers(num_models=5)
            method2.load_pretrained(tmpdir)
            
            # State should be same (empty for equal wagers)
            assert method2.state_dict() == method.state_dict()


class TestLinearPooling:
    """Test LinearPooling aggregation function."""
    
    def test_aggregate_basic(self):
        """Test basic aggregation."""
        aggregator = LinearPooling()
        
        # 2 models, 3 options
        model_logits = np.array([
            [1.0, 2.0, 3.0],  # Model 1
            [2.0, 1.0, 3.0],  # Model 2
        ])
        wagers = np.array([0.5, 0.5])
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        assert log_probs.shape == (3,)
        assert probs.shape == (3,)
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_aggregate_unequal_weights(self):
        """Test aggregation with unequal weights."""
        aggregator = LinearPooling()
        
        model_logits = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
        ])
        wagers = np.array([0.8, 0.2])  # Model 1 gets more weight
        
        log_probs, probs = aggregator.aggregate(model_logits, wagers)
        
        assert probs.shape == (3,)
        assert np.allclose(probs.sum(), 1.0)
    
    def test_aggregate_shape_validation(self):
        """Test that shape validation works."""
        aggregator = LinearPooling()
        
        # Wrong number of models
        model_logits = np.array([[1.0, 2.0, 3.0]])
        wagers = np.array([0.5, 0.5])
        
        with pytest.raises(ValueError):
            aggregator.aggregate(model_logits, wagers)
        
        # Wrong shape for logits
        model_logits = np.array([1.0, 2.0, 3.0])
        wagers = np.array([1.0])
        
        with pytest.raises(ValueError):
            aggregator.aggregate(model_logits, wagers)
    
    def test_aggregate_wager_sum_validation(self):
        """Test that wager sum validation works."""
        aggregator = LinearPooling()
        
        model_logits = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
        ])
        
        # Wagers that don't sum to 1
        wagers = np.array([0.5, 0.4])
        with pytest.raises(ValueError, match="wagers must sum to exactly 1.0"):
            aggregator.aggregate(model_logits, wagers)
        
        # Wagers that sum to more than 1
        wagers = np.array([0.6, 0.5])
        with pytest.raises(ValueError, match="wagers must sum to exactly 1.0"):
            aggregator.aggregate(model_logits, wagers)


class TestWageringFactory:
    """Test wagering method factory."""
    
    def test_load_equal_wagers(self):
        """Test loading equal wagers method."""
        method = load_wagering_method("equal_wagers", num_models=3)
        assert isinstance(method, EqualWagers)
        assert method.num_models == 3
    
    def test_load_equal_wagers_alias(self):
        """Test loading with alias."""
        method = load_wagering_method("equal", num_models=4)
        assert isinstance(method, EqualWagers)
        assert method.num_models == 4
    
    def test_load_unknown_method(self):
        """Test loading unknown method raises error."""
        with pytest.raises(ValueError):
            load_wagering_method("unknown_method", num_models=2)


class TestAggregationFactory:
    """Test aggregation function factory."""
    
    def test_load_linear_pooling(self):
        """Test loading linear pooling."""
        aggregator = load_aggregation_function("weighted_linear_pooling")
        assert isinstance(aggregator, LinearPooling)
        
        aggregator = load_aggregation_function("linear_pooling")
        assert isinstance(aggregator, LinearPooling)
    
    def test_load_logarithmic_pooling(self):
        """Test loading logarithmic pooling."""
        aggregator = load_aggregation_function("logarithmic_pooling")
        assert isinstance(aggregator, LogarithmicPooling)
        
        aggregator = load_aggregation_function("weighted_log_pooling")
        assert isinstance(aggregator, LogarithmicPooling)
        
        aggregator = load_aggregation_function("log_pooling")
        assert isinstance(aggregator, LogarithmicPooling)
    
    def test_load_unknown_method(self):
        """Test loading unknown method raises error."""
        with pytest.raises(ValueError):
            load_aggregation_function("unknown_method")


class TestIntegration:
    """Integration tests for wagering and aggregation."""
    
    def test_wagering_and_aggregation_together(self):
        """Test wagering method and aggregation function together."""
        num_models = 3
        num_options = 4
        
        # Create wagering method
        wagering = EqualWagers(num_models=num_models)
        
        # Create aggregation function
        aggregation = LinearPooling()
        
        # Simulate model logits
        model_logits = np.random.randn(num_models, num_options)
        
        # Get wagers
        wagers = wagering.compute_wagers(model_logits=model_logits)
        
        # Aggregate
        log_probs, probs = aggregation.aggregate(model_logits, wagers)
        
        # Check results
        assert probs.shape == (num_options,)
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_training_loop_simulation(self):
        """Simulate a simple training loop."""
        num_models = 2
        num_options = 3
        num_examples = 10
        
        wagering = EqualWagers(num_models=num_models)
        aggregation = LinearPooling()
        
        all_predictions = []
        all_wagers = []
        
        for step in range(num_examples):
            # Generate random logits
            model_logits = np.random.randn(num_models, num_options)
            
            # Get wagers
            wagers = wagering.compute_wagers(model_logits=model_logits)
            all_wagers.append(wagers)
            
            # Aggregate
            _, probs = aggregation.aggregate(model_logits, wagers)
            pred = int(np.argmax(probs))
            all_predictions.append(pred)
            
            # Update (no-op for equal wagers)
            gold_label = step % num_options
            wagering.update(
                aggregated_probs=probs,
                aggregated_pred=pred,
                gold_label=gold_label,
                model_probs=probs.reshape(1, -1).repeat(num_models, axis=0),
                model_logits=model_logits,
            )
        
        # Check that we got predictions
        assert len(all_predictions) == num_examples
        assert len(all_wagers) == num_examples
        
        # Check wagers are consistent (equal wagers)
        for wagers in all_wagers:
            assert np.allclose(wagers, 1.0 / num_models)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

