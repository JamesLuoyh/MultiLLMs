"""
Unit tests for wagering training pipeline.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock, patch

from wagering.methods import EqualWagers
from wagering.aggregation import LinearPooling
from wagering.training import WageringTrainer
from wagering.core.dataset import Dataset


class MockWhiteboxModel:
    """Mock WhiteboxModel for testing."""
    
    def __init__(self, model_id="test_model"):
        self.model_path = model_id
        self.instruct = False  # Required attribute
        self.tokenizer = Mock()
        # Mock encode to return list of token IDs (not dict)
        # Mock encode to return list of token IDs (not dict)
        # For option tokens, return single token IDs
        def mock_encode(text, add_special_tokens=False):
            # Return single token ID for option tokens (A, B, C, D)
            if text in ["A", "B", "C", "D"]:
                return [100 + ord(text) - ord("A")]  # Return [100], [101], [102], [103]
            return [100, 200, 300]  # Default for other text
        
        self.tokenizer.encode = Mock(side_effect=mock_encode)
        self.tokenizer.convert_ids_to_tokens = Mock(return_value=["A"])
        self.tokenizer.chat_template = None  # Required attribute
        self.tokenizer.apply_chat_template = Mock(return_value="")
        self.tokenizer.name_or_path = model_id
    
    def device(self):
        return torch.device("cpu")
    
    def tokenize(self, texts):
        return {
            "input_ids": torch.tensor([[1, 2, 3] * len(texts)]),
            "attention_mask": torch.ones((len(texts), 3)),
        }
    
    def generate(self, **kwargs):
        class MockGeneration:
            def __init__(self):
                # Mock scores: [batch_size, vocab_size]
                batch_size = kwargs.get("input_ids", torch.tensor([[1]])).shape[0]
                vocab_size = 1000
                self.scores = [
                    torch.randn(batch_size, vocab_size) * 0.1  # Small random logits
                ]
        
        return MockGeneration()


@pytest.fixture
def mock_models():
    """Create mock models."""
    return [MockWhiteboxModel(f"model_{i}") for i in range(2)]


@pytest.fixture
def simple_dataset():
    """Create a simple dataset."""
    x = ["Question 1: A or B?", "Question 2: C or D?"]
    y = ["A", "B"]
    return Dataset(x, y, batch_size=2)


@pytest.fixture
def wagering_method():
    """Create a wagering method."""
    return EqualWagers(num_models=2)


@pytest.fixture
def aggregation_function():
    """Create an aggregation function."""
    return LinearPooling()


class TestWageringTrainer:
    """Test WageringTrainer class."""
    
    def _make_cache_hit(self, num_examples: int, num_options: int = 4, hidden_dim: int = 32):
        """Return a side_effect function that simulates a full cache hit."""
        logits = np.random.randn(num_examples, num_options).astype(np.float32)
        hidden = np.random.randn(num_examples, hidden_dim).astype(np.float32)
        labels = np.arange(num_examples) % num_options
        return lambda *args, **kwargs: (logits, hidden, labels)

    @patch('wagering.training.trainer.get_cached_logits_and_hidden_states_for_model')
    def test_init(self, mock_cache, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test trainer initialization."""
        mock_cache.side_effect = self._make_cache_hit(len(simple_dataset.x))
        trainer = WageringTrainer(
            models=mock_models,
            datasets=[simple_dataset],
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
        )
        
        assert len(trainer.models) == 2
        assert len(trainer.combined_dataset.x) == 2
        assert trainer.wagering_method.num_models == 2
    
    @patch('wagering.training.trainer.get_cached_logits_and_hidden_states_for_model')
    def test_prepare_datasets(self, mock_cache, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test dataset preparation."""
        mock_cache.side_effect = self._make_cache_hit(len(simple_dataset.x))
        trainer = WageringTrainer(
            models=mock_models,
            datasets=[simple_dataset],
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
        )
        
        assert len(trainer.combined_dataset.x) == 2
        assert len(trainer.labels) == 2
    
    @patch('wagering.training.trainer.get_cached_logits_and_hidden_states_for_model')
    def test_collect_logits(self, mock_cache, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test logit collection."""
        mock_cache.side_effect = self._make_cache_hit(len(simple_dataset.x))
        trainer = WageringTrainer(
            models=mock_models,
            datasets=[simple_dataset],
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
        )
        
        # Check that logits were collected
        assert trainer.all_model_logits.shape[0] == len(mock_models)
        assert trainer.all_model_logits.shape[1] == len(simple_dataset.x)
    
    @patch('wagering.training.trainer.get_cached_logits_and_hidden_states_for_model')
    def test_train_single_epoch(self, mock_cache, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test training for one epoch."""
        num_examples = len(simple_dataset.x)
        mock_cache.side_effect = self._make_cache_hit(num_examples)
        
        trainer = WageringTrainer(
            models=mock_models,
            datasets=[simple_dataset],
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            checkpoint_dir=None,  # No checkpointing for test
            validation_split_ratio=0.0,  # No validation split for tiny test dataset
        )
        
        # Train for 1 epoch
        results = trainer.train(num_epochs=1)
        
        # Check results
        assert "final_accuracy" in results
        assert "predictions" in results
        assert "wagers_history" in results
        assert len(results["predictions"]) == num_examples
    
    @patch('wagering.training.trainer.get_cached_logits_and_hidden_states_for_model')
    def test_save_checkpoint(self, mock_cache, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test checkpoint saving."""
        mock_cache.side_effect = self._make_cache_hit(len(simple_dataset.x))
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WageringTrainer(
                models=mock_models,
                datasets=[simple_dataset],
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
                checkpoint_dir=tmpdir,
            )
            
            # Trigger checkpoint save (method must exist and not crash)
            trainer._save_checkpoint(epoch=0)
    
    @patch('wagering.training.trainer.get_cached_logits_and_hidden_states_for_model')
    def test_save_final_checkpoint(self, mock_cache, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test final checkpoint saving."""
        mock_cache.side_effect = self._make_cache_hit(len(simple_dataset.x))
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WageringTrainer(
                models=mock_models,
                datasets=[simple_dataset],
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
            )
            
            trainer.save_final_checkpoint(tmpdir)
            
            # Check that wagering state was saved
            state_file = Path(tmpdir) / "wagering_state.pt"
            assert state_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
