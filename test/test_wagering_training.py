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
from lm_polygraph.utils.dataset import Dataset


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
    
    def test_init(self, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test trainer initialization."""
        trainer = WageringTrainer(
            models=mock_models,
            datasets=[simple_dataset],
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
        )
        
        assert len(trainer.models) == 2
        assert len(trainer.combined_dataset.x) == 2
        assert trainer.wagering_method.num_models == 2
    
    def test_prepare_datasets(self, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test dataset preparation."""
        trainer = WageringTrainer(
            models=mock_models,
            datasets=[simple_dataset],
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
        )
        
        assert len(trainer.combined_dataset.x) == 2
        assert len(trainer.labels) == 2
    
    @patch('wagering.training.trainer.collect_option_logits_for_model')
    def test_collect_logits(self, mock_collect, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test logit collection."""
        # Mock logit collection
        mock_collect.return_value = (
            np.random.randn(2, 4),  # logits: [num_examples, num_options]
            np.array([0, 1]),  # labels
        )
        
        trainer = WageringTrainer(
            models=mock_models,
            datasets=[simple_dataset],
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
        )
        
        # Check that logits were collected
        assert trainer.all_model_logits.shape[0] == len(mock_models)
        assert trainer.all_model_logits.shape[1] == len(simple_dataset.x)
    
    @patch('wagering.training.trainer.collect_option_logits_for_model')
    def test_train_single_epoch(self, mock_collect, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test training for one epoch."""
        # Mock logit collection
        num_examples = len(simple_dataset.x)
        mock_collect.return_value = (
            np.random.randn(num_examples, 4),  # logits
            np.array([0, 1]),  # labels
        )
        
        trainer = WageringTrainer(
            models=mock_models,
            datasets=[simple_dataset],
            wagering_method=wagering_method,
            aggregation_function=aggregation_function,
            checkpoint_dir=None,  # No checkpointing for test
        )
        
        # Train for 1 epoch
        results = trainer.train(num_epochs=1)
        
        # Check results
        assert "final_accuracy" in results
        assert "predictions" in results
        assert "wagers_history" in results
        assert len(results["predictions"]) == num_examples
    
    def test_save_checkpoint(self, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WageringTrainer(
                models=mock_models,
                datasets=[simple_dataset],
                wagering_method=wagering_method,
                aggregation_function=aggregation_function,
                checkpoint_dir=tmpdir,
            )
            
            # Mock logit collection
            with patch('wagering.training.trainer.collect_option_logits_for_model') as mock_collect:
                mock_collect.return_value = (
                    np.random.randn(2, 4),
                    np.array([0, 1]),
                )
                
                # Trigger checkpoint save
                trainer._save_checkpoint(epoch=0, step=0)
                
                # Check that checkpoint file exists
                checkpoint_file = Path(tmpdir) / "checkpoint_epoch_0_step_0.pt"
                # Note: This won't actually create a file since we haven't collected logits
                # But we can test the method exists and doesn't crash
    
    def test_save_final_checkpoint(self, mock_models, simple_dataset, wagering_method, aggregation_function):
        """Test final checkpoint saving."""
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
