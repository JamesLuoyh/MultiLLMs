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
from wagering.methods.factory import load_wagering_method
from wagering.methods.route_llm_bert import RouteLLMBertWagers
from wagering.methods.router_dc import RouterDCWagers
from wagering.aggregation import LinearPooling
from wagering.training import WageringTrainer
from wagering.training.trainer import compute_meta_metrics
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


class TestRouteLLMBertWagers:
    """RouteLLM BERT router: prompt encoding + linear head (arXiv:2406.18665)."""

    def test_factory_compute_update_state_dict(self):
        pytest.importorskip("transformers")
        config = {
            "bert_model_name": "hf-internal-testing/tiny-random-bert",
            "max_seq_length": 32,
            "learning_rate": 1e-3,
            "temperature": 2.0,
            "grad_clip_norm": 1.0,
            "freeze_bert": True,
            "device": "cpu",
            "lr_decay_factor": 1.0,
            "lr_decay_steps": 10,
            "ranking_loss_weight": 0.0,
        }
        m = load_wagering_method("route_llm_bert", num_models=2, config=config)
        assert isinstance(m, RouteLLMBertWagers)

        m.train_mode()
        batch_size = 2
        num_models = 2
        num_options = 2
        questions = ["Question: test one?", "Question: test two?"]
        model_logits = np.random.randn(batch_size, num_models, num_options).astype(np.float32)
        gold_label = np.array([0, 1], dtype=np.int64)

        res = m.compute_wagers(questions=questions, model_logits=model_logits)
        assert res["wagers"].shape == (batch_size, num_models)

        max_logits = np.max(model_logits, axis=2, keepdims=True)
        stabilized = model_logits - max_logits
        model_probs = np.exp(stabilized) / (np.sum(np.exp(stabilized), axis=2, keepdims=True) + 1e-20)
        agg = np.ones((batch_size, num_options), dtype=np.float32) * 0.5
        pred = np.zeros(batch_size, dtype=np.int64)

        info = m.update(
            aggregated_probs=agg,
            aggregated_pred=pred,
            gold_label=gold_label,
            model_probs=model_probs,
            model_logits=model_logits,
        )
        assert "loss" in info
        assert info["batch_size"] == batch_size

        state = m.state_dict()
        assert "scheduler_state_dict" in state
        m2 = load_wagering_method("route_llm_bert", num_models=2, config=config)
        m2.load_state_dict(state)
        m2.eval_mode()


class TestRouterDCWagers:
    """RouterDC: encoder + expert embeddings + sample–LLM contrastive (arXiv:2409.19886)."""

    def test_factory_compute_update_state_dict(self):
        pytest.importorskip("transformers")
        config = {
            "encoder_model_name": "hf-internal-testing/tiny-random-bert",
            "max_seq_length": 32,
            "learning_rate": 1e-3,
            "temperature": 1.0,
            "grad_clip_norm": 1.0,
            "freeze_encoder": True,
            "device": "cpu",
            "lr_decay_factor": 1.0,
            "lr_decay_steps": 10,
            "top_k": 2,
            "last_k": 2,
            "similarity_function": "cos",
        }
        m = load_wagering_method("router_dc", num_models=2, config=config)
        assert isinstance(m, RouterDCWagers)
        m_alias = load_wagering_method("routerDC", num_models=2, config=config)
        assert isinstance(m_alias, RouterDCWagers)

        m.train_mode()
        batch_size = 2
        num_models = 2
        num_options = 2
        questions = ["Question: test one?", "Question: test two?"]
        model_logits = np.random.randn(batch_size, num_models, num_options).astype(np.float32)
        gold_label = np.array([0, 1], dtype=np.int64)

        res = m.compute_wagers(questions=questions, model_logits=model_logits)
        w = res["wagers"]
        assert w.shape == (batch_size, num_models)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)

        max_logits = np.max(model_logits, axis=2, keepdims=True)
        stabilized = model_logits - max_logits
        model_probs = np.exp(stabilized) / (np.sum(np.exp(stabilized), axis=2, keepdims=True) + 1e-20)
        agg = np.ones((batch_size, num_options), dtype=np.float32) * 0.5
        pred = np.zeros(batch_size, dtype=np.int64)

        info = m.update(
            aggregated_probs=agg,
            aggregated_pred=pred,
            gold_label=gold_label,
            model_probs=model_probs,
            model_logits=model_logits,
        )
        assert "loss" in info
        assert np.isfinite(info["loss"])
        assert info["batch_size"] == batch_size

        state = m.state_dict()
        assert "expert_embeddings_state_dict" in state
        assert "scheduler_state_dict" in state
        m2 = load_wagering_method("routerdc", num_models=2, config=config)
        m2.load_state_dict(state)
        m2.eval_mode()


class TestMetaMetrics:
    """Test helper metrics computed from wagers and model ranking targets."""

    def test_kendall_tau_and_best_model_mrr(self):
        wagers = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
            ],
            dtype=np.float64,
        )
        best_expert_ids = np.array([0, 1, 2], dtype=np.int64)
        model_brier_scores = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.1, 0.2],
                [0.2, 0.3, 0.1],
            ],
            dtype=np.float64,
        )

        metrics = compute_meta_metrics(
            wagers=wagers,
            best_expert_ids=best_expert_ids,
            model_brier_scores=model_brier_scores,
        )

        assert metrics["kendall_tau"] == pytest.approx(1.0 / 9.0, abs=1e-9)
        assert metrics["best_model_mrr"] == pytest.approx(2.0 / 3.0, abs=1e-9)

    def test_kendall_tau_with_ties_uses_total_pairs_denominator(self):
        wagers = np.array([[0.6, 0.2, 0.2]], dtype=np.float64)
        best_expert_ids = np.array([0], dtype=np.int64)
        model_brier_scores = np.array([[0.1, 0.1, 0.3]], dtype=np.float64)

        metrics = compute_meta_metrics(
            wagers=wagers,
            best_expert_ids=best_expert_ids,
            model_brier_scores=model_brier_scores,
        )

        # One concordant and zero discordant out of 3 total model pairs.
        assert metrics["kendall_tau"] == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert metrics["best_model_mrr"] == pytest.approx(1.0, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
