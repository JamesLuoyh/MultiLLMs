"""
Mixture of Experts (MoE) wagers implementation: uses question embeddings to route to models.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import AutoModel, AutoTokenizer

# Add src/ to path for lm_polygraph imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from .base import WageringMethod
from lm_polygraph.utils.model import WhiteboxModel
from wagering.aggregation.linear_pooling import LinearPooling


class MoEWagers(WageringMethod):
    """
    Mixture of Experts wagering method that routes based on question embeddings.
    
    Instead of using hidden states from all LLMs, this method:
    1. Encodes the question using a pre-trained BERT model
    2. Passes the question embedding through an MLP router
    3. Outputs a probability distribution over models via softmax
    
    This is simpler than centralized wagers as it doesn't require projection layers.
    """
    
    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MoE wagers method.
        
        Args:
            num_models: Number of LLMs in the ensemble
            config: Configuration dictionary with:
                - bert_model: BERT model name (default: 'bert-base-uncased')
                - embedding_dim: Dimension of BERT embeddings (default: 768)
                - hidden_layers: List of hidden layer sizes for router (default: [512, 256])
                - learning_rate: Learning rate for optimizer (default: 1e-4)
                - temperature: Temperature for softmax (default: 2.0)
                - grad_clip_norm: Gradient clipping norm (default: 1.0)
                - freeze_bert: Whether to freeze BERT weights (default: True)
                - device: Device to run on (default: 'cuda' if available, else 'cpu')
        """
        super().__init__(num_models, config)
        
        if config is None:
            config = {}
        
        # Get configuration
        self.bert_model_name = str(config.get("bert_model", "bert-base-uncased"))
        self.embedding_dim = int(config.get("embedding_dim", 768))
        self.hidden_layers = list(config.get("hidden_layers", [512, 256]))
        self.learning_rate = float(config.get("learning_rate", 1e-4))
        self.temperature = float(config.get("temperature", 2.0))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        self.freeze_bert = config.get("freeze_bert", True)
        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)
        
        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert = AutoModel.from_pretrained(self.bert_model_name).to(self.device)
        
        # Freeze BERT if configured
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            # Set BERT to eval mode to disable dropout and batch norm updates
            self.bert.eval()
        
        # Build MLP router
        # Input: BERT embedding (embedding_dim)
        # Output: logits for each model (num_models)
        layers = []
        prev_dim = self.embedding_dim
        
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: outputs logits for num_models
        layers.append(nn.Linear(prev_dim, num_models))
        
        self.router = nn.Sequential(*layers).to(self.device)
        
        # Optimizer (only for router parameters, BERT is frozen)
        trainable_params = list(self.router.parameters())
        if not self.freeze_bert:
            trainable_params.extend(list(self.bert.parameters()))
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        
        # Training mode flag
        self._training = True
        
        # Cache for computed values during training
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_question_embedding: Optional[torch.Tensor] = None
    
    def _encode_question(self, question: str) -> torch.Tensor:
        """
        Encode a question using BERT model.
        
        Args:
            question: The question text
            
        Returns:
            embedding: torch.Tensor of shape [embedding_dim] (CLS token embedding)
        """
        # Tokenize the question
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)
        
        # Get BERT output
        with torch.set_grad_enabled(self._training and not self.freeze_bert):
            outputs = self.bert(**inputs)
        
        # Use CLS token embedding as question representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, embedding_dim]
        
        return cls_embedding.squeeze(0)  # [embedding_dim]
    
    def _encode_questions_batch(self, questions: List[str]) -> torch.Tensor:
        """
        Encode a batch of questions using BERT model.
        
        Args:
            questions: List of question texts
            
        Returns:
            embeddings: torch.Tensor of shape [batch_size, embedding_dim]
        """
        # Tokenize the batch
        inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)
        
        # Get BERT output
        with torch.set_grad_enabled(self._training and not self.freeze_bert):
            outputs = self.bert(**inputs)
        
        # Use CLS token embedding as question representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, embedding_dim]
        
        return cls_embeddings
    
    
    def compute_wagers(
        self,
        questions: List[str],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute wagers for a batch of questions using MoE router.
        
        Args:
            questions: List of question texts
            model_logits: Optional, not used
            gold_label: Optional, ground-truth labels (used by methods that compute label-aware diagnostics)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with:
                "wagers": np.ndarray of shape [batch_size, num_models] with probabilities (sum to 1 for each row)
        """
        batch_size = len(questions)
        
        # Encode all questions in batch
        question_embeddings = self._encode_questions_batch(questions)  # [batch_size, embedding_dim]
        
        # Route through MoE router
        self.router.eval() if not self._training else self.router.train()
        with torch.set_grad_enabled(self._training):
            logits = self.router(question_embeddings)  # [batch_size, num_models]
            wagers = torch.softmax(logits / self.temperature, dim=1)  # [batch_size, num_models]
        
        if self._training:
            self._cached_wagers = wagers
            self._cached_question_embedding = question_embeddings
        
        wagers_np = wagers.detach().cpu().numpy()  # [batch_size, num_models]
        return {"wagers": wagers_np}
    
    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        hidden_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update for a single sample."""
        batch_size = model_logits.shape[0]
        
        # Reuse cached values from compute_wagers() if available
        if (self._cached_wagers is not None and 
            self._cached_question_embedding is not None):
            wagers = self._cached_wagers  # [1, num_models] or [batch_size, num_models]
            question_embedding = self._cached_question_embedding
        else:
            raise ValueError(
                "MoEWagers.update() requires cached wagers from compute_wagers(). "
                "Please ensure compute_wagers() is called before update()."
            )
        
        # Clear cache after use
        self._cached_wagers = None
        self._cached_question_embedding = None
        
        # Handle shape: if wagers is [1, num_models] and batch_size > 1,
        # we need to recompute or expand
        if wagers.shape[0] == 1 and batch_size > 1:
            # Single question case - expand wagers to match batch
            wagers = wagers.expand(batch_size, -1)
        
        # Convert model_logits to tensor: [batch_size, num_models, num_options]
        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32).to(self.device)
        
        # Compute aggregated probabilities for entire batch
        batch_aggregated_probs = LinearPooling.aggregate_torch(
            model_logits_tensor, wagers
        )  # [batch_size, num_options]
        
        # Compute cross-entropy loss for each sample
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long).to(self.device)
        batch_indices = torch.arange(batch_size, device=self.device)
        probs_at_gold = batch_aggregated_probs[batch_indices, gold_label_tensor]
        
        # Compute loss as mean -log(prob[gold_label])
        loss = -torch.mean(torch.log(probs_at_gold + 1e-10))
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), self.grad_clip_norm)
        if not self.freeze_bert:
            torch.nn.utils.clip_grad_norm_(self.bert.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        
        # Compute batch metrics
        batch_correct = (np.argmax(aggregated_probs, axis=1) == gold_label)
        batch_accuracy = float(np.mean(batch_correct))
        avg_prob_correct = float(np.mean(aggregated_probs[np.arange(batch_size), gold_label]))
        
        return {
            "loss": float(loss.item()),
            "batch_accuracy": batch_accuracy,
            "avg_prob_correct": avg_prob_correct,
            "batch_size": batch_size,
        }
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters."""
        trainable = list(self.router.parameters())
        if not self.freeze_bert:
            trainable.extend(list(self.bert.parameters()))
        return trainable
    
    def train_mode(self):
        """Set the method to training mode."""
        self.router.train()
        if not self.freeze_bert:
            self.bert.train()
        self._training = True
        # Clear cache when switching modes
        self._cached_wagers = None
        self._cached_question_embedding = None
    
    def eval_mode(self):
        """Set the method to evaluation mode."""
        self.router.eval()
        self.bert.eval()
        self._training = False
        # Clear cache when switching modes
        self._cached_wagers = None
        self._cached_question_embedding = None
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        state = {
            "router_state_dict": self.router.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "bert_model": self.bert_model_name,
                "embedding_dim": self.embedding_dim,
                "hidden_layers": self.hidden_layers,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "freeze_bert": self.freeze_bert,
                "device": self.device_str,
            },
        }
        
        # Include BERT state if not frozen
        if not self.freeze_bert:
            state["bert_state_dict"] = self.bert.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary from checkpoint."""
        if "router_state_dict" in state_dict:
            self.router.load_state_dict(state_dict["router_state_dict"])
        
        if "bert_state_dict" in state_dict and not self.freeze_bert:
            self.bert.load_state_dict(state_dict["bert_state_dict"])
        
        if "optimizer_state_dict" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                import logging
                log = logging.getLogger("lm_polygraph")
                log.warning(
                    f"Could not load optimizer state dict: {e}. "
                    "Continuing with fresh optimizer state."
                )
