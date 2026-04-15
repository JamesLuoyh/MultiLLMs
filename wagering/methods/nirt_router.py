"""
Neural IRT router for multi-LLM wagering.

This follows the NIRT-Router style architecture from the reference repo:
- Query encoder predicts item difficulty vector and discrimination scalar.
- Each model has a learnable latent ability profile projected to a knowledge space.
- A positive-constrained prediction subnet estimates per-model correctness.

Router wagers are softmax over per-model correctness logits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Must be set before tokenizer/protobuf modules are imported.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from transformers import AutoModel, AutoTokenizer

from .base import WageringMethod
from .utils import preprocess_pubmedqa_prompts_for_embedding
from wagering.aggregation.linear_pooling import LinearPooling


class NIRTRouterWagers(WageringMethod):
    """
    NIRT router with query-conditioned item parameters and model ability profiles.

    For each query x and model m, the architecture computes:
      stat_m       = sigmoid(W_model * emb_m)
      k_diff(x)    = sigmoid(W_k * h(x))
      e_diff(x)    = 9 * sigmoid(W_e * h(x))
      r(x)         = softmax(knowledge vector)
      z(x, m)      = e_diff(x) * (stat_m - k_diff(x)) * r(x)
      p_correct    = sigmoid(PosLinear(tanh(PosLinear(z))))

    Then router logits are logit(p_correct), and wagers are softmax over models.
    """

    class _PosLinear(nn.Linear):
        """Linear layer with positive-constrained effective weights."""

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            weight = 2 * F.relu(-self.weight) + self.weight
            return F.linear(input_tensor, weight, self.bias)

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models, config or {})
        cfg = self.config

        self.requires_hidden_states = False

        self.encoder_model_name = str(
            cfg.get("encoder_model_name", cfg.get("bert_model_name", "bert-base-uncased"))
        )
        self.max_seq_length = int(cfg.get("max_seq_length", 512))
        self.learning_rate = float(cfg.get("learning_rate", 5e-5))
        self.temperature = float(cfg.get("temperature", 1.0))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
        self.weight_decay = float(cfg.get("weight_decay", 0.01))
        self.freeze_encoder = bool(cfg.get("freeze_encoder", False))
        self.pubmedqa_strip_context = bool(cfg.get("pubmedqa_strip_context", True))
        self.use_concatenated_prompt_context = bool(
            cfg.get("use_concatenated_prompt_context", True)
        )

        self.knowledge_dim = int(cfg.get("knowledge_dim", 25))
        self.model_embedding_dim = int(cfg.get("model_embedding_dim", 256))
        self.router_hidden_dim = int(cfg.get("router_hidden_dim", 512))
        self.router_dropout = float(cfg.get("router_dropout", 0.1))

        self.bce_loss_weight = float(cfg.get("bce_loss_weight", 1.0))
        self.mixture_ce_weight = float(cfg.get("mixture_ce_weight", 0.25))
        self.param_l2_weight = float(cfg.get("param_l2_weight", 1e-4))

        self.lr_decay_factor = float(cfg.get("lr_decay_factor", 1.0))
        self.lr_decay_steps = int(cfg.get("lr_decay_steps", 100))

        self.device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)

        tokenizer_use_fast_cfg = cfg.get("tokenizer_use_fast", None)
        if tokenizer_use_fast_cfg is None:
            model_name_l = self.encoder_model_name.lower()
            tokenizer_use_fast = not any(
                marker in model_name_l for marker in ("deberta-v2", "deberta-v3", "mdeberta")
            )
        else:
            tokenizer_use_fast = bool(tokenizer_use_fast_cfg)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.encoder_model_name,
                truncation_side="left",
                padding=True,
                use_fast=tokenizer_use_fast,
            )
        except Exception as exc:
            err = str(exc)
            recoverable_errors = (
                "Descriptors cannot be created directly",
                "duplicate file name sentencepiece_model.proto",
                "sentencepiece_model.proto",
                "Error parsing line",
                "spm.model",
                "convert_slow_tokenizer",
                "tiktoken",
            )
            if not any(msg in err for msg in recoverable_errors):
                raise

            # Fallback for protobuf/sentencepiece compatibility issues in some envs.
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.encoder_model_name,
                    truncation_side="left",
                    padding=True,
                    use_fast=False,
                )
            except Exception as slow_exc:
                # Some transformers versions still route through fast-tokenizer
                # conversion paths; bypass AutoTokenizer for DeBERTa-based models.
                model_name_l = self.encoder_model_name.lower()
                if any(marker in model_name_l for marker in ("deberta-v2", "deberta-v3", "mdeberta")):
                    from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer

                    self.tokenizer = DebertaV2Tokenizer.from_pretrained(
                        self.encoder_model_name,
                        truncation_side="left",
                        padding=True,
                    )
                else:
                    raise slow_exc

        self.encoder = AutoModel.from_pretrained(self.encoder_model_name).to(self.device)
        hidden_size = int(self.encoder.config.hidden_size)

        # Model latent embeddings (learned in this codebase, external in upstream experiments).
        self.model_embeddings = nn.Embedding(num_models, self.model_embedding_dim).to(self.device)
        self.model_proj = nn.Linear(self.model_embedding_dim, self.knowledge_dim).to(self.device)

        self.k_difficulty = nn.Linear(hidden_size, self.knowledge_dim).to(self.device)
        self.e_difficulty = nn.Linear(hidden_size, 1).to(self.device)

        self.prednet_full1 = self._PosLinear(self.knowledge_dim, self.router_hidden_dim).to(self.device)
        self.drop_1 = nn.Dropout(p=self.router_dropout).to(self.device)
        self.prednet_full3 = self._PosLinear(self.router_hidden_dim, 1).to(self.device)

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        trainable: List[torch.nn.Parameter] = []
        trainable.extend(list(self.model_embeddings.parameters()))
        trainable.extend(list(self.model_proj.parameters()))
        trainable.extend(list(self.k_difficulty.parameters()))
        trainable.extend(list(self.e_difficulty.parameters()))
        trainable.extend(list(self.prednet_full1.parameters()))
        trainable.extend(list(self.prednet_full3.parameters()))
        if not self.freeze_encoder:
            trainable.extend(list(self.encoder.parameters()))

        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=max(1, self.lr_decay_steps),
            gamma=self.lr_decay_factor,
        )

        self._training = True
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_prob_correct: Optional[torch.Tensor] = None

    def _extract_knowledge_vectors(
        self,
        batch_size: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Optional inputs for closer alignment with upstream implementation.
        knowledge_vectors = kwargs.get("knowledge_vectors", None)
        if knowledge_vectors is None:
            knowledge_vectors = kwargs.get("relevance_vectors", None)

        if knowledge_vectors is None:
            return torch.ones(batch_size, self.knowledge_dim, device=self.device)

        kv = torch.as_tensor(knowledge_vectors, dtype=torch.float32, device=self.device)
        if kv.ndim == 1:
            kv = kv.unsqueeze(0)
        if kv.shape[0] != batch_size:
            raise ValueError(
                f"knowledge_vectors batch mismatch: expected {batch_size}, got {kv.shape[0]}"
            )
        if kv.shape[1] != self.knowledge_dim:
            raise ValueError(
                f"knowledge_vectors dim mismatch: expected {self.knowledge_dim}, got {kv.shape[1]}"
            )
        return kv

    def _encode_questions_batch(self, questions: List[str]) -> torch.Tensor:
        strip_context = self.pubmedqa_strip_context and (not self.use_concatenated_prompt_context)
        processed = preprocess_pubmedqa_prompts_for_embedding(
            questions,
            strip_context=strip_context,
        )
        inputs = self.tokenizer(
            processed,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            padding=True,
        ).to(self.device)

        grad_enc = self._training and not self.freeze_encoder
        with torch.set_grad_enabled(grad_enc):
            outputs = self.encoder(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return pooled.to(dtype=self.k_difficulty.weight.dtype)

    def _compute_prob_correct(self, pooled: torch.Tensor, knowledge_vectors: torch.Tensor) -> torch.Tensor:
        batch_size = pooled.shape[0]

        model_ids = torch.arange(self.num_models, device=self.device)
        model_emb = self.model_embeddings(model_ids)
        stat_emb = torch.sigmoid(self.model_proj(model_emb))  # [M, K]

        k_diff = torch.sigmoid(self.k_difficulty(pooled))  # [B, K]
        e_diff = 9.0 * torch.sigmoid(self.e_difficulty(pooled))  # [B, 1]
        rel = torch.softmax(knowledge_vectors, dim=1)  # [B, K]

        # [B, M, K]
        input_x = e_diff.unsqueeze(1) * (stat_emb.unsqueeze(0) - k_diff.unsqueeze(1)) * rel.unsqueeze(1)
        x = torch.tanh(self.prednet_full1(input_x))
        x = self.drop_1(x)
        p_correct = torch.sigmoid(self.prednet_full3(x)).squeeze(-1)  # [B, M]

        if p_correct.shape != (batch_size, self.num_models):
            raise RuntimeError(
                f"Unexpected p_correct shape {tuple(p_correct.shape)} for batch={batch_size}, models={self.num_models}"
            )

        return torch.clamp(p_correct, min=1e-6, max=1.0 - 1e-6)

    def compute_wagers(
        self,
        questions: Optional[List[str]] = None,
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        hidden_states_list: Optional[List[np.ndarray]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if questions is None:
            questions = kwargs.get("questions")
        if questions is None:
            raise ValueError(
                "NIRTRouterWagers.compute_wagers() requires `questions` (batch of prompt strings)."
            )

        pooled = self._encode_questions_batch(questions)
        self.model_embeddings.train() if self._training else self.model_embeddings.eval()
        self.model_proj.train() if self._training else self.model_proj.eval()
        self.k_difficulty.train() if self._training else self.k_difficulty.eval()
        self.e_difficulty.train() if self._training else self.e_difficulty.eval()
        self.prednet_full1.train() if self._training else self.prednet_full1.eval()
        self.prednet_full3.train() if self._training else self.prednet_full3.eval()
        self.drop_1.train() if self._training else self.drop_1.eval()

        knowledge_vectors = self._extract_knowledge_vectors(batch_size=len(questions), **kwargs)

        with torch.set_grad_enabled(self._training):
            p_correct = self._compute_prob_correct(pooled, knowledge_vectors)
            router_logits = torch.logit(p_correct)
            wagers = torch.softmax(router_logits / self.temperature, dim=1)

        if self._training:
            self._cached_wagers = wagers
            self._cached_prob_correct = p_correct

        return {"wagers": wagers.detach().cpu().numpy()}

    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        hidden_states: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        batch_size = model_logits.shape[0]
        if self._cached_wagers is None or self._cached_prob_correct is None:
            raise ValueError(
                "NIRTRouterWagers.update() requires cached wagers from compute_wagers() "
                "called in training mode beforehand."
            )

        wagers = self._cached_wagers
        p_correct = self._cached_prob_correct
        self._cached_wagers = None
        self._cached_prob_correct = None

        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)

        model_pred = torch.argmax(model_logits_tensor, dim=-1)
        correctness = (model_pred == gold_label_tensor.view(-1, 1)).to(torch.float32)

        bce_loss = F.binary_cross_entropy(p_correct, correctness)

        batch_aggregated_probs = LinearPooling.aggregate_torch(model_logits_tensor, wagers)
        batch_indices = torch.arange(batch_size, device=self.device)
        probs_at_gold = batch_aggregated_probs[batch_indices, gold_label_tensor]
        ce_loss = -torch.mean(torch.log(probs_at_gold + 1e-10))

        reg = torch.tensor(0.0, device=self.device)
        reg = reg + torch.mean(self.model_embeddings.weight ** 2)
        reg = reg + torch.mean(self.model_proj.weight ** 2)
        reg = reg + torch.mean(self.k_difficulty.weight ** 2)
        reg = reg + torch.mean(self.e_difficulty.weight ** 2)
        reg = reg + torch.mean(self.prednet_full1.weight ** 2)
        reg = reg + torch.mean(self.prednet_full3.weight ** 2)

        loss = (
            self.bce_loss_weight * bce_loss
            + self.mixture_ce_weight * ce_loss
            + self.param_l2_weight * reg
        )

        self.optimizer.zero_grad()
        loss.backward()

        trainable_params: List[torch.nn.Parameter] = []
        trainable_params.extend(list(self.model_embeddings.parameters()))
        trainable_params.extend(list(self.model_proj.parameters()))
        trainable_params.extend(list(self.k_difficulty.parameters()))
        trainable_params.extend(list(self.e_difficulty.parameters()))
        trainable_params.extend(list(self.prednet_full1.parameters()))
        trainable_params.extend(list(self.prednet_full3.parameters()))
        if not self.freeze_encoder:
            trainable_params.extend(list(self.encoder.parameters()))

        torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        batch_aggregated_probs_np = batch_aggregated_probs.detach().cpu().numpy()
        batch_accuracy = float(np.mean(np.argmax(batch_aggregated_probs_np, axis=1) == gold_label))
        avg_prob_correct = float(
            np.mean(batch_aggregated_probs_np[np.arange(batch_size), gold_label])
        )

        return {
            "loss": float(loss.item()),
            "bce_loss": float(bce_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "param_l2": float(reg.item()),
            "batch_accuracy": batch_accuracy,
            "avg_prob_correct": avg_prob_correct,
            "batch_size": batch_size,
        }

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        params.extend(list(self.model_embeddings.parameters()))
        params.extend(list(self.model_proj.parameters()))
        params.extend(list(self.k_difficulty.parameters()))
        params.extend(list(self.e_difficulty.parameters()))
        params.extend(list(self.prednet_full1.parameters()))
        params.extend(list(self.prednet_full3.parameters()))
        if not self.freeze_encoder:
            params.extend(list(self.encoder.parameters()))
        return params

    def train_mode(self) -> None:
        self.model_embeddings.train()
        self.model_proj.train()
        self.k_difficulty.train()
        self.e_difficulty.train()
        self.prednet_full1.train()
        self.prednet_full3.train()
        self.drop_1.train()
        if not self.freeze_encoder:
            self.encoder.train()
        self._training = True
        self._cached_wagers = None
        self._cached_prob_correct = None

    def eval_mode(self) -> None:
        self.model_embeddings.eval()
        self.model_proj.eval()
        self.k_difficulty.eval()
        self.e_difficulty.eval()
        self.prednet_full1.eval()
        self.prednet_full3.eval()
        self.drop_1.eval()
        self.encoder.eval()
        self._training = False
        self._cached_wagers = None
        self._cached_prob_correct = None

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "encoder_state_dict": self.encoder.state_dict(),
            "model_embeddings_state_dict": self.model_embeddings.state_dict(),
            "model_proj_state_dict": self.model_proj.state_dict(),
            "k_difficulty_state_dict": self.k_difficulty.state_dict(),
            "e_difficulty_state_dict": self.e_difficulty.state_dict(),
            "prednet_full1_state_dict": self.prednet_full1.state_dict(),
            "prednet_full3_state_dict": self.prednet_full3.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "encoder_model_name": self.encoder_model_name,
                "max_seq_length": self.max_seq_length,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "weight_decay": self.weight_decay,
                "freeze_encoder": self.freeze_encoder,
                "pubmedqa_strip_context": self.pubmedqa_strip_context,
                "use_concatenated_prompt_context": self.use_concatenated_prompt_context,
                "knowledge_dim": self.knowledge_dim,
                "model_embedding_dim": self.model_embedding_dim,
                "router_hidden_dim": self.router_hidden_dim,
                "router_dropout": self.router_dropout,
                "bce_loss_weight": self.bce_loss_weight,
                "mixture_ce_weight": self.mixture_ce_weight,
                "param_l2_weight": self.param_l2_weight,
                "lr_decay_factor": self.lr_decay_factor,
                "lr_decay_steps": self.lr_decay_steps,
                "device": self.device_str,
            },
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "encoder_state_dict" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder_state_dict"])
        if "model_embeddings_state_dict" in state_dict:
            self.model_embeddings.load_state_dict(state_dict["model_embeddings_state_dict"])
        if "model_proj_state_dict" in state_dict:
            self.model_proj.load_state_dict(state_dict["model_proj_state_dict"])
        if "k_difficulty_state_dict" in state_dict:
            self.k_difficulty.load_state_dict(state_dict["k_difficulty_state_dict"])
        if "e_difficulty_state_dict" in state_dict:
            self.e_difficulty.load_state_dict(state_dict["e_difficulty_state_dict"])
        if "prednet_full1_state_dict" in state_dict:
            self.prednet_full1.load_state_dict(state_dict["prednet_full1_state_dict"])
        if "prednet_full3_state_dict" in state_dict:
            self.prednet_full3.load_state_dict(state_dict["prednet_full3_state_dict"])
        if "optimizer_state_dict" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                import logging

                logging.getLogger("wagering").warning(
                    "Could not load optimizer state dict: %s. Using fresh optimizer.", e
                )
        if "scheduler_state_dict" in state_dict:
            try:
                self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            except (ValueError, KeyError) as e:
                import logging

                logging.getLogger("wagering").warning(
                    "Could not load scheduler state dict: %s. Using fresh scheduler.", e
                )
