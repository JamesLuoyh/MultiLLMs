"""
RouterDC-style query router for multi-LLM wagering (Chen et al., NeurIPS 2024; arXiv:2409.19886).

Query encoder + trainable expert embeddings; routing scores are similarity (cosine or dot)
between the query embedding and each expert embedding, then softmax with temperature.

Training uses sample–LLM contrastive loss: positives/negatives are derived from each
expert's probability on the gold label (from cached `model_logits`), matching the
reference implementation's use of per-expert scores without requiring task/cluster IDs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
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


class RouterDCWagers(WageringMethod):
    """
    Encoder + expert embeddings + similarity routing (RouterDC-style).

    Unlike `CentralizedWagers`, routing uses only the task prompt text, not LLM hidden states.
    Unlike `RouteLLMBertWagers`, experts are represented by trainable embedding vectors and
    the training objective is sample–LLM contrastive (multi-positive) rather than a linear
    head + mixture cross-entropy (optional CE can be added later).
    """

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models, config or {})
        cfg = self.config

        # RouterDC routes from prompt text embeddings only.
        self.requires_hidden_states = False

        self.encoder_model_name = str(
            cfg.get("encoder_model_name", cfg.get("bert_model_name", "microsoft/mdeberta-v3-base"))
        )
        self.max_seq_length = int(cfg.get("max_seq_length", 512))
        self.learning_rate = float(cfg.get("learning_rate", 5e-5))
        self.temperature = float(cfg.get("temperature", 1.0))
        self.wager_floor = float(cfg.get("wager_floor", 1e-16))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
        self.weight_decay = float(cfg.get("weight_decay", 0.01))
        self.freeze_encoder = bool(cfg.get("freeze_encoder", False))
        self.pubmedqa_strip_context = bool(cfg.get("pubmedqa_strip_context", True))
        self.use_concatenated_prompt_context = bool(
            cfg.get("use_concatenated_prompt_context", True)
        )
        self.similarity_function = str(cfg.get("similarity_function", "cos")).lower()
        if self.similarity_function not in ("cos", "dot"):
            raise ValueError("similarity_function must be 'cos' or 'dot'")

        self.top_k = int(cfg.get("top_k", 3))
        self.last_k = int(cfg.get("last_k", 3))
        self.min_pos_p = float(cfg.get("min_pos_p", 0.01))
        self.neg_mask_threshold = float(cfg.get("neg_mask_threshold", 0.5))

        self.lr_decay_factor = float(cfg.get("lr_decay_factor", 1.0))
        self.lr_decay_steps = int(cfg.get("lr_decay_steps", 100))

        self.device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)

        tokenizer_use_fast_cfg = cfg.get("tokenizer_use_fast", None)
        if tokenizer_use_fast_cfg is None:
            model_name_l = self.encoder_model_name.lower()
            # DeBERTa-v2/v3 tokenizers are SentencePiece-backed and can hit
            # protobuf compatibility issues during fast tokenizer conversion.
            tokenizer_use_fast = not any(
                marker in model_name_l for marker in ("deberta-v2", "deberta-v3", "mdeberta")
            )
        else:
            tokenizer_use_fast = bool(tokenizer_use_fast_cfg)

        # Some SentencePiece-based models can fail to initialize a fast tokenizer
        # when protobuf/sentencepiece versions are incompatible in the environment.
        # Fall back to the slow tokenizer so training/eval can proceed.
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

        std_dev = float(cfg.get("expert_embedding_std", 0.78))
        self.expert_embeddings = torch.nn.Embedding(num_models, hidden_size).to(self.device)
        with torch.no_grad():
            torch.nn.init.normal_(self.expert_embeddings.weight, mean=0.0, std=std_dev)

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        trainable: List[torch.nn.Parameter] = list(self.expert_embeddings.parameters())
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
        self._cached_router_logits: Optional[torch.Tensor] = None

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
        # First token representation (matches RouterDC reference code)
        hidden = outputs.last_hidden_state[:, 0, :]
        hidden = hidden.to(dtype=self.expert_embeddings.weight.dtype)
        return hidden

    def _compute_similarity(self, query_emb: torch.Tensor) -> torch.Tensor:
        """query_emb: [B, H], returns logits [B, M] before temperature scaling."""
        expert_w = self.expert_embeddings.weight  # [M, H]
        query_emb = query_emb.to(dtype=expert_w.dtype)
        if self.similarity_function == "cos":
            q = F.normalize(query_emb, dim=-1)
            e = F.normalize(expert_w, dim=-1)
            return q @ e.T
        return query_emb @ expert_w.T

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
                "RouterDCWagers.compute_wagers() requires `questions` (batch of prompt strings)."
            )

        query_emb = self._encode_questions_batch(questions)
        self.expert_embeddings.train() if self._training else self.expert_embeddings.eval()

        with torch.set_grad_enabled(self._training):
            logits = self._compute_similarity(query_emb)
            logits = logits / self.temperature
            wagers = torch.softmax(logits, dim=1)
            wagers = torch.clamp(wagers, min=self.wager_floor, max=1.0 - self.wager_floor)
            wagers = wagers / wagers.sum(dim=1, keepdim=True)

        if self._training:
            self._cached_wagers = wagers
            self._cached_router_logits = logits

        return {"wagers": wagers.detach().cpu().numpy()}

    def _sample_llm_contrastive_loss(
        self,
        router_logits: torch.Tensor,
        p_gold: torch.Tensor,
    ) -> torch.Tensor:
        """
        router_logits: [B, M] (already scaled by temperature)
        p_gold: [B, M] probability each expert assigns to gold label
        """
        B, M = router_logits.shape
        device = router_logits.device
        k_pos = min(self.top_k, M)
        k_neg = min(self.last_k, M)

        _, top_idx = torch.topk(p_gold, k=k_pos, dim=1)
        _, bot_idx = torch.topk(p_gold, k=k_neg, dim=1, largest=False)

        total = torch.zeros((), device=device)
        n_terms = 0

        for i in range(k_pos):
            pos_idx = top_idx[:, i]
            pos_logit = torch.gather(router_logits, 1, pos_idx.unsqueeze(1)).squeeze(1)
            pos_p = torch.gather(p_gold, 1, pos_idx.unsqueeze(1)).squeeze(1)
            mask = pos_p > self.min_pos_p

            neg_logits = torch.gather(router_logits, 1, bot_idx)
            neg_p = torch.gather(p_gold, 1, bot_idx)
            neg_logits = torch.where(
                neg_p > self.neg_mask_threshold,
                torch.full_like(neg_logits, float("-inf")),
                neg_logits,
            )
            neg_logits = torch.where(
                bot_idx == pos_idx.unsqueeze(1),
                torch.full_like(neg_logits, float("-inf")),
                neg_logits,
            )

            stacked = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
            log_probs = F.log_softmax(stacked, dim=1)
            term = -log_probs[:, 0]
            denom = mask.float().sum().clamp_min(1.0)
            total = total + (term * mask.float()).sum() / denom
            n_terms += 1

        return total / max(n_terms, 1)

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
        if self._cached_wagers is None or self._cached_router_logits is None:
            raise ValueError(
                "RouterDCWagers.update() requires cached wagers from compute_wagers() "
                "called in training mode beforehand."
            )

        wagers = self._cached_wagers
        router_logits = self._cached_router_logits
        self._cached_wagers = None
        self._cached_router_logits = None

        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)

        probs = F.softmax(model_logits_tensor, dim=-1)
        idx = gold_label_tensor.view(-1, 1, 1).expand(-1, self.num_models, 1)
        p_gold = torch.gather(probs, dim=2, index=idx).squeeze(2)

        loss = self._sample_llm_contrastive_loss(router_logits, p_gold)

        self.optimizer.zero_grad()
        loss.backward()

        trainable_params = list(self.expert_embeddings.parameters())
        if not self.freeze_encoder:
            trainable_params.extend(list(self.encoder.parameters()))
        torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        batch_aggregated_probs = LinearPooling.aggregate_torch(model_logits_tensor, wagers)
        batch_indices = torch.arange(batch_size, device=self.device)
        batch_aggregated_probs_np = batch_aggregated_probs.detach().cpu().numpy()
        batch_accuracy = float(np.mean(np.argmax(batch_aggregated_probs_np, axis=1) == gold_label))
        avg_prob_correct = float(
            np.mean(batch_aggregated_probs_np[np.arange(batch_size), gold_label])
        )

        return {
            "loss": float(loss.item()),
            "batch_accuracy": batch_accuracy,
            "avg_prob_correct": avg_prob_correct,
            "batch_size": batch_size,
        }

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        params = list(self.expert_embeddings.parameters())
        if not self.freeze_encoder:
            params.extend(list(self.encoder.parameters()))
        return params

    def train_mode(self) -> None:
        self.expert_embeddings.train()
        if not self.freeze_encoder:
            self.encoder.train()
        self._training = True
        self._cached_wagers = None
        self._cached_router_logits = None

    def eval_mode(self) -> None:
        self.expert_embeddings.eval()
        self.encoder.eval()
        self._training = False
        self._cached_wagers = None
        self._cached_router_logits = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder_state_dict": self.encoder.state_dict(),
            "expert_embeddings_state_dict": self.expert_embeddings.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "encoder_model_name": self.encoder_model_name,
                "max_seq_length": self.max_seq_length,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "wager_floor": self.wager_floor,
                "grad_clip_norm": self.grad_clip_norm,
                "weight_decay": self.weight_decay,
                "freeze_encoder": self.freeze_encoder,
                "pubmedqa_strip_context": self.pubmedqa_strip_context,
                "use_concatenated_prompt_context": self.use_concatenated_prompt_context,
                "similarity_function": self.similarity_function,
                "top_k": self.top_k,
                "last_k": self.last_k,
                "min_pos_p": self.min_pos_p,
                "neg_mask_threshold": self.neg_mask_threshold,
                "lr_decay_factor": self.lr_decay_factor,
                "lr_decay_steps": self.lr_decay_steps,
                "device": self.device_str,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "encoder_state_dict" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder_state_dict"])
        if "expert_embeddings_state_dict" in state_dict:
            self.expert_embeddings.load_state_dict(state_dict["expert_embeddings_state_dict"])
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
