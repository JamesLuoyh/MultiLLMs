"""
Pre-inference MSE-BR wagers implementation.

This method follows the MSE BR v2 wager objective, but computes wagers from a
question embedding router (MoE-style) instead of model hidden states.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from .base import WageringMethod
from .utils import preprocess_pubmedqa_prompts_for_embedding


class PreInferenceMSEBrWagersV2(WageringMethod):
    """
    MSE-BR wagers with pre-inference routing from question embeddings.

    Architecture:
    - Encode each question with a transformer encoder (default: BERT)
    - Each model has its own router MLP that maps embedding -> scalar logit
    - Apply sigmoid + normalization to obtain non-negative wagers summing to 1

        PubMedQA handling:
        - By default, context blocks are removed for embedding generation while
            preserving the rest of the prompt (question, long answer, and instruction).
    """

    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(num_models, config)

        if config is None:
            config = {}

        # Encoder/router config
        self.bert_model_name = str(config.get("bert_model", "bert-base-uncased"))
        self.embedding_dim = int(config.get("embedding_dim", 768))
        self.hidden_layers = list(config.get("hidden_layers", [512, 256]))
        self.learning_rate = float(config.get("learning_rate", 1e-5))
        self.temperature = float(config.get("temperature", 2.0))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        self.freeze_bert = bool(config.get("freeze_bert", True))
        self.lr_decay_factor = float(config.get("lr_decay_factor", 1.0))
        self.lr_decay_steps = int(config.get("lr_decay_steps", 1))
        # Keep backward compatibility: old key pubmedqa_question_only now maps to
        # stripping the context section for embeddings.
        self.pubmedqa_strip_context = bool(
            config.get(
                "pubmedqa_strip_context",
                config.get("pubmedqa_question_only", True),
            )
        )

        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert = AutoModel.from_pretrained(self.bert_model_name).to(self.device)

        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()

        self.routers = nn.ModuleList()
        self.optimizers = []
        self.schedulers = []

        for i in range(num_models):
            router = self._build_router().to(self.device)
            self.routers.append(router)

        for i in range(num_models):
            params = list(self.routers[i].parameters())
            if not self.freeze_bert:
                params += list(self.bert.parameters())

            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_decay_steps,
                gamma=self.lr_decay_factor,
            )
            self.optimizers.append((i, optimizer))
            self.schedulers.append((i, scheduler))

        self._training = True
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_question_embeddings: Optional[torch.Tensor] = None

    def _build_router(self) -> nn.Module:
        layers = []
        prev_dim = self.embedding_dim

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def _prepare_questions_for_encoding(self, questions: List[str]) -> List[str]:
        return preprocess_pubmedqa_prompts_for_embedding(
            questions,
            strip_context=self.pubmedqa_strip_context,
        )

    def _encode_questions_batch(self, questions: List[str]) -> torch.Tensor:
        processed_questions = self._prepare_questions_for_encoding(questions)
        inputs = self.tokenizer(
            processed_questions,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)

        with torch.set_grad_enabled(self._training and not self.freeze_bert):
            outputs = self.bert(**inputs)

        return outputs.last_hidden_state[:, 0, :]

    def extract_wagers_brs_and_nash_gap(
        self,
        sigmoid_wagers: torch.Tensor,
        model_logits_tensor: torch.Tensor,
        gold_label_tensor: torch.Tensor,
    ):
        probs = F.softmax(model_logits_tensor, dim=-1)

        batch_size, num_models, num_options = probs.shape
        gt_onehot = F.one_hot(gold_label_tensor, num_classes=num_options).float()
        gt_onehot_expanded = gt_onehot.unsqueeze(1).expand(batch_size, num_models, num_options)

        squared_errors = (probs - gt_onehot_expanded) ** 2
        brier_scores = squared_errors.sum(dim=-1)

        scores = 0.5 * (2 - brier_scores)

        wagers_except_i = torch.clamp(
            sigmoid_wagers.sum(dim=1, keepdim=True) - sigmoid_wagers,
            min=1e-16,
        )
        average_scores = (
            (scores * sigmoid_wagers).sum(dim=1, keepdim=True).expand_as(scores * sigmoid_wagers)
            - (scores * sigmoid_wagers)
        ) / wagers_except_i

        brs = scores - average_scores
        brs = torch.clamp(brs, min=1e-16, max=1.0 - 1e-16)
        total_payout = sigmoid_wagers * (scores - average_scores - 0.5 * sigmoid_wagers)
        nash_gap = brs * (scores - average_scores - 0.5 * brs) - total_payout
        score_diff = scores - average_scores

        return brs, nash_gap, score_diff, total_payout

    def compute_wagers(
        self,
        questions: List[str],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size = len(questions)

        self.router_train_mode(self._training)

        with torch.set_grad_enabled(self._training):
            question_embeddings = self._encode_questions_batch(questions)

            raw_wagers_list = []
            for i in range(self.num_models):
                raw_wagers_list.append(self.routers[i](question_embeddings))

            raw_wagers_tensor = torch.cat(raw_wagers_list, dim=1)
            sigmoid_wagers = torch.sigmoid(raw_wagers_tensor / self.temperature)
            sigmoid_wagers = torch.clamp(sigmoid_wagers, min=1e-16, max=1.0 - 1e-16)

            sigmoid_sum = torch.sum(sigmoid_wagers, dim=1, keepdim=True)
            if torch.any(sigmoid_sum < 1e-16):
                raise RuntimeError("Near-zero sigmoid sum detected during compute_wagers().")

            wagers = sigmoid_wagers / sigmoid_sum

        if self._training:
            self._cached_wagers = sigmoid_wagers
            self._cached_question_embeddings = question_embeddings

        result: Dict[str, Any] = {
            "wagers": wagers.detach().cpu().numpy(),
            "sigmoid_wagers": sigmoid_wagers.detach().cpu().numpy(),
        }

        if model_logits is not None and gold_label is not None:
            is_batch = model_logits.ndim == 3
            if not is_batch:
                model_logits = model_logits[np.newaxis, :, :]
                gold_label = np.array([gold_label])

            model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
            gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)
            _, nash_gap, score_diff, total_payout = self.extract_wagers_brs_and_nash_gap(
                sigmoid_wagers,
                model_logits_tensor,
                gold_label_tensor,
            )
            result["nash_gap"] = nash_gap.detach().cpu().numpy()
            result["score_diff"] = score_diff.detach().cpu().numpy()
            result["total_payout"] = total_payout.detach().cpu().numpy()

        wagers_np = result["wagers"]
        if np.any(np.isnan(wagers_np)) or np.any(np.isinf(wagers_np)):
            raise ValueError("Invalid wagers detected (NaN or inf).")

        return result

    def router_train_mode(self, is_training: bool):
        if is_training:
            for router in self.routers:
                router.train()
            if not self.freeze_bert:
                self.bert.train()
        else:
            for router in self.routers:
                router.eval()
            self.bert.eval()

    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        hidden_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self._training:
            return {}

        if self._cached_wagers is None or self._cached_question_embeddings is None:
            raise ValueError(
                "PreInferenceMSEBrWagersV2.update() requires cached values from compute_wagers(). "
                "Ensure compute_wagers() is called before update()."
            )

        sigmoid_wagers = self._cached_wagers
        question_embeddings = self._cached_question_embeddings

        self._cached_wagers = None
        self._cached_question_embeddings = None

        is_batch = model_logits.ndim == 3
        if not is_batch:
            model_logits = model_logits[np.newaxis, :, :]
            gold_label = np.array([gold_label])

        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32, device=self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long, device=self.device)

        with torch.enable_grad():
            if question_embeddings.shape[0] != model_logits_tensor.shape[0]:
                if question_embeddings.shape[0] == 1:
                    question_embeddings = question_embeddings.expand(model_logits_tensor.shape[0], -1)
                else:
                    raise ValueError(
                        f"Question embedding batch ({question_embeddings.shape[0]}) does not match logits batch "
                        f"({model_logits_tensor.shape[0]})."
                    )

            raw_wagers_list = []
            for i in range(self.num_models):
                raw_wagers_list.append(self.routers[i](question_embeddings))

            raw_wagers_tensor = torch.cat(raw_wagers_list, dim=1)
            sigmoid_wagers = torch.sigmoid(raw_wagers_tensor / self.temperature)
            sigmoid_wagers = torch.clamp(sigmoid_wagers, min=1e-16, max=1.0 - 1e-16)

            brs, _, _, _ = self.extract_wagers_brs_and_nash_gap(
                sigmoid_wagers,
                model_logits_tensor,
                gold_label_tensor,
            )
            mseloss = F.mse_loss(sigmoid_wagers, brs, reduction="none")

            all_losses = []
            for i in range(self.num_models):
                all_losses.append(mseloss[:, i].mean())

            total_loss = 0.0
            for i in range(self.num_models):
                optimizer_i = None
                scheduler_i = None
                for model_idx, opt in self.optimizers:
                    if model_idx == i:
                        optimizer_i = opt
                        break
                for model_idx, sch in self.schedulers:
                    if model_idx == i:
                        scheduler_i = sch
                        break

                if optimizer_i is None:
                    raise RuntimeError(f"No optimizer found for model {i}")

                optimizer_i.zero_grad()

                params_i = list(self.routers[i].parameters())
                if not self.freeze_bert:
                    params_i += list(self.bert.parameters())

                retain = i < self.num_models - 1
                grads = torch.autograd.grad(
                    all_losses[i],
                    params_i,
                    retain_graph=retain,
                    allow_unused=True,
                )

                for param, grad in zip(params_i, grads):
                    if grad is not None:
                        param.grad = grad

                torch.nn.utils.clip_grad_norm_(self.routers[i].parameters(), self.grad_clip_norm)
                if not self.freeze_bert:
                    torch.nn.utils.clip_grad_norm_(self.bert.parameters(), self.grad_clip_norm)

                optimizer_i.step()
                if scheduler_i is not None:
                    scheduler_i.step()

                total_loss += float(all_losses[i].detach().cpu().numpy())

        return {"loss": total_loss / self.num_models}

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        params = list(self.routers.parameters())
        if not self.freeze_bert:
            params.extend(list(self.bert.parameters()))
        return params

    def train_mode(self):
        self._training = True
        self.router_train_mode(True)
        self._cached_wagers = None
        self._cached_question_embeddings = None

    def eval_mode(self):
        self._training = False
        self.router_train_mode(False)
        self._cached_wagers = None
        self._cached_question_embeddings = None

    def state_dict(self) -> Dict[str, Any]:
        routers_state_dict = {f"router_{i}": router.state_dict() for i, router in enumerate(self.routers)}
        optimizers_state_dict = {f"optimizer_{model_idx}": opt.state_dict() for model_idx, opt in self.optimizers}

        state = {
            "routers_state_dict": routers_state_dict,
            "optimizers_state_dict": optimizers_state_dict,
            "config": {
                "bert_model": self.bert_model_name,
                "embedding_dim": self.embedding_dim,
                "hidden_layers": self.hidden_layers,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "freeze_bert": self.freeze_bert,
                "lr_decay_factor": self.lr_decay_factor,
                "lr_decay_steps": self.lr_decay_steps,
                "pubmedqa_strip_context": self.pubmedqa_strip_context,
                "device": self.device_str,
            },
        }

        if not self.freeze_bert:
            state["bert_state_dict"] = self.bert.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "routers_state_dict" in state_dict:
            routers_state_dict = state_dict["routers_state_dict"]
            for i, router in enumerate(self.routers):
                key = f"router_{i}"
                if key in routers_state_dict:
                    router.load_state_dict(routers_state_dict[key])

        if "bert_state_dict" in state_dict and not self.freeze_bert:
            self.bert.load_state_dict(state_dict["bert_state_dict"])

        if "optimizers_state_dict" in state_dict:
            optimizers_state_dict = state_dict["optimizers_state_dict"]
            for model_idx, optimizer in self.optimizers:
                key = f"optimizer_{model_idx}"
                if key in optimizers_state_dict:
                    try:
                        optimizer.load_state_dict(optimizers_state_dict[key])
                    except (ValueError, KeyError) as e:
                        import logging

                        log = logging.getLogger("wagering")
                        log.warning(
                            f"Could not load optimizer {model_idx} state dict (parameter mismatch): {e}. "
                            "Continuing with fresh optimizer state."
                        )
