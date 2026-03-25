import os
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer


# Disable background safetensors conversion PR attempts to avoid network-thread failures.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

log = logging.getLogger(__name__)


def _should_retry_with_slow_tokenizer(error: Exception) -> bool:
    """Detect known fast-tokenizer conversion failures for SentencePiece models."""
    message = str(error)
    retry_markers = (
        "Could not extract SentencePiece model",
        "Descriptors cannot be created directly",
        "Error parsing line",
        "tokenizer.model",
    )
    return any(marker in message for marker in retry_markers)


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    return model


def load_tokenizer(
    model_path: str,
    add_bos_token: bool = True,
    use_fast: bool = True,
):
    load_kwargs = {
        "padding_side": "left",
        "add_bos_token": add_bos_token,
        "use_fast": use_fast,
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            **load_kwargs,
        )
    except Exception as error:
        # SentencePiece conversion can fail when protobuf/sentencepiece versions mismatch.
        # Retry with slow tokenizer to keep model loading functional.
        if use_fast and _should_retry_with_slow_tokenizer(error):
            log.warning(
                "Fast tokenizer load failed for %s (%s). Retrying with use_fast=False.",
                model_path,
                error,
            )
            retry_kwargs = dict(load_kwargs)
            retry_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                **retry_kwargs,
            )
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
