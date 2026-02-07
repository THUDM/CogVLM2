"""
Compatibility patches for CogVLM2 with newer versions of transformers.

This module provides patches for breaking changes in transformers >= 4.45
where `_extract_past_from_model_output` was removed from GenerationMixin.

Usage:
    from transformers_compat import patch_model_for_transformers_compat
    model = AutoModelForCausalLM.from_pretrained(...)
    patch_model_for_transformers_compat(model)
"""
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.utils import ModelOutput


def patch_model_for_transformers_compat(model):
    """
    Patch CogVLM2 model for compatibility with transformers >= 4.45.

    In transformers 4.45+, the `_extract_past_from_model_output` method was removed
    from GenerationMixin. This patch replaces the model's `_update_model_kwargs_for_generation`
    method with a compatible version that works with both old and new transformers versions.

    Args:
        model: The CogVLM2 model instance to patch.

    Returns:
        The patched model (same instance, modified in place).
    """
    # Check if patch is needed
    if hasattr(model, "_extract_past_from_model_output"):
        # Old transformers version, no patch needed
        return model

    # Store the original method
    original_update_kwargs = model._update_model_kwargs_for_generation

    def patched_update_model_kwargs_for_generation(
        outputs: "ModelOutput",
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Patched version of _update_model_kwargs_for_generation that works with
        transformers >= 4.45 where _extract_past_from_model_output was removed.
        """
        import torch

        # Get past_key_values directly from outputs
        model_kwargs["past_key_values"] = getattr(outputs, "past_key_values", None)

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            # LANGUAGE_TOKEN_TYPE = 0 (from modeling_cogvlm.py)
            LANGUAGE_TOKEN_TYPE = 0
            new_token_type_ids = torch.ones(
                size=(token_type_ids.shape[0], 1),
                dtype=token_type_ids.dtype,
                device=token_type_ids.device
            ) * LANGUAGE_TOKEN_TYPE
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, new_token_type_ids], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

    # Apply the patch
    import types
    model._update_model_kwargs_for_generation = types.MethodType(
        patched_update_model_kwargs_for_generation, model
    )

    return model
