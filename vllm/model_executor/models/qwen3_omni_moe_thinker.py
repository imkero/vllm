# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3-Omni-Moe model (thinker part)."""

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioInputs,
    Qwen2AudioProcessingInfo,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalKwargsItems,
                                    MultiModalKwargsOptionalItems,
                                    NestedTensors)
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import decode_tokens

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .qwen2_5_omni_thinker import Qwen2_5OmniConditionalGenerationMixin
from .qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder as Qwen3OmniMoeThinkerDummyInputsBuilder,
)
from .qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerMultiModalProcessor,
)
from .qwen2_5_vl import (
    Qwen2_5_VLProcessingInfo,
)
from .qwen3_moe import (
    Qwen3MoeLLMForCausalLM,
    Qwen3MoeLLMModel,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
)
from .vision import get_vit_attn_backend

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)

class Qwen3OmniMoeThinkerProcessingInfo(
    Qwen2AudioProcessingInfo,
    Qwen2_5_VLProcessingInfo
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3OmniMoeConfig).thinker_config

    def get_hf_processor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, List[float]]] = None,
        **kwargs: object,
    ) -> Qwen3OmniMoeProcessor:
        if fps is not None:
            kwargs["fps"] = fps
        processor = self.ctx.get_hf_processor(
            Qwen3OmniMoeProcessor,
            image_processor=self.get_image_processor(
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                size=size,
            ),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None, "image": None, "video": None}


class Qwen3OmniMoeThinkerMultiModalProcessor(
    Qwen2_5OmniThinkerMultiModalProcessor,
):
    
    def _get_feat_extract_output_lengths(
        self, 
        input_lengths: torch.Tensor
    )-> torch.Tensor:
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return feat_lengths, output_lengths

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """Handle prompt updates while supporting ``use_audio_in_video``."""

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = False
        if "video" in mm_kwargs:
            video_kwargs = [item for item in mm_kwargs["video"] if item is not None]
            if video_kwargs:
                use_audio_in_video = all(
                    "use_audio_in_video" in item
                    and bool(item["use_audio_in_video"].data)
                    for item in video_kwargs)

        placeholder_counts = dict(mm_item_counts)
        if use_audio_in_video and "video" in placeholder_counts:
            assert "audio" in placeholder_counts
            placeholder_counts["audio"] -= placeholder_counts["video"]

        if is_update_applied:
            prompt_ids = self._get_raw_input_ids(prompt_ids, use_audio_in_video)

        prompt_ids, prompt, mm_placeholders = self._apply_prompt_updates(
            prompt_ids,
            mm_prompt_updates,
        )

        self._validate_mm_placeholders(
            mm_placeholders,
            placeholder_counts,
            use_audio_in_video=use_audio_in_video,
        )

        tokenizer = self.info.get_tokenizer()
        prompt = decode_tokens(tokenizer, prompt_ids)

        return prompt_ids, prompt, mm_placeholders

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = self._get_feat_extract_output_lengths(
                audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = self._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1))
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0
        audio_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            nonlocal audio_item_idx
            item_idx += audio_in_video_item_idx

            audio_item_idx += 1

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model")

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_data[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        use_audio_in_video = hf_processor_mm_kwargs.get(
            "use_audio_in_video", False)
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx
            audio_num_features = audio_output_lengths[audio_item_idx +
                                                      item_idx]
            video_grid_thw = out_mm_data["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = hf_processor_mm_kwargs.get(
                "second_per_grid_ts", None)
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[item_idx]
            else:
                video_second_per_grid_t = 1.0

            return MRotaryEmbedding.omni3_get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )

        video_replacement_fn = (
            get_replacement_qwen2_use_audio_in_video if use_audio_in_video else
            partial(get_replacement_qwen2_vision, modality="video"))

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision,
                                    modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
        *,
        use_audio_in_video: bool = False,
    ) -> None:
        if use_audio_in_video:
            mm_item_counts = dict(mm_item_counts)
            if "video" in mm_item_counts:
                assert "audio" in mm_item_counts
                mm_item_counts["audio"] -= mm_item_counts["video"]

        super()._validate_mm_placeholders(mm_placeholders, mm_item_counts)

    def _get_raw_input_ids(
        self,
        token_ids: list[int],
        use_audio_in_video: bool = False,
    ) -> list[int]:
        
        tokenizer = self.info.get_tokenizer()
        vision_bos_token = tokenizer.encode(tokenizer.vision_bos_token)[0]
        vision_eos_token = tokenizer.encode(tokenizer.vision_eos_token)[0]
        audio_bos_token = tokenizer.encode(tokenizer.audio_bos_token)[0]
        audio_eos_token = tokenizer.encode(tokenizer.audio_eos_token)[0]
        audio_token = tokenizer.encode("<|audio_pad|>")[0]
        image_token = tokenizer.encode("<|image_pad|>")[0]
        video_token = tokenizer.encode("<|video_pad|>")[0]

        result = token_ids[:]
        if use_audio_in_video:
            while True:
                start = None
                for i in range(len(result) - 1):
                    if result[i: i + 2] == [vision_bos_token, audio_bos_token]:
                        start = i
                        break
                if start is not None:
                    end = None
                    for i in range(start + 2, len(result) - 1):
                        if result[i: i + 2] == [audio_eos_token, vision_eos_token]:
                            end = i
                            break
                    if end is not None:
                        result = result[:start] + [vision_bos_token, video_token, vision_eos_token] + result[end + 2:]
                else:
                    break

        for mm_token in [audio_token, image_token, video_token]:
            compressed = []
            for x in result:
                if x != mm_token or (not compressed or compressed[-1] != mm_token):
                    compressed.append(x)
            result = compressed
        
        return result


class Qwen3OmniMoeConditionalGenerationMixin(
    Qwen2_5OmniConditionalGenerationMixin
):
    
    def _validate_and_reshape_mm_tensor(self,
                                        mm_input: object,
                                        name: str,
                                        dim: int = 0) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if name == "feature_attention_mask":
            dim = -1
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input), dim=dim)
        else:
            if isinstance(mm_input[0], list):
                return torch.concat([torch.concat(mm_input[i], dim=dim) for i in range(len(mm_input))], dim=dim)
            else:
                return torch.concat(mm_input, dim=dim)
    
    def _get_feat_extract_output_lengths(
        self, 
        input_lengths: torch.Tensor
    )-> torch.Tensor:
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths, output_lengths

    def _process_audio_input(
        self,
        audio_input: Qwen2AudioInputs,
        audio_hashes: list[str] = None,
        cached_audio_features: torch.Tensor = None,
    ) -> torch.Tensor:

        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)

        if not isinstance(audio_feature_lengths, torch.Tensor):
            audio_feature_lengths = torch.cat(audio_feature_lengths) 
        if audio_feature_lengths.ndim == 2:
            audio_feature_lengths = audio_feature_lengths.reshape(
                -1)

        audio_feat_lengths, audio_output_lengths = (
            self._get_feat_extract_output_lengths(
                audio_feature_lengths))

        audio_outputs = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state
        return audio_features.split(audio_output_lengths.tolist())


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )
    
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return f"<|audio_start|><|audio_pad|><|audio_end|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        thinker_config: Qwen3OmniMoeThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config
        )
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        # force "use_flash_attention_2=True" to audio tower to align
        # the results.
        if flash_attn is not None:
            audio_config = thinker_config.audio_config
            audio_config._attn_implementation_autoset = True
            audio_config._attn_implementation = "flash_attention_2"
        else:
            logger.warning(
                "flash_attn is not available, the model may not yield the "
                "exactly same result as the transformers implementation "
                "in the audio tower part."
            )

        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)

        self.quant_config = quant_config
        self.use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
        )
        if (not multimodal_config.get_limit_per_prompt("image")
                and not multimodal_config.get_limit_per_prompt("video")):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                vision_config=thinker_config.vision_config,
                norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
            )

        self.language_model = Qwen3MoeLLMForCausalLM(
            vllm_config=vllm_config.with_hf_config(
                thinker_config.text_config,
                architectures=["Qwen3MoeForCausalLM"],
            ),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.use_deepstack = (
            self.visual is not None
            and getattr(
                thinker_config.vision_config, "deepstack_visual_indexes", None
            )
            is not None
        )
        self.deepstack_num_level = (
            len(thinker_config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = thinker_config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    thinker_config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None

    def _get_deepstack_input_embeds(self,
                                    num_tokens: int) -> Optional[IntermediateTensors]:
        if not self.use_deepstack or self.deepstack_input_embeds is None:
            return None
        return IntermediateTensors({
            f"deepstack_input_embeds_{idx}":
            self.deepstack_input_embeds[idx][:num_tokens]
            for idx in range(self.deepstack_num_level)
        })

    def _set_deepstack_input_embeds(self,
                                    deepstack_input_embeds: torch.Tensor) -> None:
        if not self.use_deepstack or self.deepstack_input_embeds is None:
            return
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self.deepstack_input_embeds = [
                torch.zeros(
                    num_tokens,
                    self.config.text_config.hidden_size,
                    device=self.deepstack_input_embeds[0].device,
                    dtype=self.deepstack_input_embeds[0].dtype,
                )
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx])

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        if not self.use_deepstack or self.deepstack_input_embeds is None:
            return
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = (
                    self._parse_and_validate_image_input(**kwargs)
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = (
                    self._parse_and_validate_video_input(**kwargs)
                )
            if (
                input_key in ("input_audio_features")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = (
                    self._parse_and_validate_audio_input(**kwargs)
                )
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs
        )
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += audio_embeddings
        return multimodal_embeddings

    def _compute_deepstack_embeds(
        self,
        *,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_visual: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], MultiModalEmbeddings]:
        if self.deepstack_num_level == 0:
            return None, multimodal_embeddings

        visual_embeddings: list[torch.Tensor] = []
        visual_lengths: list[int] = []
        processed_embeddings: list[torch.Tensor] = []

        for embeddings in multimodal_embeddings:
            if embeddings.shape[-1] == self.visual_dim + self.multiscale_dim:
                visual_lengths.append(len(embeddings))
                visual_main, visual_multiscale = torch.split(
                    embeddings,
                    [self.visual_dim, self.multiscale_dim],
                    dim=-1,
                )
                processed_embeddings.append(visual_main)
                visual_embeddings.append(visual_multiscale)
            else:
                processed_embeddings.append(embeddings)

        if not visual_embeddings:
            return None, multimodal_embeddings

        visual_multiscale = torch.split(
            torch.cat(visual_embeddings, dim=0),
            visual_lengths,
            dim=0,
        )

        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0),
            self.deepstack_num_level * inputs_embeds.size(1),
        )
        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=visual_multiscale,
            is_multimodal=is_visual,
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.shape[0], self.deepstack_num_level, self.visual_dim
        )
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)

        return deepstack_input_embeds, tuple(processed_embeddings)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._get_text_embeddings(
            input_ids,
            self.language_model.get_input_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        if is_multimodal is None:
            raise ValueError(
                "`get_input_embeddings` now requires `is_multimodal` arg, "
                "please update your model runner according to "
                "https://github.com/vllm-project/vllm/pull/16229."
            )

        deepstack_input_embeds: Optional[torch.Tensor] = None
        if self.use_deepstack and self.visual is not None:
            vision_token_ids = input_ids.new_tensor(
                [self.config.image_token_id, self.config.video_token_id]
            )
            is_visual = is_multimodal & torch.isin(input_ids, vision_token_ids)
            (
                deepstack_input_embeds,
                multimodal_embeddings,
            ) = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_visual=is_visual,
            )

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        deepstack_input_embeds: Optional[IntermediateTensors]
        deepstack_input_embeds = None
        if self.use_deepstack and inputs_embeds is not None and \
            get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0))
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if (self.use_deepstack and inputs_embeds is not None
                and get_pp_group().is_first_rank()):
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(
            hidden_states, sampling_metadata
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "code2wav."],
        )
        loaded_weights = loader.load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

        return loaded_weights
