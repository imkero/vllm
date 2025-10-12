# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, Tuple

from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    (such as vision embeddings from images) during request processing. It
    provides memory-aware caching to avoid recomputing encoder outputs when the
    same multimodal inputs appear in different stages of request processing.

    This manager is particularly important for:
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
      cached
    - Any multimodal model where encoder computation is expensive and
      cacheable

    The cache operates at the granularity of individual multimodal input items
    within requests, allowing for fine-grained memory management and enabling
    chunked processing of multimodal inputs.

    Cache is enabled to share embeddings of same multimodal data 
    item (identified by their hash value) between different requests, 
    and eviction takes place at allocation time when there's no free 
    space for new embeddings.
    Oldest cached embeddings with no request referenced will be first evicted.
    
    Args:
        cache_size: Historical limit of the cache, measured in tokens. The
            value is still required for backward compatibility but is no longer
            used for accounting when `tail_size` or `max_items` are provided.

    Attributes:
        cache_size: Total cache capacity measured in encoder tokens (legacy).
        encoder_cache_tail_size: Number of tokens per encoder output retained
            in the cache (tail length).
        encoder_cache_max_item: Maximum number of encoder outputs that can be
            stored simultaneously.
        num_free_slots: Current available cache capacity measured in encoder
            outputs.
        num_freeable_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder outputs).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freeable: List of tuples (mm_hash, num_tokens) representing entries
            whose no current running request is needed and that can be freed to
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
            last call to get_freed_mm_hashes(). This list is cleared on return.
    """

    def __init__(self,
                 tail_size: int,
                 max_items: int):
        self.encoder_cache_tail_size = tail_size
        self.encoder_cache_max_item = max_items
        self.num_free_slots = self.encoder_cache_max_item
        self.num_freeable_slots = self.encoder_cache_max_item

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}

        # mm_hash of mm_data => cache slots required for the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    def _is_range_cached(self, request: Request, input_id: int,
                         needed_range: Optional[Tuple[int, int]]) -> bool:
        """Return True if the requested range is covered by the cached tail."""

        if needed_range is None:
            return True

        start, end = needed_range
        if start >= end:
            return True

        num_tokens = request.get_num_encoder_tokens(input_id)
        if num_tokens == 0:
            return True

        tail_len = min(num_tokens, self.encoder_cache_tail_size)
        tail_start = max(num_tokens - tail_len, 0)
        return start >= tail_start

    def check_and_update_cache(self,
                               request: Request,
                               input_id: int,
                               needed_range: Optional[Tuple[int, int]] = None
                               ) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            needed_range: Optional tuple representing the [start, end) range
                (relative to the encoder output) needed by the scheduler in
                this step. When provided, the method verifies the requested
                range is within the cached tail. If the range extends beyond
                the retained tail, the encoder output is treated as uncached.

        Returns:
            True if the encoder output for this input is already cached
        """
        mm_hash = request.mm_features[input_id].identifier
        # Not cached at all
        if mm_hash not in self.cached:
            return False

        if not self._is_range_cached(request, input_id, needed_range):
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_slots = self.freeable.pop(mm_hash, 0)
            self.num_freeable_slots -= num_slots

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(self, request: Request, input_id: int,
                     encoder_compute_budget: int,
                     num_items_to_schedule: int) -> bool:
        """Check if there's sufficient cache space for a multimodal input. 
        If there is, return True and update EncoderCacheManager state.

        If there is not enough free space in `num_free_slots` but there is
        enough reclaimable space in `num_freeable_slots`, entries will be
        evicted from `freeable` (their mm_hash appended to `freed`) until
        enough space is available, and then this method returns True. 
        Older entries are evicted first.
        
        Returns False only if the requested number of tokens exceeds both 
        the free and reclaimable capacities combined.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            encoder_compute_budget: Number of encoder tokens allowed to be 
                computed when this method is invoked.
            num_items_to_schedule: Number of encoder outputs already scheduled
                to be allocated with cache space when this method is invoked.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freeable` entries); otherwise
            False.

        Note: This method does not allocate physical memory for the encoder 
        output but only the state of EncoderCacheManager.
        """
        num_tokens = request.get_num_encoder_tokens(input_id)

        # Not enough compute budget
        if num_tokens > encoder_compute_budget:
            return False

        num_slots_needed = 1 + num_items_to_schedule

        # Enough free slots
        if num_slots_needed <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        if num_slots_needed > self.num_freeable_slots:
            return False

        # Not enough free slots but enough reclaimable slots
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_slots_needed > self.num_free_slots:
            mm_hash, num_slots = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_slots
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # that are scheduled since eviction takes place at can_allocate().
        assert self.num_free_slots >= 1
        assert self.num_freeable_slots >= 1

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= 1
        self.num_freeable_slots -= 1

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.

        Returns the set of input IDs whose `mm_hash` exists in the cache map.
        This includes entries that are currently unreferenced (and thus present
        in `freeable`); for such entries, freeing for this request will be a
        no-op.
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input (`mm_data`)

        When the reference set for the corresponding `mm_hash` becomes empty,
        the entry is appended to `freeable` and `num_freeable_slots` is
        increased by the number of encoder tokens for that input. 

        The entry is NOT physically freed until capacity is needed (e.g., by
        `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            self.freeable[mm_hash] = 1
            self.num_freeable_slots += 1

    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*.

        For each cached input ID, `free_encoder_input` is invoked.  
        The data stays in memory until eviction is triggered by a future 
        attempt allocation called by 'can_allocate'.

        Typically called when a request is finished, cancelled, or aborted.
        """
        input_ids = self.get_cached_input_ids(request).copy()
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        Returns:
            List of mm_hash strings that were actually evicted since the last
            call to be used by the scheduler to notify workers about which 
            encoder outputs can be removed from their caches. The internal 
            list is cleared after this call. 
        """
        freed = self.freed
        self.freed = []
        return freed


def compute_encoder_budget(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
    mm_registry: MultiModalRegistry,
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """
    if mm_registry.supports_multimodal_inputs(model_config):
        max_tokens_by_modality = mm_registry \
            .get_max_tokens_per_item_by_nonzero_modality(model_config)

        return compute_mm_encoder_budget(
            scheduler_config,
            max_tokens_by_modality,
        )

    return compute_text_encoder_budget(scheduler_config)


def compute_text_encoder_budget(
        scheduler_config: "SchedulerConfig") -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations for a text-only model.

    Args:
        scheduler_config: Scheduler configuration.

    Returns:
        - Compute budget for encoder execution, in unit of number of tokens 
            in the input sequence.
        - Space budget for encoder cache size, in unit of number of tokens 
            in the input sequence.
    """
    # Currently text-only encoder-decoder models are not supported
    return 0, 0


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    max_tokens_by_modality: Mapping[str, int],
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations for a multimodal model.

    Args:
        scheduler_config: Scheduler configuration.
        max_tokens_by_modality: The maximum number of tokens for each
            non-text modality.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """

    if not max_tokens_by_modality:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized.")
        return 0, 0

    max_tokens_per_mm_item = max(max_tokens_by_modality.values())

    if (scheduler_config.disable_chunked_mm_input and max_tokens_per_mm_item
            > scheduler_config.max_num_batched_tokens):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens.")

    encoder_compute_budget = max(scheduler_config.max_num_encoder_input_tokens,
                                 max_tokens_per_mm_item)
    encoder_cache_size = max(scheduler_config.encoder_cache_size,
                             max_tokens_per_mm_item)

    return encoder_compute_budget, encoder_cache_size
