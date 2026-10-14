import os
import threading
from pathlib import Path
from .model_persister import ModelPersister
from safetensors.torch import load_file, save_file


class SafetensorModelPersister(ModelPersister):

    # Layer -> execution index (matches AirLLM layer_names order):
    #   0 = model.embed_tokens, 1..64 = model.layers.0..63, 65 = model.norm, 66 = lm_head
    LAYERS_PER_BLOCK = 4

    def __init__(self):
        # Main thread's current block (the one whose layers are being consumed).
        self._current_block_id = -1
        self._current_block = None
        # Prefetch worker's block (loaded ahead of the main thread).
        self._prefetch_block_id = -1
        self._prefetch_block = None
        # Pinned pools back the block views (one per slot, so prefetch never overwrites the
        # block the main thread is still consuming). A pinned pool makes the main thread's
        # host->device copy (v.cuda() during dequantisation) run at ~12-28 GB/s instead of
        # ~8 GB/s pageable, and - crucially - does not contend with the worker's I/O.
        self._current_pool = None
        self._prefetch_pool = None
        self._block_lock = threading.Lock()

    @staticmethod
    def _layer_index(layer_name):
        # split_and_save_layers passes tensor-name prefixes that keep the trailing dot
        # (e.g. "model.layers.0."), while runtime layer_names have no trailing dot.
        layer_name = layer_name.rstrip('.')
        if layer_name == 'model.embed_tokens':
            return 0
        if layer_name == 'model.norm':
            return 65
        if layer_name == 'lm_head':
            return 66
        if layer_name.startswith('model.layers.'):
            return 1 + int(layer_name.rsplit('.', 1)[-1])
        raise KeyError(layer_name)

    def _block_id_for(self, layer_name):
        return self._layer_index(layer_name) // self.LAYERS_PER_BLOCK

    def _block_path(self, saving_path, block_id):
        return Path(saving_path) / f'model.block_{block_id:03d}.safetensors'

    def _copy_to_pool(self, sd, slot):
        """Copy a super-block's tensors into a pinned pool and return pinned CPU views.

        The returned dict shares the pool's storage; the caller must not hold references to a
        previous block's views once a new block overwrites the same pool (slot ownership is
        managed by ``promote_prefetch``/``_get_block`` so this never happens).
        """
        import torch
        total = sum(v.numel() * v.element_size() for v in sd.values())
        pool_attr = f'_{slot}_pool'
        pool = getattr(self, pool_attr, None)
        # +64KB covers the 256-byte alignment padding (44 tensors * <256B worst case).
        if pool is None or pool.numel() < total + 65536:
            pool = torch.empty(total + 65536, dtype=torch.uint8, pin_memory=True)
            setattr(self, pool_attr, pool)
        views = {}
        offset = 0
        for k, v in sd.items():
            src = v.view(torch.uint8) if v.dtype != torch.uint8 else v
            src = src.flatten()
            nbytes = src.numel()
            offset = (offset + 255) // 256 * 256
            pool[offset:offset + nbytes].copy_(src)
            views[k] = pool[offset:offset + nbytes].view(v.dtype).view(v.shape)
            offset += nbytes
        return views

    def _get_block(self, block_id, path, use_prefetch_cache=False):
        """Load a whole super-block into a pinned CPU pool, keeping two dedicated slots.

        The main thread consumes the ``_current_block`` slot; the prefetch worker writes the next
        block into ``_prefetch_block`` so the two never evict each other. When the main thread
        reaches a block the worker already fetched, the prefetch slot is promoted to current
        (no reload). A double-checked lock serialises only the cold/reload path.

        The returned tensors are CPU views into a pinned pool (raw 4bit/quant params, not yet
        dequantised). The main thread dequantises on GPU afterwards, so ``v.cuda()`` runs from
        pinned memory (fast H2D) and never contends with the worker's disk I/O.
        """
        if not use_prefetch_cache:
            if self._current_block_id == block_id and self._current_block is not None:
                return self._current_block
            with self._block_lock:
                if self._current_block_id == block_id and self._current_block is not None:
                    return self._current_block
                # Promote the prefetched block if the worker already fetched exactly this one.
                if self._prefetch_block_id == block_id and self._prefetch_block is not None:
                    self._current_block = self._prefetch_block
                    self._current_block_id = block_id
                    self._current_pool = self._prefetch_pool
                    self._prefetch_block = None
                    self._prefetch_block_id = -1
                    self._prefetch_pool = None
                    return self._current_block
                sd = load_file(self._block_path(path, block_id), device='cpu')
                block = self._copy_to_pool(sd, 'current')
                self._current_block = block
                self._current_block_id = block_id
                return block

        # Prefetch worker path: write into the dedicated prefetch slot only.
        if self._prefetch_block_id == block_id and self._prefetch_block is not None:
            return self._prefetch_block
        with self._block_lock:
            if self._prefetch_block_id == block_id and self._prefetch_block is not None:
                return self._prefetch_block
            sd = load_file(self._block_path(path, block_id), device='cpu')
            block = self._copy_to_pool(sd, 'prefetch')
            self._prefetch_block = block
            self._prefetch_block_id = block_id
            return block

    def promote_prefetch(self):
        """Promote the prefetched block into the main thread's current slot.

        Called by the main thread right after consuming a prefetch future: block-internal layers
        must find the block in the current slot, while the worker immediately starts filling the
        prefetch slot with the next block. The slots (block dicts AND pools) are swapped so the
        worker reuses the old current pool (reallocated if the next block is bigger).
        """
        with self._block_lock:
            if self._prefetch_block is not None:
                self._current_block, self._prefetch_block = self._prefetch_block, self._current_block
                self._current_block_id, self._prefetch_block_id = self._prefetch_block_id, self._current_block_id
                self._current_pool, self._prefetch_pool = self._prefetch_pool, self._current_pool
                self._prefetch_block = None
                self._prefetch_block_id = -1
                self._prefetch_pool = None

    def model_persist_exist(self, layer_name, saving_path):
        mmap_exists = os.path.exists(str(saving_path / (layer_name + 'mmap')))
        mmap_done = os.path.exists(str(saving_path / (layer_name + 'mmap.done')))
        if mmap_exists and mmap_done:
            return True

        safetensor_exists = os.path.exists(str(saving_path / (layer_name + 'safetensors')))
        done_marker_exists = os.path.exists(str(saving_path / (layer_name + 'safetensors.done')))
        if safetensor_exists and done_marker_exists:
            return True

        # Super-block format: block file + done marker.
        try:
            block_id = self._block_id_for(layer_name)
        except KeyError:
            return False
        block_exists = self._block_path(saving_path, block_id).exists()
        block_done = os.path.exists(str(saving_path / f'model.block_{block_id:03d}.safetensors.done'))
        return block_exists and block_done

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + 'safetensors'))
        print(f"saved as: {saving_path / (layer_name + 'safetensors')}")
        (saving_path / (layer_name + 'safetensors.done')).touch()

    def load_model(self, layer_name, path, use_prefetch_cache=False):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prefer super-block files when present.
        block_path = self._block_path(path, self._block_id_for(layer_name))
        if block_path.exists():
            block_id = self._block_id_for(layer_name)
            block = self._get_block(block_id, path, use_prefetch_cache=use_prefetch_cache)
            prefix = layer_name + '.'
            return {k: v for k, v in block.items() if k.startswith(prefix)}

        # Legacy per-layer fallback.
        layer_state_dict = load_file(Path(path) / (layer_name + ".safetensors"), device=device)
        return layer_state_dict