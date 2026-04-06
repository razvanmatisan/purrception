"""This file contains the definition of data loader using webdataset.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
import os
import random
from typing import List, Text, Union

import torch
import webdataset as wds
from torch.utils.data import default_collate
from webdataset.shardlists import expand_urls  # Helper to expand brace notation


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


class SimpleImageDataset:
    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        batch_size: int,
        num_classes: int,
        num_workers: int = 12,
        data_fraction: float = None,
        shuffle_shards_before_subset: bool = True,
        use_latent: bool = True,
        **kwargs,  #
    ):
        if data_fraction is not None and not (0.0 < data_fraction <= 1.0):
            raise ValueError("data_fraction must be between 0.0 (exclusive) and 1.0 (inclusive), or None")

        train_processing_pipeline = build_train_processing_pipeline(num_classes=num_classes, use_latent=use_latent)

        # 1. Expand the shard paths into a full list of URLs
        all_shard_urls = expand_urls(train_shards_path)
        num_total_shards = len(all_shard_urls)

        if num_total_shards == 0:
            raise ValueError(f"No shards found for path: {train_shards_path}")

        # Check if this is the main process (rank 0) for printing
        rank = int(os.environ.get("RANK", 0))
        is_main_process = rank == 0

        # 2. Determine the shards to use based on data_fraction
        if data_fraction is not None and data_fraction < 1.0:
            if shuffle_shards_before_subset and is_main_process:
                print("Shuffling shard list before taking subset...")
            random.shuffle(all_shard_urls)  # Shuffle the list in place

            # Calculate the number of shards for the subset
            # Use math.ceil to ensure we get at least one shard if fraction > 0
            num_subset_shards = max(1, int(math.ceil(num_total_shards * data_fraction)))
            # Ensure we don't exceed total if fraction is very close to 1.0 due to ceiling
            num_subset_shards = min(num_subset_shards, num_total_shards)

            selected_shard_urls = all_shard_urls[:num_subset_shards]
            if is_main_process:
                print(f"Total shards found: {num_total_shards}")
                print(f"Using data_fraction: {data_fraction}, selecting {len(selected_shard_urls)} shards.")
        else:
            # Use all shards if data_fraction is None or 1.0
            selected_shard_urls = all_shard_urls
            if is_main_process:
                print(f"Using all {num_total_shards} shards.")
        # --- Modification End ---

        # Create train dataset and loader.
        pipeline = [
            wds.ResampledShards(selected_shard_urls),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=5000, initial=1000),
            *train_processing_pipeline,
            wds.batched(batch_size, partial=False, collation_fn=default_collate),
        ]

        # Each worker is iterating over the complete dataset.
        self._train_dataset = wds.DataPipeline(*pipeline)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def train_dataset(self):
        return self._train_dataset

    def train_dataloader(self):
        return self._train_dataloader


def build_train_processing_pipeline(num_classes=0, use_latent=True):
    # Build the set of keys to keep
    keys = set()
    if num_classes > 0:
        keys.add("cls_id")
    if use_latent:
        keys.add("latent")
    print("keys", keys)

    # Build the rename mapping
    rename_map = {}
    if num_classes > 0:
        rename_map["cls_id"] = "cls;cls_id.cls"
    if use_latent:  # only save latent or image
        rename_map["latent"] = "latent.npy"
    print("rename_map", rename_map)

    # Build the pipeline
    pipeline = [
        wds.decode(
            wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"]),
        ),
        wds.rename(
            **rename_map,
            handler=wds.warn_and_continue,
        ),
        wds.map(filter_keys(keys)),
    ]

    if num_classes > 0:
        pipeline.append(
            wds.map_dict(
                **({"cls_id": (lambda x: int(x) if x is not None else -1)} if num_classes > 0 else {}),
                handler=wds.warn_and_continue,
            ),
        )

    if use_latent:

        def decode_latent(arr):
            # arr = np.load(io.BytesIO(arr))
            # print("latent shape", arr.shape, "dtype", arr.dtype, "std", arr.std())
            return torch.from_numpy(arr)

        pipeline.append(
            wds.map_dict(
                latent=decode_latent,
                handler=wds.warn_and_continue,
            )
        )

    return pipeline
