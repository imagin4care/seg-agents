"""Splits NV-Reason-CT's checkpoint into six files while its image is built.

The model is published as one 10.6 GB `model.safetensors`, and GitHub's
registry refuses an image layer over 10 GB. Six files of about 1.8 GB, each
tensor kept whole, are copied into the image one layer each, with the index
that tells transformers which file holds which tensor
(`model.safetensors.index.json`). The model's other files (its code,
configuration, tokenizer, card and licence) are copied as they are.

    python shard.py <the downloaded model> <model folder> <shards folder>

The shards are written one after the other, each read alone from the
checkpoint, so that the build holds about one shard in memory at a time (a
GitHub runner has 16 GB). Reading the checkpoint's table and planning the
shards take the standard library only, so that they are tested without torch.
"""
import json
import os
import shutil
import struct
import sys

CHECKPOINT = "model.safetensors"
INDEX = "model.safetensors.index.json"
PARTS = 6


def table(path):
    """The metadata of a safetensors file, and the size in bytes of each of
    its tensors, from its header: eight bytes saying how long the header is,
    then the header in JSON."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    metadata = header.pop("__metadata__", None) or {}
    return metadata, {name: tensor["data_offsets"][1] - tensor["data_offsets"][0] for name, tensor in header.items()}


def plan(sizes, parts=PARTS):
    """The tensors of each of `parts` files, balanced by bytes: the largest
    tensor first, each into the lightest file so far."""
    if len(sizes) < parts:
        raise ValueError(f"{len(sizes)} tensors cannot fill {parts} files")
    files = [[] for _ in range(parts)]
    weights = [0] * parts
    for name in sorted(sizes, key=lambda n: (-sizes[n], n)):
        lightest = weights.index(min(weights))
        files[lightest].append(name)
        weights[lightest] += sizes[name]
    return [sorted(names) for names in files]


def shard_name(i, parts=PARTS):
    """The name transformers gives the `i`-th file (from 0) of a checkpoint in `parts`."""
    return f"model-{i + 1:05d}-of-{parts:05d}.safetensors"


def copy_the_rest(source, model):
    """Every file of `source` but the checkpoint, into `model`, its folders
    kept. Hidden ones are left: the download's own cache and git's files."""
    for folder, folders, names in os.walk(source):
        folders[:] = [f for f in folders if not f.startswith(".")]
        for name in names:
            if name.startswith(".") or (folder == source and name == CHECKPOINT):
                continue
            target = os.path.join(model, os.path.relpath(folder, source))
            os.makedirs(target, exist_ok=True)
            shutil.copy2(os.path.join(folder, name), target)


def split(source, model, shards):
    checkpoint = os.path.join(source, CHECKPOINT)
    if not os.path.isfile(checkpoint):
        sys.exit(f"shard.py: no {CHECKPOINT} in {source}: was the model downloaded?")
    from safetensors import safe_open
    from safetensors.torch import save_file

    metadata, sizes = table(checkpoint)
    copy_the_rest(source, model)
    os.makedirs(shards, exist_ok=True)
    weight_map = {}
    with safe_open(checkpoint, framework="pt") as whole:
        for i, names in enumerate(plan(sizes)):
            tensors = {name: whole.get_tensor(name) for name in names}
            save_file(tensors, os.path.join(shards, shard_name(i)), metadata={"format": "pt", **metadata})
            del tensors
            weight_map.update(dict.fromkeys(names, shard_name(i)))
            print(f"{shard_name(i)}: {len(names)} tensors, {sum(sizes[n] for n in names) / 1e9:.2f} GB", flush=True)
    with open(os.path.join(model, INDEX), "w", encoding="utf8") as f:
        json.dump({"metadata": {"total_size": sum(sizes.values())}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python shard.py <the downloaded model> <model folder> <shards folder>")
    split(*sys.argv[1:])
