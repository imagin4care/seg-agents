"""Writes the manifest of the MONAI whole-body CT agent from the bundle's own
metadata (its version, and its table of output channels): run when the image
is built, so the manifest an image serves is that of the bundle it carries.
Standard library only.

    python make_manifest.py <bundle folder> <image reference> <gpu|cpu> > manifest.json
"""
import json
import os
import sys

from palette import colour


def manifest(bundle, image, device):
    with open(os.path.join(bundle, "configs", "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    channels = meta["network_data_format"]["outputs"]["pred"]["channel_def"]
    labels = sorted((int(value), name) for value, name in channels.items() if int(value) > 0)
    return {
        "manifest": "seg-agent/1",
        "name": "monai-wholebody-ct",
        "title": "MONAI whole body (CT)",
        # The bundle's version, and the model used of the two it holds.
        "version": f"{meta['version']}-lowres",
        "summary": f"Segments {len(labels)} anatomical structures of a whole CT image by itself.",
        "description": "MONAI's whole-body CT model (a SegResNet trained on TotalSegmentator's data), with its 3 mm model: "
                       "organs, bones, muscles and vessels as one label each. "
                       + ("It takes seconds on a GPU. " if device == "gpu" else "On a CPU it takes a minute or more. ")
                       + "The labels are a draft to correct, not a result to use as it is.",
        "author": "MONAI Consortium",
        "purpose": {"category": "anatomy", "detail": "Every structure of a whole-body CT image."},
        "mode": "automatic",
        # 512 × 512 × 1024 voxels: a whole body at the resolution of a scanner.
        "accepts": {"image": {"modalities": ["CT"], "anatomy": ["whole-body"], "maxVoxels": 268435456}},
        "returns": {
            "kind": "labelmap",
            "labels": [{"value": value, "name": name.replace("_", " ").capitalize(), "color": colour(value)} for value, name in labels],
        },
        "contract": {"kind": "job", "version": 1, "dialect": "seg"},
        "runtime": {
            "type": "container", "image": image, "port": 8000, "gpu": {"memoryGb": 16} if device == "gpu" else None,
            "timeouts": {"startS": 300, "runS": 300 if device == "gpu" else 1200, "idleS": 60}, "maxSessions": 1,
            "models": [{"name": "wholeBody_ct_segmentation, 3 mm (model_lowres.pt)", "sizeGb": 0.08}],
        },
        "legal": {
            "license": {"code": "Apache-2.0", "weights": "Apache-2.0", "url": "https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation"},
            "commercialUse": "yes",
            "intendedUse": "Research on anatomical segmentation. Research use only, not a medical device.",
            "limitations": "CT only, in Hounsfield units. The 3 mm model is coarser than the 1.5 mm one the bundle also holds, "
                           "which needs some 29 GB of GPU memory and is not used here. Trained on TotalSegmentator's adult images.",
            "citation": "MONAI Consortium, Whole Body CT Segmentation (MONAI Model Zoo); "
                        "Wasserthal J. et al., TotalSegmentator, Radiology: AI, 2023 (the training data).",
            "homepage": "https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation",
        },
    }


if __name__ == "__main__":
    json.dump(manifest(sys.argv[1], sys.argv[2], sys.argv[3]), sys.stdout, indent=2)
    sys.stdout.write("\n")
