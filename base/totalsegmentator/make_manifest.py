"""Writes the manifest of the TotalSegmentator agent for one task, from
TotalSegmentator's own table of classes: run when the image is built, so the
manifest an image serves is the one of the model it carries.

    python make_manifest.py <task> <image reference> <gpu|cpu> > manifest.json

The manifest registered on a server is that same file (see README.md here).
"""
import json
import sys
from importlib.metadata import version

from palette import colour
from totalsegmentator.registry import get_task_classes

TASKS = {
    "total": {"name": "totalsegmentator-ct", "title": "TotalSegmentator (CT)", "modality": "CT", "weights": "0.13"},
    "total_mr": {"name": "totalsegmentator-mr", "title": "TotalSegmentator (MR)", "modality": "MR", "weights": "0.13"},
}


def manifest(task, image, device):
    about = TASKS[task]
    classes = {int(value): name for value, name in get_task_classes(task).items() if int(value) > 0}
    return {
        "manifest": "seg-agent/1",
        "name": about["name"],
        "title": about["title"],
        "version": f"{version('TotalSegmentator')}-fast",
        "summary": f"Segments {len(classes)} anatomical structures of a whole {about['modality']} image by itself.",
        "description": "TotalSegmentator, with its fast model (3 mm): organs, bones, muscles and vessels as one label each. "
                       + ("It takes about a minute on a GPU. " if device == "gpu" else "On a CPU it takes several minutes. ")
                       + "The labels are a draft to correct, not a result to use as it is.",
        "author": "Wasserthal et al., University Hospital Basel",
        "purpose": {"category": "anatomy", "detail": f"Every structure of a whole-body {about['modality']} image."},
        "mode": "automatic",
        # 512 × 512 × 1024 voxels: a whole body at the resolution of a scanner.
        "accepts": {"image": {"modalities": [about["modality"]], "anatomy": ["whole-body"], "maxVoxels": 268435456}},
        "returns": {
            "kind": "labelmap",
            "labels": [{"value": value, "name": name.replace("_", " ").capitalize(), "color": colour(value)} for value, name in sorted(classes.items())],
        },
        "contract": {"kind": "job", "version": 1, "dialect": "seg"},
        "runtime": {
            "type": "container", "image": image, "port": 8000, "gpu": {"memoryGb": 16} if device == "gpu" else None,
            "timeouts": {"startS": 300, "runS": 600 if device == "gpu" else 1800, "idleS": 60}, "maxSessions": 1,
            "models": [{"name": f"TotalSegmentator {task}, 3 mm", "sizeGb": float(about["weights"])}],
        },
        "legal": {
            "license": {"code": "Apache-2.0", "weights": "Apache-2.0", "url": "https://github.com/wasserth/TotalSegmentator"},
            "commercialUse": "yes",
            "intendedUse": "Research on anatomical segmentation. Research use only, not a medical device.",
            "limitations": "Trained on adult images; the fast model is coarser than the full one. Other tasks of TotalSegmentator need a licence and are not included.",
            "citation": "Wasserthal J. et al., TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images, Radiology: AI, 2023; "
                        "Akinci D'Antonoli T. et al., TotalSegmentator MRI, Radiology, 2025.",
            "homepage": "https://github.com/wasserth/TotalSegmentator",
        },
    }


if __name__ == "__main__":
    json.dump(manifest(sys.argv[1], sys.argv[2], sys.argv[3]), sys.stdout, indent=2)
    sys.stdout.write("\n")
