"""TotalSegmentator as an automatic agent: it segments a whole CT or MR image
into its anatomical structures (https://github.com/wasserth/TotalSegmentator,
Apache-2.0, code and the weights of the tasks used here).

One image serves one task (`TOTALSEG_TASK`: `total` for CT, `total_mr` for
MR), with the fast 3 mm model, whose weights the image carries, on one kind of
worker (`TOTALSEG_DEVICE`: `gpu`, or `cpu` for an image built for one). The
manifest beside this file is made for that task and that worker by
make_manifest.py when the image is built: its table of labels is
TotalSegmentator's own.
"""
import os

import nibabel
import numpy

from geometry import ras_affine, refusal
from seg_agent_kit import ImageRefused, LabelMap, serve

TASK = os.environ.get("TOTALSEG_TASK", "total_mr")
DEVICE = os.environ.get("TOTALSEG_DEVICE", "gpu")


def load():
    """Imports the model code (torch, nnU-Net), which takes seconds, before
    the agent says it is ready. An image built for a GPU that sees none does
    not start."""
    import torch
    import totalsegmentator.python_api  # noqa: F401 — imported for its loading

    # It would run on the CPU, slowly, at the price of the GPU.
    if DEVICE == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("this image is built for a GPU and sees none (is the host's CUDA older than the image's torch?)")


def segment(image, job):
    from totalsegmentator.python_api import totalsegmentator

    reason = refusal(image.shape, job.geometry)
    if reason:
        raise ImageRefused(reason)
    job.progress(0.05, "Reading the image")
    # The platform sends voxels as (z, y, x) in patient space LPS; NIfTI wants (x, y, z) and RAS.
    volume = nibabel.Nifti1Image(numpy.ascontiguousarray(numpy.transpose(image.numpy(), (2, 1, 0))), numpy.array(ras_affine(job.geometry)))
    job.progress(0.1, "Segmenting")
    result = totalsegmentator(volume, None, task=TASK, fast=True, ml=True, quiet=True, device=DEVICE)
    job.progress(0.95, "Writing the labels")
    labels = numpy.asanyarray(result.dataobj).astype(numpy.uint8)
    return LabelMap.from_numpy(numpy.ascontiguousarray(numpy.transpose(labels, (2, 1, 0))))


if __name__ == "__main__":
    serve(job=segment, load=load)
