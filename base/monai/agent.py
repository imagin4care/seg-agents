"""MONAI's whole-body CT segmentation as an automatic agent: the bundle
`wholeBody_ct_segmentation` of the MONAI Model Zoo (Apache-2.0, code and
weights), with its 3 mm model, segments a whole CT image into 104 anatomical
structures.

The bundle is run as it is published: its own inference config (resampling,
normalisation, sliding window, and back to the image's grid), told only where
the image is, where to write, and which of its two models to use. The image
is written as a NIfTI file and the labels are read from the one it writes.
The manifest beside this file is made from the bundle's metadata by
make_manifest.py when the image is built.
"""
import glob
import os
import tempfile

import nibabel
import numpy

from geometry import ras_affine, refusal
from seg_agent_kit import ImageRefused, LabelMap, serve

BUNDLE = os.environ.get("MONAI_BUNDLE", "/opt/bundle/wholeBody_ct_segmentation")
DEVICE = os.environ.get("MONAI_DEVICE", "gpu")


def load():
    """Imports the model code (torch, MONAI), which takes seconds, before the
    agent says it is ready. An image built for a GPU that sees none does not
    start."""
    import monai.bundle  # noqa: F401 — imported for its loading
    import torch

    # It would run on the CPU, slowly, at the price of the GPU.
    if DEVICE == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("this image is built for a GPU and sees none (is the host's CUDA older than the image's torch?)")


def run_bundle(source, out):
    """The bundle's inference on the NIfTI file `source`, written under `out`."""
    from monai.bundle import ConfigWorkflow

    configs = os.path.join(BUNDLE, "configs")
    workflow = ConfigWorkflow(
        config_file=os.path.join(configs, "inference.json"), meta_file=os.path.join(configs, "metadata.json"),
        logging_file=False, workflow_type="infer",
        **{
            "bundle_root": BUNDLE,
            # The 3 mm model (6 GB of GPU memory); the 1.5 mm one needs some 29 GB.
            "displayable_configs::highres": False,
            "datalist": [source],
            "output_dir": out,
            "output_ext": ".nii",
            # No worker processes: a container's shared memory is small.
            "dataloader::num_workers": 0,
            "evaluator::amp": DEVICE == "gpu",
            # The weights were saved from a GPU: they are loaded onto the device the bundle runs on.
            "checkpointloader::map_location": "$@device",
        },
    )
    workflow.initialize()
    workflow.run()


def segment(image, job):
    reason = refusal(image.shape, job.geometry)
    if reason:
        raise ImageRefused(reason)
    with tempfile.TemporaryDirectory() as work:
        job.progress(0.05, "Reading the image")
        # The platform sends voxels as (z, y, x) in patient space LPS, in Hounsfield units; NIfTI wants (x, y, z) and RAS.
        source = os.path.join(work, "image.nii")
        nibabel.save(nibabel.Nifti1Image(numpy.ascontiguousarray(numpy.transpose(image.numpy(), (2, 1, 0))), numpy.array(ras_affine(job.geometry))), source)
        job.progress(0.1, "Segmenting")
        run_bundle(source, os.path.join(work, "labels"))
        job.progress(0.95, "Writing the labels")
        [written] = glob.glob(os.path.join(work, "labels", "**", "*.nii"), recursive=True)
        labels = numpy.rint(numpy.squeeze(numpy.asanyarray(nibabel.load(written).dataobj))).astype(numpy.uint8)
    labels = numpy.ascontiguousarray(numpy.transpose(labels, (2, 1, 0)))
    if labels.shape != tuple(image.shape):
        raise RuntimeError(f"the bundle wrote labels of {labels.shape} for an image of {tuple(image.shape)}")
    return LabelMap.from_numpy(labels)


if __name__ == "__main__":
    serve(job=segment, load=load)
