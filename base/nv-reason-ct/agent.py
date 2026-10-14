"""NV-Reason-CT as an agent that reads the whole volume and answers in words:
NVIDIA's 3D vision-language model of CT (https://huggingface.co/nvidia/NV-Reason-CT,
OpenMDW-1.1, code and weights) writes a structured report, a reasoning
analysis or the answer to a question about a chest or abdominal CT, and
answers follow-up questions about it.

The model is run as its card shows: the image goes to its own processor as a
NIfTI file in Hounsfield units (which reorients it to LPS, resamples it to
2 mm and crops 192³ voxels around the region asked), the conversation goes
through its chat template, and what it generates is read as it is written,
its reasoning apart from its answer (thinking.py). The platform sends the
image already resampled to 2 mm (the manifest's `accepts.image.spacing`): the
processor's own resampling then has nothing left to do.

Environment: NV_MODEL (the model's folder, default /opt/model).
"""
import os
import tempfile
import threading

import nibabel
import numpy

from geometry import ras_affine, refusal
from seg_agent_kit import ImageRefused, serve
from thinking import Router

MODEL_DIR = os.environ.get("NV_MODEL", "/opt/model")
# The most the manifest lets a user ask for: the reasoning and the answer share it.
MAX_TOKENS = 4096
# One model on one GPU: one generation at a time, whatever reaches the worker.
GENERATING = threading.Lock()
model = processor = None


def load():
    """Reads the model onto the GPU once, before the agent says it is ready.
    An image that sees no GPU does not start: on a CPU it would write for hours."""
    global model, processor
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("this image is built for a GPU and sees none (is the host's CUDA older than the image's torch?)")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval().to("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)


def conversation(history, question):
    """The chat the model is given: the earlier turns as text, then the
    question; the image once, at the start of the first question, as NVIDIA's
    own demo does."""
    messages = [{"role": turn["role"], "content": [{"type": "text", "text": turn["text"]}]} for turn in history]
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
    first = next(m for m in messages if m["role"] == "user")
    first["content"].insert(0, {"type": "image"})
    return messages


def read(image, job):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    reason = refusal(image.shape, job.geometry)
    if reason:
        raise ImageRefused(reason)
    thinking = bool(job.parameters.get("reasoning", True))
    most = int(job.parameters.get("maxTokens", MAX_TOKENS))
    with GENERATING:
        try:
            job.progress(0.02, "Reading the scan")
            with tempfile.TemporaryDirectory() as work:
                # The platform sends voxels as (z, y, x) in patient space LPS, in Hounsfield units; NIfTI wants (x, y, z) and RAS.
                source = os.path.join(work, "image.nii.gz")
                voxels = numpy.ascontiguousarray(numpy.transpose(image.numpy(), (2, 1, 0)))
                nibabel.save(nibabel.Nifti1Image(voxels, numpy.array(ras_affine(job.geometry), dtype=numpy.float64)), source)
                prompt = processor.apply_chat_template(
                    conversation(job.history, job.question), tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
                # The processor reads the file now: it is not needed once the inputs are made.
                inputs = processor(text=prompt, images3d=[source], anatomy_region=job.parameters.get("region", "chest"),
                                   return_tensors="pt").to(model.device)
            prompt_length = inputs["input_ids"].shape[1]

            class Watch(StoppingCriteria):
                """Called after each token: says how far the writing is, and
                stops it once nobody waits for it."""

                written = 0

                def __call__(self, input_ids, scores, **kwargs):
                    self.written = input_ids.shape[1] - prompt_length
                    job.progress(0.05 + 0.9 * self.written / most, "Writing")
                    return torch.full((input_ids.shape[0],), job.cancelled(), dtype=torch.bool, device=input_ids.device)

            watch = Watch()
            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            failure = []

            def generate():
                try:
                    with torch.inference_mode():
                        model.generate(**inputs, max_new_tokens=most, do_sample=False, use_cache=True,
                                       streamer=streamer, stopping_criteria=StoppingCriteriaList([watch]))
                except Exception as e:  # noqa: BLE001 — raised again where the job is run, once the text is read
                    failure.append(e)
                    # The text is read until the streamer says it is over: a generation that failed says so too.
                    streamer.end()

            writer = threading.Thread(target=generate, daemon=True)
            writer.start()
            router = Router(thinking, job.reason, job.answer)
            for piece in streamer:
                router.feed(piece)
            writer.join()
            router.close()
            if failure:
                raise failure[0]
            if router.in_reasoning and watch.written >= most:
                raise ValueError("the reasoning used every token: ask again with more tokens, or without reasoning")
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    serve(job=read, load=load)
