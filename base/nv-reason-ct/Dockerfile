# syntax=docker/dockerfile:1
# NV-Reason-CT (NVIDIA's 3D vision-language model of CT, OpenMDW-1.1) as an
# agent that reads the whole volume and answers in words. Built from
# apps/demo/deploy/agents (the kit is copied from beside it), by the workflow
# "Agent images" on GitHub's runners, or by hand on a machine with some 60 GB
# of free disk (the build holds the checkpoint twice beside torch):
#
#   docker build -f base/nv-reason-ct/Dockerfile -t ghcr.io/<owner>/seg-agent-nv-reason-ct:<tag> .
#
# torch for CUDA 12.8, and the model's requirements at exact versions: its own
# pins (transformers, dynamic-network-architectures) and the first MONAI it
# allows. The checkpoint, at a revision that does not move, is baked in: a
# serverless worker keeps its image, and a volume would be billed at rest. It
# is one 10.6 GB file, and GitHub's registry refuses a layer over 10 GB:
# shard.py splits it into six files of about 1.8 GB, each copied in a layer of
# its own. The image is some 18 GB. The model's LICENSE stays beside its
# weights (/opt/model/LICENSE), as OpenMDW asks of a copy.
#
# The manifest is the file beside this one; the stage `manifest` holds it
# alone (`--target manifest --output type=local,dest=…`), as the stage of the
# agents whose manifest is made while they are built does.
FROM python:3.11-slim AS runtime
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH=2.9.1
ARG TORCHVISION=0.24.1
# torch with its CUDA libraries is some 6 GB: a layer of its own, kept when the
# rest changes. torchvision (which timm imports) comes from the same index,
# built for that torch: from PyPI it could be of another build.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 torch==${TORCH} torchvision==${TORCHVISION} --index-url ${TORCH_INDEX}
# The model's requirements.txt at exact versions: the latest that suit its pins
# and Python 3.11 (timm below 1.0.23 for dynamic-network-architectures 0.4.3,
# huggingface_hub below 2 for transformers 5.6.2), but numpy, kept at the 2.3
# series MONAI 1.5.1 was released with (2.4 came after it).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 transformers==5.6.2 monai==1.5.1 dynamic-network-architectures==0.4.3 \
      nibabel==5.4.2 scipy==1.17.1 numpy==2.3.5 Pillow==12.3.0 timm==1.0.22 einops==0.8.2 \
      safetensors==0.8.0 huggingface_hub==1.33.0

# The model's files, its checkpoint split in six (shard.py), in one step: the
# download is removed in the step that made it, so that no layer keeps it.
# Its picture is left out; its card and licence are kept.
FROM runtime AS weights
ARG REVISION=386b93e034983f6c1fc841a43833a1b6a0cd9c13
COPY base/nv-reason-ct/shard.py /w/
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/NV-Reason-CT', revision='${REVISION}', local_dir='/w/src', ignore_patterns=['assets/*', '.git*'])" \
    && python /w/shard.py /w/src /w/model /w/shards \
    && rm -rf /w/src

FROM scratch AS manifest
COPY base/nv-reason-ct/manifest.json /manifest.json

# The image itself (the stage built when no target is named).
FROM runtime
COPY --from=weights /w/model/ /opt/model/
# One shard a layer: GitHub's registry takes no layer over 10 GB.
COPY --from=weights /w/shards/model-00001-of-00006.safetensors /opt/model/
COPY --from=weights /w/shards/model-00002-of-00006.safetensors /opt/model/
COPY --from=weights /w/shards/model-00003-of-00006.safetensors /opt/model/
COPY --from=weights /w/shards/model-00004-of-00006.safetensors /opt/model/
COPY --from=weights /w/shards/model-00005-of-00006.safetensors /opt/model/
COPY --from=weights /w/shards/model-00006-of-00006.safetensors /opt/model/
# Nothing is fetched once the image is built: the worker starts with what it
# carries. transformers copies the model's own code (trust_remote_code) under
# HF_HOME before it imports it: a folder anyone running the container can write.
ENV NV_MODEL=/opt/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    PYTHONUNBUFFERED=1
WORKDIR /agent
COPY kit/seg_agent_kit.py base/geometry.py base/nv-reason-ct/thinking.py base/nv-reason-ct/agent.py base/nv-reason-ct/manifest.json ./
EXPOSE 8000
CMD ["python", "agent.py"]
