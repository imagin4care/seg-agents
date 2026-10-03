# syntax=docker/dockerfile:1
# MONAI's whole-body CT segmentation (the bundle wholeBody_ct_segmentation of
# the MONAI Model Zoo) as an automatic agent of the marketplace. Built from
# apps/demo/deploy/agents (the kit is copied from beside it), by the workflow
# "Agent images" or by hand:
#
#   docker build -f base/monai/Dockerfile \
#     --build-arg IMAGE=ghcr.io/<owner>/seg-agent-monai-wholebody-ct:<tag> -t <the same> .
#
# For a worker without a GPU: DEVICE=cpu and
# TORCH_INDEX=https://download.pytorch.org/whl/cpu; the manifest then asks for
# a CPU worker, and the agent for no GPU.
#
# MONAI and torch are the versions the bundle was made with (its
# metadata.json). The bundle is baked in, with its 3 mm model only: a
# serverless worker keeps its image, and a volume would be billed at rest.
# The manifest is made from the bundle's metadata while the image is built;
# the stage `manifest` holds it alone (`--target manifest --output type=local,dest=…`).
FROM python:3.11-slim AS agent
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH=2.4.1
ARG MONAI=1.4.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 torch==${TORCH} --index-url ${TORCH_INDEX} \
    && pip install --retries 10 monai==${MONAI} "numpy<2" nibabel pytorch-ignite==0.4.11 huggingface_hub

ARG BUNDLE_VERSION=0.2.7
ENV MONAI_BUNDLE=/opt/bundle/wholeBody_ct_segmentation \
    PYTHONUNBUFFERED=1
# From where MONAI publishes its bundles today (Hugging Face, MONAI/<bundle>: what monai.bundle.download reads
# from MONAI 1.5 on; 1.4's asks a host that is gone), at the tag of its version, which does not move.
# The 3 mm model only, its configs, its documents and its licence.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='MONAI/wholeBody_ct_segmentation', revision='${BUNDLE_VERSION}', local_dir='${MONAI_BUNDLE}', allow_patterns=['configs/*', 'docs/*', 'LICENSE', 'models/model_lowres.pt'])" \
    && test -f ${MONAI_BUNDLE}/models/model_lowres.pt && test -f ${MONAI_BUNDLE}/configs/inference.json

ARG DEVICE=gpu
ENV MONAI_DEVICE=${DEVICE}
WORKDIR /agent
COPY kit/seg_agent_kit.py base/geometry.py base/palette.py base/monai/agent.py base/monai/make_manifest.py ./
ARG IMAGE
RUN python make_manifest.py ${MONAI_BUNDLE} ${IMAGE} ${DEVICE} > manifest.json
EXPOSE 8000
CMD ["python", "agent.py"]

FROM scratch AS manifest
COPY --from=agent /agent/manifest.json /

# The image itself: the agent (the stage built when no target is named).
FROM agent
