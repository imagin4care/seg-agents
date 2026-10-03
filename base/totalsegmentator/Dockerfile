# syntax=docker/dockerfile:1
# TotalSegmentator as an automatic agent of the marketplace, for one task.
# Built from apps/demo/deploy/agents (the kit is copied from beside it), by
# the workflow "Agent images" or by hand:
#
#   docker build -f base/totalsegmentator/Dockerfile --build-arg TASK=total_mr --build-arg WEIGHTS=total_fast_mr \
#     --build-arg IMAGE=ghcr.io/<owner>/seg-agent-totalsegmentator-mr:<tag> -t <the same> .
#
# For CT: TASK=total, WEIGHTS=total_fast. (TotalSegmentator names the weights
# of a task's fast model in its own way: `totalseg_download_weights -h`.)
# For a worker without a GPU: DEVICE=cpu and
# TORCH_INDEX=https://download.pytorch.org/whl/cpu; the manifest then asks for
# a CPU worker, and the agent for no GPU.
#
# The weights of the task's fast model are baked in: a serverless worker
# keeps its image, and a volume would be billed at rest. The manifest is made
# from the model's own table of classes while the image is built; the stage
# `manifest` holds it alone (`--target manifest --output type=local,dest=…`).
FROM python:3.11-slim AS agent
# torch and torchvision come from one index, built for each other: left to
# TotalSegmentator's own requirements, torchvision is of another build and
# fails at the first import.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TOTALSEGMENTATOR=2.18.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 torch torchvision --index-url ${TORCH_INDEX} \
    && pip install --retries 10 TotalSegmentator==${TOTALSEGMENTATOR}

ARG TASK=total_mr
ARG WEIGHTS=total_fast_mr
ARG DEVICE=gpu
ENV TOTALSEG_TASK=${TASK} \
    TOTALSEG_DEVICE=${DEVICE} \
    TOTALSEG_HOME_DIR=/opt/totalsegmentator \
    PYTHONUNBUFFERED=1
# No usage statistics leave the agent; the weights of the fast model only.
RUN mkdir -p ${TOTALSEG_HOME_DIR} \
    && echo '{"send_usage_stats": false}' > ${TOTALSEG_HOME_DIR}/config.json \
    && totalseg_download_weights -t ${WEIGHTS}

WORKDIR /agent
COPY kit/seg_agent_kit.py base/geometry.py base/palette.py base/totalsegmentator/agent.py base/totalsegmentator/make_manifest.py ./
ARG IMAGE
RUN python make_manifest.py ${TASK} ${IMAGE} ${DEVICE} > manifest.json
EXPOSE 8000
CMD ["python", "agent.py"]

FROM scratch AS manifest
COPY --from=agent /agent/manifest.json /

# The image itself: the agent (the stage built when no target is named).
FROM agent
