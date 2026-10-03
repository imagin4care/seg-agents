# syntax=docker/dockerfile:1
# Lingshu-7B (a medical vision-language model, MIT) as an agent that answers
# in words: vLLM serves the model, the kit stands in front of it.
# Built from apps/demo/deploy/agents (the kit is copied from beside it):
#
#   docker build -f base/lingshu/Dockerfile --build-arg VLLM=<tag of vllm/vllm-openai> \
#     -t ghcr.io/<owner>/seg-agent-lingshu:1.0.0 .
#
# The model's weights (some 17 GB) are baked in: a serverless worker keeps its
# image, and a volume would be billed at rest. The image is about 25 GB.
ARG VLLM
FROM vllm/vllm-openai:${VLLM}
ARG MODEL=lingshu-medical-mllm/Lingshu-7B
ENV CHAT_MODEL=${MODEL} \
    HF_HOME=/opt/models \
    HF_HUB_OFFLINE=0 \
    PYTHONUNBUFFERED=1
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"
# Nothing is fetched once the image is built: the worker starts with what it carries.
ENV HF_HUB_OFFLINE=1 \
    CHAT_URL=http://127.0.0.1:8001 \
    CHAT_COMMAND="vllm serve ${MODEL} --host 127.0.0.1 --port 8001 --max-model-len 16384 --limit-mm-per-prompt.image 20"
WORKDIR /agent
COPY kit/seg_agent_kit.py base/lingshu/agent.py base/lingshu/manifest.json ./
EXPOSE 8000
# The base image starts vLLM's own server: here the agent does, behind the kit.
ENTRYPOINT []
CMD ["python3", "agent.py"]
