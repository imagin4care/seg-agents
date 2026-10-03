# syntax=docker/dockerfile:1
# SLIP as an interactive agent of the marketplace: the image production already
# runs (the model, its checkpoint and its server), with the kit in front of it.
# Built from apps/demo/deploy/agents:
#
#   docker build -f base/slip/Dockerfile --build-arg SLIP_IMAGE=ghcr.io/imagin4care/slip-server:<commit> \
#     -t ghcr.io/imagin4care/seg-agent-slip:1.0.0 .
#
# The server of that image (app.py) is started by the agent on the loopback
# address, without the guard production puts before it (pod_main.py: token,
# uploads in pieces, self-deletion): the kit does that part here.
ARG SLIP_IMAGE
FROM ${SLIP_IMAGE}
WORKDIR /agent
COPY kit/seg_agent_kit.py base/bridge/agent.py base/slip/manifest.json ./
ENV BRIDGE_COMMAND="python -m uvicorn app:app --app-dir /app --host 127.0.0.1 --port 1529" \
    BRIDGE_URL=http://127.0.0.1:1529
EXPOSE 8000
CMD ["python", "agent.py"]
