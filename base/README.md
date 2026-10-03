# Base agents

The agents the platform publishes itself, as sources: each is the model's own
code behind the kit (`../kit/seg_agent_kit.py`), a Dockerfile, and a manifest.
`geometry.py` (where an image is in space, for a model that reads NIfTI) and
`palette.py` (a colour per label) are shared by those that need them.

| Agent | What it is | Image built from | State |
|---|---|---|---|
| `monai/` | MONAI's whole-body CT model (bundle `wholeBody_ct_segmentation` 0.2.7, its 3 mm model), automatic, 104 structures, for a GPU worker or a CPU one (`DEVICE`) | `python:3.11-slim`, MONAI and torch at the versions the bundle was made with, the bundle baked in | Built for a CPU and run on a synthetic CT on a developer's machine (`docs/superpowers/verification/2026-10-agent-marketplace-monai/`). **Never run on a GPU or on RunPod.** |
| `slip/` | SLIP, interactive, point prompts | the image production runs (`ghcr.io/imagin4care/slip-server:<commit>`), with `bridge/agent.py` in front of its server | Written; the bridge is tested against a stand-in for SLIP's server. **The image was never built or run.** |
| `totalsegmentator/` | TotalSegmentator, automatic, one task per image (`total` for CT, `total_mr` for MR), fast model, for a GPU worker or a CPU one (`DEVICE`) | `python:3.11-slim`, TotalSegmentator from PyPI, weights baked in | Built for a CPU and run on the sample head on a developer's machine (see `docs/superpowers/verification/2026-10-agent-marketplace-slice-4/`). **Never run on a GPU or on RunPod.** |
| `lingshu/` | Lingshu-7B, a medical vision-language model (MIT), answering questions about rendered slices; served by vLLM in the same container | `vllm/vllm-openai:<tag>`, the model's weights baked in (17 GB; an image of some 25 GB) | Written; its agent is tested against a stand-in for vLLM. **The image was never built or run**: it needs a 24 GB GPU. |
| `bridge/` | Not an agent of its own: the kit in front of any server that speaks nnInteractive's click protocol | — | Tested (`test_base.py`). |

nnInteractive itself is not a base agent on a paid site: its weights are CC
BY-NC-SA 4.0 (design, §13). Someone who may use it can build it the way SLIP
is built here, from `coendevente/nninteractive-slicer-server`, and run it on
their own RunPod key.

An automatic agent is sent the image in the units of its modality:
Hounsfield units for a CT (the app applies the series' rescale,
`apps/demo/src/agents/agentImage.ts`).

## Tests

```bash
cd apps/demo/deploy/agents
python -m unittest discover -s base     # the bridge, the geometry, MONAI's manifest; no model needed
```

## Building and publishing one

**By the workflow (the usual way).** The kit, the base agents and the
template are published as the public repository `imagin4care/seg-agents`
(`../publish/`: `publish.sh` lays it out from what is committed here). Its
workflow "Agent images" (Actions → Agent images → Run workflow, choosing the
agent) builds the image on GitHub's runners, pushes it to
`ghcr.io/imagin4care/seg-agent-<agent>:<commit>`, and leaves the manifest with
the image pinned to its digest, as the artifact `manifest-<agent>`. A new
package may be private: make it public once (Package settings → Change
visibility). It runs from that repository, not from this one: here, a
workflow run by hand must be on `main`, and a push to `main` deploys the site.

**By hand**, from `apps/demo/deploy/agents` (the Dockerfiles copy the kit from beside them):

```bash
# MONAI's whole-body CT model (for a CPU worker: add --build-arg DEVICE=cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu).
docker build -f base/monai/Dockerfile \
  --build-arg IMAGE=ghcr.io/imagin4care/seg-agent-monai-wholebody-ct:<tag> \
  -t ghcr.io/imagin4care/seg-agent-monai-wholebody-ct:<tag> .

# SLIP: on top of the image production runs.
docker build -f base/slip/Dockerfile \
  --build-arg SLIP_IMAGE=ghcr.io/imagin4care/slip-server:<commit> \
  -t ghcr.io/imagin4care/seg-agent-slip:<tag> .

# TotalSegmentator for MR (for CT: TASK=total, WEIGHTS=total_fast, and "-ct" in the names;
# for a CPU worker: add --build-arg DEVICE=cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu).
docker build -f base/totalsegmentator/Dockerfile \
  --build-arg TASK=total_mr --build-arg WEIGHTS=total_fast_mr \
  --build-arg IMAGE=ghcr.io/imagin4care/seg-agent-totalsegmentator-mr:<tag> \
  -t ghcr.io/imagin4care/seg-agent-totalsegmentator-mr:<tag> .

# Lingshu behind vLLM: name the tag of vllm/vllm-openai to build on (a recent one; the model's card says which it was tried with).
docker build -f base/lingshu/Dockerfile --build-arg VLLM=<tag> -t ghcr.io/imagin4care/seg-agent-lingshu:<tag> .

docker push <each>
```

A version of an agent is an image that never changes: a new build is a new
tag, and a new model a new `version` in the manifest.

The MONAI and TotalSegmentator images write their own manifest while they are
built, from the model's table of classes; the stage `manifest` of their
Dockerfile holds it alone. To take it out (with the same build arguments):

```bash
docker build -f base/monai/Dockerfile --build-arg IMAGE=<the same> --target manifest --output type=local,dest=out .
node base/pin-manifest.mjs out/manifest.json ghcr.io/imagin4care/seg-agent-monai-wholebody-ct <digest> > monai-wholebody-ct.json
```

`monai/manifest.json` and `totalsegmentator/manifest-mr.json` here are such
manifests, kept to be read and tested; the one registered is made with the image.

## Telling a server about them

The gateway reads the base agents at boot from `gateway/agents/*.json` (or the
folder `SEG_AGENTS_SEED` names), which a deploy copies to the server. Put each
manifest there with its image **pinned by digest** (`…/seg-agent-…@sha256:…`:
the workflow's artifact, or `pin-manifest.mjs` by hand): a tag can be moved, a
digest cannot, and a run names the digest it ran. The gateway's tests check
every file there against its schema, and that it is pinned. A base agent's
image must be public, and it cannot name secrets: what it needs is in its image.

An image built for a GPU does not start on a host whose CUDA is older than
its torch (it says so and stops, instead of running on the CPU at the price
of the GPU). SLIP's manifest asks RunPod for CUDA 13.0 hosts, as production
does; MONAI's and TotalSegmentator's ask for none in particular (their torch
is built for CUDA 12.4): a manifest can only list exact versions
(`gpu.cuda`), where RunPod also takes a floor (`gpu.minCudaVersion`) — to
settle when the agents first run there.

The check a user's agent goes through (the platform starts it, compares the
manifest it serves, runs it once) is not run on a base agent: run it by hand
the first time, by creating the same manifest as a user on a test stack.
