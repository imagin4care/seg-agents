# Segmentation agents

The agents of Imagin's segmentation marketplace, as sources:

- `kit/` — the kit an agent is written with (`seg_agent_kit.py`, Python's
  standard library only), and its tests.
- `base/` — the agents the platform publishes itself, each the model's own
  code behind the kit, with its Dockerfile (`base/README.md`).
- `template/` — what an author starts an agent of their own from.
- `.github/workflows/agent-images.yml` — builds the image of one base agent
  and publishes it to the GitHub Container Registry, with its manifest pinned
  to the image's digest. Run by hand: Actions → Agent images → Run workflow.

These files are published from the platform's own repository, where they are
written and tested; changes are made there.

## Images

| Agent | Image | Model | Licence of the model |
|---|---|---|---|
| `monai-wholebody-ct` | `ghcr.io/imagin4care/seg-agent-monai-wholebody-ct` | MONAI Model Zoo, `wholeBody_ct_segmentation` 0.2.7, 3 mm model: 104 structures of a CT | Apache-2.0 |
| `totalsegmentator-ct`, `totalsegmentator-mr` | `ghcr.io/imagin4care/seg-agent-totalsegmentator-ct`, `…-mr` | TotalSegmentator 2.18.0, tasks `total` and `total_mr`, fast models | Apache-2.0 |
| `slip` | `ghcr.io/imagin4care/seg-agent-slip` | SLIP (IRCAD) | Code GPL-3.0; the weights' licence is not stated |

Research use only: none of them is a medical device.

No licence has been chosen yet for this repository's own files; each model
keeps its own, above.
