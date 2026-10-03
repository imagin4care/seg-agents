# An agent for the segmentation marketplace

This folder is a repository to start from: an example agent, the kit that
serves it the way the platform calls it, a Dockerfile, and the two files that
make GitHub or GitLab build the image for you.

| File | What it is |
|---|---|
| `agent.py` | Your agent. The example grows a region from each click; replace what is in `Session` with your model. |
| `manifest.json` | What the agent says of itself: what it is for, what it accepts and returns, how it runs, its licence. |
| `seg_agent_kit.py` | The kit: one file, standard library only. It is not yours to change; its first lines say what it offers. |
| `Dockerfile` | The image: Python, the kit, your code, the manifest. Add what your model needs. |
| `.github/workflows/build.yml` | On GitHub: builds the image and pushes it to `ghcr.io/<you>/<repository>`. |
| `.gitlab-ci.yml` | On GitLab: builds the image and pushes it to the project's registry. |

## From here to an agent people use

1. **Write it.** An interactive agent is a class with the prompts it takes
   (`point`, `bbox`, `scribble`, `lasso`); an automatic one is a function
   that takes an image and returns a label map. Both are shown at the top of
   `seg_agent_kit.py`.
2. **Say what it is**, in `manifest.json`: its `name` (it cannot change
   later), its `version`, what it accepts. The platform's editor (AI agents ›
   New agent) shows every field and says what is wrong with one.
3. **Try it here**, without Docker:

   ```bash
   AGENT_TOKEN=test python agent.py
   curl localhost:8000/ping                                   # 200 once it is ready
   curl -H "x-agent-token: test" localhost:8000/manifest      # the manifest it serves
   ```

4. **Build the image.** Push this folder to a repository of yours on GitHub
   or GitLab and run its pipeline (or push a tag `v1.0.0`): it prints the
   address of the image. Or by hand: `docker build -t <registry>/<you>/<name>:1.0.0 . && docker push …`.
5. **Tell the platform.** AI agents › New agent: paste the manifest (the JSON
   view takes it whole) and set its image to the address the pipeline
   printed. The version in the manifest you paste and the one in the image
   must be the same: the platform compares them.
   - A **private** image: connect its registry first (AI agents › Private
     images), with a token that may only read.
   - **Secrets** (an API key, a licence): list their names under
     `runtime.secrets`, read them from the environment in your code, and give
     their values on the agent's page. Never put one in the image, in
     `runtime.env` or in this repository.
6. **Run the check.** The platform starts the image, compares the manifest it
   serves with the one you saved, sends it a test volume and tries every
   prompt it declares. A version that passes is published; one that does not
   says which step failed.

A version never changes once saved: a new build is a new `version`, in the
manifest and in the image.

## What your container gets, and does not

Its environment holds `PORT`, `AGENT_TOKEN` (the kit checks it on every
call), the settings of `runtime.env` and your secrets. Nothing of the
platform's. It is started when somebody uses the agent and stopped when
nobody does: what it writes to disk is gone at the next start.

Images are sent to it as arrays, with nothing that names a patient. Every
agent here is for research use only.
