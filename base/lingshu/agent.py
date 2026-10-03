"""An open model that reads medical images, as an agent that answers in
words: the model is served by vLLM beside this file, in the same container,
and reached by nobody else; the kit in front of it takes care of the
platform's side.

vLLM speaks OpenAI's chat API: each picture goes in as a data URL after its
caption, the question last.

Environment: CHAT_MODEL (the model vLLM serves, e.g.
lingshu-medical-mllm/Lingshu-7B), CHAT_COMMAND (how vLLM is started; left out
when something else starts it), CHAT_URL (where it listens, default
http://127.0.0.1:8001), CHAT_START_S (how long it may take to load the
model, default 840).
"""
import base64
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request

from seg_agent_kit import serve

URL = os.environ.get("CHAT_URL", "http://127.0.0.1:8001")
MODEL = os.environ.get("CHAT_MODEL", "")
START_S = float(os.environ.get("CHAT_START_S", "840"))
# The platform gives up on a question after the time the manifest allows; this only ends one that never would.
CALL_S = 3600
MAX_ANSWER_TOKENS = 1024
SYSTEM = (
    "You are shown a few rendered slices of a medical image, each with a caption that says where it was taken, "
    "and a short description of the image. Answer the question in plain words, from what is visible. "
    "You see slices, not the whole volume: say so when the question cannot be answered from them. "
    "This is for research and teaching: describe, do not diagnose."
)


def interpret(pictures, question, context):
    content = []
    for picture in pictures:
        content.append({"type": "text", "text": picture.caption})
        data = base64.b64encode(picture.data).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{picture.media_type};base64,{data}"}})
    content.append({"type": "text", "text": f"{context}\n\n{question}" if context else question})
    asked = {
        "model": MODEL, "max_tokens": MAX_ANSWER_TOKENS,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(asked).encode("utf8"), headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=CALL_S) as answer:
            said = json.loads(answer.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"the model server answered HTTP {e.code}") from None
    return said["choices"][0]["message"]["content"]


def load():
    """Starts the model server when told how, and waits until it has the model in memory."""
    command = os.environ.get("CHAT_COMMAND")
    server = subprocess.Popen(shlex.split(command)) if command else None
    deadline = time.monotonic() + START_S
    while time.monotonic() < deadline:
        if server and server.poll() is not None:
            raise RuntimeError(f"the model server stopped with code {server.returncode}")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5):
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError("the model server did not come up in time")


if __name__ == "__main__":
    serve(interpret=interpret, load=load)
