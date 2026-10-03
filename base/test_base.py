"""Tests of the base agents, as far as they go without a model: the bridge in
front of a server that speaks the click protocol (a stand-in for SLIP's), the
geometry the agents reading NIfTI hand to their model, the manifest of the
MONAI agent made from a bundle's metadata, and the open model's agent in
front of a stand-in for vLLM.

    python -m unittest discover -s base     (from apps/demo/deploy/agents)
"""
import base64
import gzip
import http.server
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "..", "kit")
sys.path[:0] = [KIT, HERE]

from geometry import extent_mm, ras_affine, refusal  # noqa: E402 — after the path that finds it
from palette import colour  # noqa: E402
from seg_agent_kit import Mask, Volume  # noqa: E402
from test_kit import TOKEN, Agent, form, volume_npy  # noqa: E402


def module(path):
    """A base agent's module by its file: two agents may name theirs alike."""
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class ClickServer(http.server.BaseHTTPRequestHandler):
    """What SLIP's server does, without the model: it keeps the shape of the
    image it was sent, answers a point with a mask of that shape holding the
    voxel clicked, and refuses the prompts SLIP does not take."""

    calls = []
    shape = None

    def log_message(self, format, *args):  # noqa: A002 — the base class's signature
        pass

    def _send(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — the base class's name
        self._send(200, b'{"status": "ok"}')

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["content-length"]))
        ClickServer.calls.append(self.path)
        if self.path == "/upload_image":
            array = Volume.from_npy(body[body.index(b"\r\n\r\n") + 4:body.rindex(b"\r\n--")])
            if array.shape[0] > 100:
                return self._send(422, json.dumps({"error": "too many slices for this model"}).encode())
            ClickServer.shape = array.shape
            return self._send(200, b'{"status": "ok"}')
        if self.path == "/add_point_interaction":
            mask = Mask.empty(ClickServer.shape)
            mask.set(*json.loads(body)["voxel_coord"])
            return self._send(200, mask.encoded(), "application/octet-stream")
        if self.path == "/upload_segment":
            return self._send(200, b'{"status": "ok"}')
        return self._send(501, json.dumps({"error": "SLIP supports point prompts only"}).encode())


class Bridge(Agent):
    """The bridge agent as a process, in front of `upstream`."""

    def __init__(self, upstream):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        environment = {
            **os.environ, "PORT": str(self.port), "PYTHONPATH": KIT, "AGENT_TOKEN": TOKEN, "BRIDGE_URL": upstream,
            "AGENT_MANIFEST": os.path.join(HERE, "slip", "manifest.json"),
        }
        self.process = subprocess.Popen([sys.executable, "-u", "agent.py"], cwd=os.path.join(HERE, "bridge"), env=environment,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if self.call("GET", "/ping", token=None)[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.05)
        raise RuntimeError("the bridge did not become ready")


class BridgeTest(unittest.TestCase):
    def setUp(self):
        ClickServer.calls, ClickServer.shape = [], None
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ClickServer)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.bridge = Bridge(f"http://127.0.0.1:{self.server.server_address[1]}")
        self.addCleanup(self.bridge.stop)

    def test_the_agent_serves_the_manifest_of_slip_and_passes_the_image_and_the_clicks_on(self):
        status, body = self.bridge.call("GET", "/manifest")
        self.assertEqual([status, json.loads(body)["name"], json.loads(body)["runtime"]["maxSessions"]], [200, "slip", 1])

        self.assertEqual(self.bridge.upload((4, 6, 8))[0], 200)
        self.assertEqual(ClickServer.shape, (4, 6, 8))
        mask = self.bridge.click((1, 2, 3))
        self.assertEqual(mask.shape, (4, 6, 8))
        self.assertEqual([i for i, v in enumerate(mask.data) if v], [(1 * 6 + 2) * 8 + 3])
        self.assertEqual(ClickServer.calls, ["/upload_image", "/add_point_interaction"])

    def test_what_the_server_refuses_is_refused_in_its_words_and_the_agent_goes_on(self):
        # An image the model does not take is refused as an image, with the server's reason.
        status, body = self.bridge.upload((101, 2, 2))
        self.assertEqual([status, json.loads(body)], [422, {"error": "image-refused", "message": "too many slices for this model"}])
        self.bridge.upload((4, 6, 8))
        # A prompt SLIP does not take fails that call, in the server's words, and nothing else.
        box = json.dumps({"outer_point_one": [0, 0, 0], "outer_point_two": [1, 1, 1], "positive_click": True})
        status, body = self.bridge.call("POST", "/add_bbox_interaction", box, "application/json")
        self.assertEqual([status, json.loads(body)], [500, {"error": "agent-error", "message": "RuntimeError: SLIP supports point prompts only"}])
        self.assertEqual(self.bridge.click((0, 0, 0)).shape, (4, 6, 8))
        # A new object: the seed goes to the server as a mask of the image.
        seed = gzip.decompress(Mask.empty((4, 6, 8)).encoded())
        self.assertEqual(self.bridge.call("POST", "/upload_segment", *form(seed))[0], 200)
        self.assertEqual(ClickServer.calls[-1], "/upload_segment")

    def test_an_agent_whose_server_never_comes_up_stops_instead_of_starting_for_ever(self):
        environment = {**os.environ, "PORT": "0", "PYTHONPATH": KIT, "AGENT_TOKEN": TOKEN, "BRIDGE_URL": "http://127.0.0.1:9",
                       "BRIDGE_START_S": "1", "AGENT_MANIFEST": os.path.join(HERE, "slip", "manifest.json")}
        process = subprocess.Popen([sys.executable, "-u", "agent.py"], cwd=os.path.join(HERE, "bridge"), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.addCleanup(process.kill)
        self.assertEqual(process.wait(timeout=20), 1)
        self.assertIn(b"agent could not load: RuntimeError: the model server did not come up in time", process.stdout.read())


class ChatServer(http.server.BaseHTTPRequestHandler):
    """What vLLM does, without the model: it is healthy, and answers a chat
    completion with what it was asked, so that the request can be read."""

    asked = []

    def log_message(self, format, *args):  # noqa: A002 — the base class's signature
        pass

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — the base class's name
        self._send(200 if self.path == "/health" else 404, b"{}")

    def do_POST(self):  # noqa: N802
        asked = json.loads(self.rfile.read(int(self.headers["content-length"])))
        ChatServer.asked.append((self.path, asked))
        if "overloaded" in json.dumps(asked):
            return self._send(503, b"{}")
        self._send(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": "A bright ellipsoid."}}]}).encode())


class OpenModelTest(unittest.TestCase):
    """The Lingshu agent in front of a stand-in for vLLM."""

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ChatServer)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.agent = Agent(os.path.join("..", "base", "lingshu"), env={
            "CHAT_URL": f"http://127.0.0.1:{cls.server.server_address[1]}", "CHAT_MODEL": "lingshu-medical-mllm/Lingshu-7B",
        })

    @classmethod
    def tearDownClass(cls):
        cls.agent.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def ask(self, question):
        picture = base64.b64encode(b"a rendered slice").decode()
        asked = {"images": [{"mediaType": "image/jpeg", "data": picture, "caption": "Axial slice 16 of 32"}], "question": question, "context": "MR"}
        return self.agent.call("POST", "/interpret", json.dumps(asked), "application/json")

    def test_a_question_goes_to_the_model_as_a_chat_with_its_pictures_and_comes_back_as_text(self):
        status, body = self.ask("What is this?")
        self.assertEqual([status, json.loads(body)], [200, {"text": "A bright ellipsoid."}])
        path, asked = ChatServer.asked[-1]
        self.assertEqual([path, asked["model"], [m["role"] for m in asked["messages"]]], ["/v1/chat/completions", "lingshu-medical-mllm/Lingshu-7B", ["system", "user"]])
        # The caption, then the picture as a data URL, then what is said of the image and the question.
        content = asked["messages"][1]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url", "text"])
        self.assertEqual(content[0]["text"], "Axial slice 16 of 32")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64," + base64.b64encode(b"a rendered slice").decode())
        self.assertEqual(content[2]["text"], "MR\n\nWhat is this?")
        self.assertIn("describe, do not diagnose", asked["messages"][0]["content"])

    def test_a_model_server_that_fails_fails_that_question_in_words(self):
        status, body = self.ask("Are you overloaded?")
        self.assertEqual([status, json.loads(body)], [500, {"error": "agent-error", "message": "RuntimeError: the model server answered HTTP 503"}])
        self.assertEqual(self.ask("And now?")[0], 200)


class GeometryTest(unittest.TestCase):
    def test_the_affine_of_nifti_is_the_platforms_geometry_in_ras(self):
        # Axes along the patient's, voxels of 0.5 x 0.5 x 2 mm: left and posterior become right and anterior.
        straight = {"spacing": [0.5, 0.5, 2.0], "origin": [10.0, -20.0, 30.0], "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
        self.assertEqual(ras_affine(straight), [[-0.5, 0.0, 0.0, -10.0], [0.0, -0.5, 0.0, 20.0], [0.0, 0.0, 2.0, 30.0], [0.0, 0.0, 0.0, 1.0]])
        # A sagittal acquisition: the first voxel axis runs along the patient's y, the second along z, the third along x.
        sagittal = {"spacing": [1.0, 1.0, 3.0], "origin": [0.0, 0.0, 0.0], "direction": [0, 0, 1, 1, 0, 0, 0, 1, 0]}
        self.assertEqual(ras_affine(sagittal), [[0.0, 0.0, -3.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def test_how_far_an_image_reaches_is_its_voxels_times_their_spacing_axis_by_axis(self):
        # 300 slices of 5 mm, each 512 × 256 voxels of 0.7 × 0.9 mm.
        reach = extent_mm((300, 256, 512), {"spacing": [0.7, 0.9, 5.0]})
        self.assertEqual([round(mm, 6) for mm in reach], [1500.0, 230.4, 358.4])

    def test_a_volume_is_handed_on_as_the_npy_file_it_came_as(self):
        raw = volume_npy((2, 3, 4), values=list(range(24)))
        self.assertEqual(Volume.from_npy(raw).npy(), raw)

    def test_an_image_without_a_place_in_space_or_larger_than_a_body_is_refused_in_words(self):
        body = {"spacing": [0.8, 0.8, 1.5], "origin": [0, 0, 0], "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
        self.assertIsNone(refusal((400, 512, 512), body))
        self.assertIn("where the image is in space is needed", refusal((4, 4, 4), None))
        self.assertIn("larger than a body", refusal((400, 512, 512), {**body, "spacing": [0.8, 0.8, 10.0]}))


class ManifestTest(unittest.TestCase):
    """The MONAI agent's manifest, made from a bundle's metadata."""

    def test_the_manifest_is_made_from_the_bundles_version_and_its_table_of_channels(self):
        make = module(os.path.join(HERE, "monai", "make_manifest.py"))
        with tempfile.TemporaryDirectory() as bundle:
            os.makedirs(os.path.join(bundle, "configs"))
            channels = {"0": "background", "1": "spleen", "2": "kidney_right", "10": "lung_upper_lobe_left"}
            meta = {"version": "0.2.7", "network_data_format": {"outputs": {"pred": {"channel_def": channels}}}}
            with open(os.path.join(bundle, "configs", "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            on_gpu = make.manifest(bundle, "ghcr.io/org/seg-agent-monai-wholebody-ct:abc", "gpu")
            on_cpu = make.manifest(bundle, "ghcr.io/org/seg-agent-monai-wholebody-ct:abc", "cpu")
        self.assertEqual([on_gpu["version"], on_gpu["summary"]], ["0.2.7-lowres", "Segments 3 anatomical structures of a whole CT image by itself."])
        # The background is no label; each other channel is one, named in words, coloured apart from its neighbours.
        self.assertEqual([(l["value"], l["name"]) for l in on_gpu["returns"]["labels"]], [(1, "Spleen"), (2, "Kidney right"), (10, "Lung upper lobe left")])
        self.assertEqual(on_gpu["returns"]["labels"][0]["color"], colour(1))
        self.assertNotEqual(colour(1), colour(2))
        self.assertEqual([on_gpu["runtime"]["gpu"], on_cpu["runtime"]["gpu"]], [{"memoryGb": 16}, None])
        self.assertEqual(on_gpu["runtime"]["image"], "ghcr.io/org/seg-agent-monai-wholebody-ct:abc")


if __name__ == "__main__":
    unittest.main()
