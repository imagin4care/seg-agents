"""Tests of the kit, through the fake agents built with it: each test starts
an agent as the platform would (its own process, PORT and AGENT_TOKEN in the
environment) and calls it over HTTP.

    python -m unittest discover -s apps/demo/deploy/agents/kit
"""
import base64
import gzip
import http.client
import json
import os
import socket
import struct
import subprocess
import sys
import time
import unittest

from seg_agent_kit import LabelMap, Mask, Volume, _Jobs, _Sessions

HERE = os.path.dirname(os.path.abspath(__file__))
FAKES = os.path.join(HERE, "..", "fakes")
TOKEN = "the-endpoint-token"


def volume_npy(shape, value=0, values=None):
    """A volume of 16-bit values, as the app uploads it: all `value`, or `values`."""
    count = shape[0] * shape[1] * shape[2]
    header = "{'descr': '<i2', 'fortran_order': False, 'shape': (%d, %d, %d), }" % tuple(shape)
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    data = struct.pack("<%dh" % count, *values) if values else struct.pack("<h", value) * count
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + data


def job_spec(parameters=None):
    """What a job is asked with, as the platform puts it in the `x-agent-job` header."""
    asked = {"geometry": {"spacing": [1, 1, 2], "origin": [0, 0, 0], "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]}, "parameters": parameters or {}}
    return {"x-agent-job": base64.urlsafe_b64encode(json.dumps(asked).encode()).decode().rstrip("=")}


def form(file, fields=None):
    """A multipart form with `file` first, like the browser's."""
    body = b'--edge\r\nContent-Disposition: form-data; name="file"; filename="image.npy"\r\n\r\n' + file
    for name, value in (fields or {}).items():
        body += b'\r\n--edge\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s' % (name.encode(), value.encode())
    return body + b"\r\n--edge--\r\n", "multipart/form-data; boundary=edge"


class Agent:
    """One fake agent running as a process."""

    def __init__(self, name, env=None, token=TOKEN):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        environment = {**os.environ, "PORT": str(self.port), "PYTHONPATH": HERE, **(env or {})}
        if token is not None:
            environment["AGENT_TOKEN"] = token
        self.process = subprocess.Popen([sys.executable, "-u", "agent.py"], cwd=os.path.join(FAKES, name), env=environment,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                self.call("GET", "/ping", token=None)
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("the agent did not start")

    def call(self, method, path, body=None, content_type=None, session: "str | None" = "s1", token: "str | None" = TOKEN, headers=None):
        """Returns (status, body). `token` and `session` None: the header is left out."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        sent = dict(headers or {})
        if token is not None:
            sent["x-agent-token"] = token
        if session is not None:
            sent["x-agent-session"] = session
        if content_type:
            sent["content-type"] = content_type
        connection.request(method, path, body=body, headers=sent)
        response = connection.getresponse()
        answer = response.status, response.read()
        connection.close()
        return answer

    def upload(self, shape, session="s1"):
        return self.call("POST", "/upload_image", *form(volume_npy(shape)), session=session)

    def click_status(self, at, positive=True, session="s1"):
        """Returns (status, body) of a click."""
        body = json.dumps({"voxel_coord": list(at), "positive_click": positive})
        return self.call("POST", "/add_point_interaction", body, "application/json", session=session)

    def click(self, at, positive=True, session="s1"):
        """The mask a click answers."""
        status, answer = self.click_status(at, positive, session)
        if status != 200:
            raise AssertionError(f"the click was refused: {status} {answer!r}")
        return Volume.from_npy(answer)

    def stop(self):
        self.process.kill()
        self.process.wait()


class KitTest(unittest.TestCase):
    def start(self, name="sphere", **options):
        agent = Agent(name, **options)
        self.addCleanup(agent.stop)
        return agent

    def test_ping_needs_no_token_and_everything_else_does(self):
        agent = self.start()
        self.assertEqual(agent.call("GET", "/ping", token=None), (200, b"ok"))
        for token in (None, "wrong", ""):
            self.assertEqual(agent.call("GET", "/manifest", token=token)[0], 401)
            self.assertEqual(agent.call("POST", "/upload_image", *form(volume_npy((2, 2, 2))), token=token)[0], 401)
        status, body = agent.call("GET", "/manifest")
        self.assertEqual((status, json.loads(body)["name"]), (200, "sphere"))

    def test_a_body_is_not_read_from_a_stranger_nor_beyond_what_the_platform_sends(self):
        agent = self.start()
        # The answer comes although the body announced never does: it was not waited for.
        for token, status in (("wrong", 401), (TOKEN, 413)):
            connection = http.client.HTTPConnection("127.0.0.1", agent.port, timeout=10)
            connection.putrequest("POST", "/upload_image")
            connection.putheader("x-agent-token", token)
            connection.putheader("x-agent-session", "s1")
            connection.putheader("content-length", str(64 * 1024 ** 2))
            connection.endheaders()
            self.assertEqual(connection.getresponse().status, status)
            connection.close()

    def test_an_agent_started_without_a_token_serves_nobody(self):
        agent = self.start(token=None)
        self.assertEqual(agent.call("GET", "/ping", token=None)[0], 200)
        for token in (None, "", "anything"):
            self.assertEqual(agent.call("GET", "/manifest", token=token)[0], 401)

    def test_a_click_answers_a_mask_of_the_shape_of_the_image(self):
        agent = self.start()
        self.assertEqual(agent.upload((20, 24, 28))[0], 200)
        mask = agent.click((10, 12, 14))
        self.assertEqual((mask.shape, mask.dtype), ((20, 24, 28), "|u1"))
        self.assertEqual(mask.data[(10 * 24 + 12) * 28 + 14], 1)
        self.assertEqual(mask.data[0], 0)
        inside = sum(mask.data)
        # A negative click takes its ball out of the object again.
        after = agent.click((10, 12, 14), positive=False)
        self.assertEqual((inside > 500, sum(after.data)), (True, 0))

    def test_the_template_an_author_starts_from_is_an_agent_that_runs(self):
        agent = self.start(os.path.join("..", "template"))
        status, body = agent.call("GET", "/manifest")
        self.assertEqual((status, json.loads(body)["name"]), (200, "my-agent"))
        # A bright block of 3 × 3 × 3 voxels in a dark image: a click in it finds all of it, and nothing else.
        shape = (6, 7, 8)
        block = {(z * 7 + y) * 8 + x for z in (2, 3, 4) for y in (2, 3, 4) for x in (3, 4, 5)}
        image = volume_npy(shape, values=[900 if i in block else 20 for i in range(6 * 7 * 8)])
        self.assertEqual(agent.call("POST", "/upload_image", *form(image))[0], 200)
        mask = agent.click((3, 3, 4))
        self.assertEqual({i for i, v in enumerate(mask.data) if v}, block)
        # A click outside the image changes nothing; a negative one in the block takes it out again.
        self.assertEqual(sum(agent.click((60, 3, 4)).data), 27)
        self.assertEqual(sum(agent.click((3, 3, 4), positive=False).data), 0)

    def test_a_box_and_a_scribble_are_prompts_too(self):
        agent = self.start()
        agent.upload((4, 5, 6))
        box = json.dumps({"outer_point_one": [0, 0, 0], "outer_point_two": [1, 2, 3]})
        status, answer = agent.call("POST", "/add_bbox_interaction", box, "application/json")
        self.assertEqual((status, sum(Volume.from_npy(answer).data)), (200, 2 * 3 * 4))
        drawn = Mask.empty((4, 5, 6))
        drawn.set(3, 4, 5)
        status, answer = agent.call("POST", "/add_scribble_interaction", *form(drawn.encoded(), {"positive_click": "true"}))
        self.assertEqual((status, sum(Volume.from_npy(answer).data)), (200, 25))
        # A seed replaces the object.
        self.assertEqual(agent.call("POST", "/upload_segment", *form(drawn.encoded()))[0], 200)
        self.assertEqual(sum(agent.click((0, 0, 0), positive=False).data), 1)
        # A mask of another shape than the image is not passed on, nor one that inflates past it.
        other = Mask.empty((4, 5, 7)).encoded()
        self.assertEqual(agent.call("POST", "/add_scribble_interaction", *form(other))[0], 400)
        bomb = gzip.compress(bytes(64 * 1024 ** 2), compresslevel=1)
        status, answer = agent.call("POST", "/add_scribble_interaction", *form(bomb))
        self.assertEqual((status, json.loads(answer)["message"]), (400, "the array is too large"))

    def test_a_prompt_before_the_image_is_refused(self):
        agent = self.start()
        status, body = agent.click_status((0, 0, 0))
        self.assertEqual((status, json.loads(body)["error"]), (409, "no-session"))
        self.assertEqual(agent.call("POST", "/upload_image", b"not a form", "text/plain")[0], 400)
        self.assertEqual(agent.call("POST", "/upload_image", *form(b"not an array"))[0], 400)
        self.assertEqual(agent.call("POST", "/nope")[0], 409)
        self.assertEqual(agent.upload((2, 2, 2), session=None)[0], 400)

    def test_sessions_are_kept_apart(self):
        agent = self.start()  # its manifest keeps four sessions apart
        shapes = {"a": (4, 4, 4), "b": (5, 6, 7), "c": (2, 3, 4), "d": (3, 3, 3)}
        for session, shape in shapes.items():
            self.assertEqual(agent.upload(shape, session=session)[0], 200)
        for session, shape in shapes.items():
            self.assertEqual(agent.click((1, 1, 1), session=session).shape, shape)
        # A session that is closed is gone; a known session may send another image.
        self.assertEqual(agent.call("DELETE", "/session", session="a")[0], 200)
        self.assertEqual(agent.click_status((1, 1, 1), session="a")[0], 409)
        self.assertEqual(agent.upload((9, 9, 9), session="b")[0], 200)
        self.assertEqual(agent.click((1, 1, 1), session="b").shape, (9, 9, 9))

    def test_a_session_too_many_takes_the_place_of_the_one_used_longest_ago(self):
        closed = []

        class Session:
            def __init__(self, image):
                self.shape = image.shape

            def point(self, at, positive):
                return Mask.empty(self.shape)

            def close(self):
                closed.append(self.shape)

        agent = _Sessions(Session, 2)
        click = json.dumps({"voxel_coord": [0, 0, 0]})

        def call(session, path, shape=None):
            body, content_type = form(volume_npy(shape)) if shape else (click, "application/json")
            return agent.call("POST", path, session, content_type, body)[0]

        self.assertEqual([call("a", "/upload_image", (1, 1, 1)), call("b", "/upload_image", (2, 2, 2))], [200, 200])
        self.assertEqual(call("a", "/add_point_interaction"), 200)  # "b" is now the one used longest ago
        self.assertEqual((call("c", "/upload_image", (3, 3, 3)), closed), (200, [(2, 2, 2)]))
        self.assertEqual(sorted(agent.sessions), ["a", "c"])
        # Another image for a session closes the object of the first.
        self.assertEqual((call("a", "/upload_image", (4, 4, 4)), closed), (200, [(2, 2, 2), (1, 1, 1)]))
        self.assertEqual(agent.call("DELETE", "/session", "c", "", b"")[0], 200)
        self.assertEqual(closed[-1], (3, 3, 3))

    def test_an_upload_in_pieces_is_replayed_as_one(self):
        agent = self.start()
        body, content_type = form(volume_npy((10, 20, 30)))
        pieces = [body[i:i + 5000] for i in range(0, len(body), 5000)]
        self.assertGreater(len(pieces), 2)
        for seq, piece in enumerate(pieces):
            self.assertEqual(agent.call("PUT", f"/_relay/upload-0001/{seq}", piece, "application/octet-stream")[0], 200)
        # Out of order, or to a path of the relay itself: refused.
        self.assertEqual(agent.call("PUT", "/_relay/upload-0001/7", b"x")[0], 409)
        relay = {"x-relay-path": "/_relay/x/0", "x-relay-content-type": content_type}
        self.assertEqual(agent.call("POST", "/_relay/upload-0002/commit", headers=relay)[0], 400)
        relay["x-relay-path"] = "/upload_image"
        self.assertEqual(agent.call("POST", "/_relay/upload-0001/commit", headers=relay)[0], 200)
        self.assertEqual(agent.click((9, 19, 29)).shape, (10, 20, 30))
        # The pieces are gone once replayed.
        self.assertEqual(agent.call("POST", "/_relay/upload-0001/commit", headers=relay)[0], 400)
        # An upload given up is committed to a path of the relay: its pieces are dropped.
        self.assertEqual(agent.call("PUT", "/_relay/upload-0003/0", b"x", "application/octet-stream")[0], 200)
        self.assertEqual(agent.call("POST", "/_relay/upload-0003/commit", headers={"x-relay-path": "/_relay/refused"})[0], 400)
        self.assertEqual(agent.call("PUT", "/_relay/upload-0003/1", b"x", "application/octet-stream")[0], 409)

    def follow(self, agent, job_id, until):
        """Ask how far the job is until it is in one of the states `until`; returns what it said."""
        deadline = time.time() + 10
        while time.time() < deadline:
            said = json.loads(agent.call("GET", f"/jobs/{job_id}")[1])
            if said["state"] in until:
                return said
            time.sleep(0.05)
        raise AssertionError(f"the job is still {said['state']}")

    def test_a_job_is_started_followed_and_gives_a_label_map_of_the_image(self):
        agent = self.start("threshold", env={"THRESHOLD_SECONDS": "0.6"})
        self.assertEqual(json.loads(agent.call("GET", "/manifest")[1])["contract"]["kind"], "job")
        # Six voxels from dark to bright: the upper third, the middle third, and the rest.
        image = volume_npy((1, 2, 3), values=[0, 10, 40, 50, 90, 100])
        status, started = agent.call("POST", "/jobs", *form(image), headers=job_spec())
        self.assertEqual(status, 202)
        job_id = json.loads(started)["id"]
        self.assertEqual(agent.call("GET", f"/jobs/{job_id}/result")[0], 409)
        said = self.follow(agent, job_id, ("succeeded",))
        self.assertEqual((said["progress"], said["message"]), (1.0, "slice 1 of 1"))
        status, result = agent.call("GET", f"/jobs/{job_id}/result")
        labels = Volume.from_npy(result)
        self.assertEqual((status, labels.shape, labels.dtype, list(labels.data)), (200, (1, 2, 3), "|u1", [0, 0, 1, 1, 2, 2]))
        # A label map is given once.
        self.assertEqual(agent.call("GET", f"/jobs/{job_id}/result")[0], 404)

        # The parameters reach the agent.
        job_id = json.loads(agent.call("POST", "/jobs", *form(image), headers=job_spec({"brightOnly": True}))[1])["id"]
        self.follow(agent, job_id, ("succeeded",))
        self.assertEqual(list(Volume.from_npy(agent.call("GET", f"/jobs/{job_id}/result")[1]).data), [0, 0, 0, 0, 2, 2])

    def test_a_job_that_fails_says_why_and_one_that_is_cancelled_gives_nothing(self):
        agent = self.start("threshold", env={"THRESHOLD_SECONDS": "3"})
        failing = json.loads(agent.call("POST", "/jobs", *form(volume_npy((2, 2, 2))), headers=job_spec({"fail": True}))[1])["id"]
        said = self.follow(agent, failing, ("failed",))
        self.assertEqual(said["message"], "RuntimeError: asked to fail, to show what a failure looks like")

        slow = json.loads(agent.call("POST", "/jobs", *form(volume_npy((4, 2, 2))), headers=job_spec())[1])["id"]
        self.follow(agent, slow, ("running",))
        self.assertEqual(agent.call("POST", f"/jobs/{slow}/cancel")[0], 200)
        self.follow(agent, slow, ("cancelled",))
        self.assertEqual(agent.call("GET", f"/jobs/{slow}/result")[0], 409)

        self.assertEqual(agent.call("POST", "/jobs", *form(b"not an array"), headers=job_spec())[0], 400)
        self.assertEqual(agent.call("GET", "/jobs/nope")[0], 404)
        self.assertEqual(agent.call("POST", "/jobs", *form(volume_npy((2, 2, 2))), token="wrong")[0], 401)
        # An automatic agent has no session, and an interactive one no job.
        self.assertEqual(agent.call("POST", "/upload_image", *form(volume_npy((2, 2, 2))))[0], 404)
        self.assertEqual(self.start("sphere").call("POST", "/jobs", *form(volume_npy((2, 2, 2))))[0], 404)

    def test_a_job_too_many_takes_the_place_of_the_oldest_which_is_told_to_stop(self):
        started = []

        def never_done(image, job):
            started.append(job)
            while not job.cancelled():
                time.sleep(0.01)
            return LabelMap.empty(image.shape)

        jobs = _Jobs(never_done, 2)
        ids = [json.loads(jobs.call("POST", "/jobs", *form(volume_npy((1, 1, 1)))[::-1], "")[2])["id"] for _ in range(3)]
        deadline = time.time() + 5
        while len(started) < 3 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(sorted(jobs.jobs), sorted(ids[1:]))
        self.assertEqual([job.cancelled() for job in started], [True, False, False])
        for job_id in ids[1:]:
            jobs.call("POST", f"/jobs/{job_id}/cancel", "", b"", "")

    def test_a_job_that_is_over_gives_its_place_before_one_that_runs(self):
        started = []

        def work(image, job):
            started.append(job)
            if job.parameters.get("fail"):
                raise RuntimeError("no")
            while not job.cancelled():
                time.sleep(0.01)
            return LabelMap.empty(image.shape)

        jobs = _Jobs(work, 2)
        failing = base64.urlsafe_b64encode(json.dumps({"parameters": {"fail": True}}).encode()).decode()
        start = lambda spec="": json.loads(jobs.call("POST", "/jobs", *form(volume_npy((1, 1, 1)))[::-1], spec)[2])["id"]
        running, failed = start(), start(failing)
        deadline = time.time() + 5
        while (len(started) < 2 or started[1].state != "failed") and time.time() < deadline:
            time.sleep(0.01)
        # The table is full: the one that failed leaves, the one that runs goes on.
        third = start()
        self.assertEqual(sorted(jobs.jobs), sorted([running, third]))
        self.assertFalse(started[0].cancelled())
        for job_id in (running, third):
            jobs.call("POST", f"/jobs/{job_id}/cancel", "", b"", "")

    def test_what_a_job_says_of_itself_is_cut_to_one_short_line(self):
        def chatty(image, job):
            job.progress(0.5, "x" * 5000)
            raise RuntimeError("y" * 5000)

        jobs = _Jobs(chatty, 1)
        job_id = json.loads(jobs.call("POST", "/jobs", *form(volume_npy((1, 1, 1)))[::-1], "")[2])["id"]
        deadline = time.time() + 5
        while jobs.jobs[job_id].state != "failed" and time.time() < deadline:
            time.sleep(0.01)
        said = jobs.call("GET", f"/jobs/{job_id}", "", b"", "")[2]
        self.assertLess(len(said), 400)
        self.assertEqual(json.loads(said)["message"], ("RuntimeError: " + "y" * 5000)[:200])

    def test_an_agent_that_answers_in_words_is_given_the_pictures_and_the_question(self):
        agent = self.start("reader")
        picture = base64.b64encode(b"a rendered slice").decode()
        asked = {"images": [{"mediaType": "image/jpeg", "data": picture, "caption": "Axial slice 44 of 88"}], "question": "What is this?", "context": "MR"}
        status, body = agent.call("POST", "/interpret", json.dumps(asked), "application/json")
        self.assertEqual(status, 200)
        said = json.loads(body)["text"]
        self.assertIn("I was shown 1 rendered slice (Axial slice 44 of 88).", said)
        self.assertIn("You asked: What is this?", said)
        # What is not a question is refused; an agent that serves no such contract does not know the route.
        self.assertEqual(agent.call("POST", "/interpret", json.dumps({"images": []}), "application/json")[0], 400)
        self.assertEqual(agent.call("POST", "/interpret", json.dumps({**asked, "images": [{"mediaType": "image/png", "data": "not base64!"}]}), "application/json")[0], 400)
        self.assertEqual(agent.call("POST", "/interpret", json.dumps(asked), "application/json", token="wrong")[0], 401)
        self.assertEqual(self.start().call("POST", "/interpret", json.dumps(asked), "application/json")[0], 404)

    def test_the_broken_agent_fails_the_way_it_is_told(self):
        slow = self.start("broken", env={"BROKEN_MODE": "slow-start"})
        self.assertEqual(slow.call("GET", "/ping", token=None), (204, b""))
        wrong = self.start("broken", env={"BROKEN_MODE": "wrong-shape"})
        wrong.upload((4, 6, 8))
        self.assertEqual(wrong.click((1, 1, 1)).shape, (5, 7, 9))
        dying = self.start("broken", env={"BROKEN_MODE": "crash"})
        dying.upload((4, 6, 8))
        with self.assertRaises((OSError, http.client.HTTPException)):
            dying.click_status((1, 1, 1))
        self.assertIsNotNone(dying.process.wait(timeout=5))
        # Whatever ends the loading, the agent stops: it is not left starting for ever.
        never = subprocess.Popen([sys.executable, "-u", "agent.py"], cwd=os.path.join(FAKES, "broken"),
                                 env={**os.environ, "PORT": "0", "PYTHONPATH": HERE, "AGENT_TOKEN": TOKEN, "BROKEN_MODE": "no-start"},
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(never.kill)
        self.assertEqual(never.wait(timeout=10), 1)


class EncodingTest(unittest.TestCase):
    def test_a_mask_is_written_as_numpy_reads_it(self):
        mask = Mask.empty((2, 3, 4))
        mask.set(1, 2, 3)
        mask.set(9, 9, 9)  # outside: ignored
        raw = gzip.decompress(mask.encoded())
        self.assertEqual(raw[:6], b"\x93NUMPY")
        self.assertEqual(len(raw) % 64, 24 % 64)  # a header padded to 64 bytes, then 24 voxels
        back = Volume.from_npy(mask.encoded())
        self.assertEqual((back.shape, back.dtype, back.data[-1], sum(back.data)), ((2, 3, 4), "|u1", 1, 1))

    def test_a_label_map_is_written_in_one_or_two_bytes_per_voxel(self):
        small = LabelMap.empty((1, 2, 2))
        small.set(0, 1, 1, 7)
        back = Volume.from_npy(small.encoded())
        self.assertEqual((back.shape, back.dtype, list(back.values())), ((1, 2, 2), "|u1", [0, 0, 0, 7]))
        wide = LabelMap.empty((1, 2, 2), bits=16)
        wide.set(0, 0, 1, 300)
        wide.set(5, 5, 5, 9)  # outside: ignored
        back = Volume.from_npy(wide.encoded())
        self.assertEqual((back.dtype, list(back.values())), ("<u2", [0, 300, 0, 0]))

    def test_the_values_of_a_volume_are_read_as_numbers(self):
        image = Volume.from_npy(volume_npy((1, 1, 3), values=[-5, 0, 1200]))
        self.assertEqual(list(image.values()), [-5, 0, 1200])
        with self.assertRaisesRegex(ValueError, "cannot be read as numbers"):
            Volume((1,), "<c8", b"").values()

    def test_an_array_that_is_not_what_it_says_is_refused(self):
        good = volume_npy((2, 2, 2))
        for bad in (good[:-1], b"PK\x03\x04", good.replace(b"False", b"True "), gzip.compress(good)[:-12]):
            with self.assertRaises(ValueError):
                Volume.from_npy(bad)
        self.assertEqual(Volume.from_npy(gzip.compress(good)).shape, (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "too large"):
            Volume.from_npy(gzip.compress(good), limit=len(good) - 1)

    def test_a_part_is_named_by_its_name_not_its_file_name(self):
        from seg_agent_kit import _multipart
        body = b'--edge\r\nContent-Disposition: form-data; filename="x"; name="file"\r\n\r\nabc\r\n--edge--\r\n'
        self.assertEqual(_multipart("multipart/form-data; boundary=edge", body), {"file": b"abc"})


if __name__ == "__main__":
    unittest.main()
