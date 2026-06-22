"""Tests for the vision hook plumbing (roadmap 1.5). Pure functions only — no
network; the actual VLM is a machine-side server."""
import os
os.environ.setdefault("ERIS_GPU", "0")

import base64
import tempfile
import unittest

from eris.interface.vision import encode_image, build_vision_messages, is_configured


class TestVisionPlumbing(unittest.TestCase):
    def _tiny_png(self) -> str:
        # 1x1 PNG
        data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.write(data)
        f.close()
        return f.name

    def test_encode_image_is_data_url(self):
        url = encode_image(self._tiny_png())
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertGreater(len(url), len("data:image/png;base64,"))

    def test_build_messages_structure(self):
        msgs = build_vision_messages("what is this?", [self._tiny_png()],
                                     system="be terse")
        self.assertEqual(msgs[0]["role"], "system")
        user = msgs[1]
        self.assertEqual(user["role"], "user")
        kinds = [part["type"] for part in user["content"]]
        self.assertEqual(kinds[0], "text")
        self.assertIn("image_url", kinds)
        self.assertEqual(user["content"][0]["text"], "what is this?")

    def test_multiple_images(self):
        imgs = [self._tiny_png(), self._tiny_png()]
        msgs = build_vision_messages("compare", imgs)
        img_parts = [p for p in msgs[0]["content"] if p["type"] == "image_url"]
        self.assertEqual(len(img_parts), 2)

    def test_is_configured_reads_env(self):
        old = os.environ.pop("ERIS_VISION_BASE_URL", None)
        try:
            self.assertFalse(is_configured())
            os.environ["ERIS_VISION_BASE_URL"] = "http://localhost:8000/v1"
            self.assertTrue(is_configured())
        finally:
            os.environ.pop("ERIS_VISION_BASE_URL", None)
            if old is not None:
                os.environ["ERIS_VISION_BASE_URL"] = old


if __name__ == "__main__":
    unittest.main()
