from __future__ import annotations

import unittest

from qwen_caption_validate.caption_refiner_specialist_facts import (
    format_head_gaze_facts,
    render_specialist_prompt,
)


class CaptionRefinerDeltaV13Tests(unittest.TestCase):
    def _record(
        self,
        *,
        yaw_authority="corroborated_direction",
        pitch_authority="reduced",
        head_horizontal="frame_right",
        head_vertical="down",
        strength="profile",
        gaze_available=True,
        gaze_publishable=True,
        gaze_authority="high",
        gaze_horizontal="frame_right",
        gaze_vertical="center",
        camera="off_camera",
    ):
        return {
            "schema_version": "head-gaze-evidence-0.3",
            "status": "ok",
            "head": {
                "available": True,
                "authority": "reduced",
                "frame_horizontal": head_horizontal,
                "vertical": head_vertical,
                "yaw_strength": strength,
                "axis_authority": {
                    "yaw": {"authority": yaw_authority},
                    "pitch": {"authority": pitch_authority},
                },
            },
            "gaze_observability": {"eligible": gaze_available},
            "gaze": {
                "available": gaze_available,
                "publishable": gaze_publishable,
                "authority": gaze_authority,
                "horizontal": gaze_horizontal,
                "vertical": gaze_vertical,
                "camera_relationship": camera,
            },
        }

    def test_profile_can_publish_reliable_yaw_without_reduced_pitch(self):
        text = format_head_gaze_facts(self._record())
        self.assertIn("near-profile toward image-right", text)
        self.assertIn("Head pitch is not reliable enough for correction", text)
        self.assertNotIn("Reliable head pitch: tilted downward", text)
        self.assertIn("Publishable gaze: toward image-right and vertically near center", text)
        self.assertIn("gaze is clearly off-camera", text)

    def test_unavailable_gaze_cannot_drive_correction(self):
        record = self._record(gaze_available=False, gaze_publishable=False, gaze_authority="unavailable")
        text = format_head_gaze_facts(record)
        self.assertIn("Gaze is unavailable/non-publishable", text)
        self.assertIn("do not add, remove, or correct gaze", text)
        self.assertNotIn("Reliable camera relationship", text)

    def test_uncertain_camera_relationship_keeps_direction_but_not_camera_claim(self):
        record = self._record(
            yaw_authority="corroborated",
            pitch_authority="corroborated",
            strength="turned",
            gaze_horizontal="frame_left",
            gaze_vertical="down",
            camera="uncertain",
        )
        text = format_head_gaze_facts(record)
        self.assertIn("toward image-left and downward", text)
        self.assertIn("Camera relationship is uncertain", text)
        self.assertNotIn("gaze is clearly off-camera", text)
        self.assertNotIn("gaze is broadly toward the camera", text)

    def test_reduced_head_yaw_is_not_published_as_reliable(self):
        text = format_head_gaze_facts(self._record(yaw_authority="reduced"))
        self.assertIn("Head yaw is not reliable enough for correction", text)
        self.assertNotIn("Reliable head yaw", text)

    def test_prompt_replaces_all_specialist_placeholders(self):
        template = "CAP={{CURRENT_CAPTION}}\nLAT={{LATERALITY_FACTS}}\nSPEC={{HEAD_GAZE_FACTS}}"
        rendered = render_specialist_prompt(template, "caption", "laterality", "specialists")
        self.assertEqual(rendered, "CAP=caption\nLAT=laterality\nSPEC=specialists")
        self.assertNotIn("{{", rendered)


if __name__ == "__main__":
    unittest.main()
