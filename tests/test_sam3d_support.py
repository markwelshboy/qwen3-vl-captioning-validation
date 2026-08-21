from __future__ import annotations

import unittest

from qwen_caption_validate.sam3d_support import qualify_sam3d_geometry


def _analysis(shoulders: str = "visible", hips: str = "visible", embedded: object | None = None, non_target: object | None = None) -> dict:
    def item(state: str, confidence: float = 0.95) -> dict:
        return {"visibility": state, "confidence": confidence, "evidence": f"synthetic {state}"}

    return {
        "schema_version": "2.1",
        "target_subject": {
            "geometry_landmark_visibility": {
                "head": item("not_visible"),
                "left_shoulder": item(shoulders),
                "right_shoulder": item(shoulders),
                "left_hip": item(hips),
                "right_hip": item(hips),
                "left_knee": item("unknown", 0.2),
                "right_knee": item("unknown", 0.2),
                "left_ankle": item("not_visible"),
                "right_ankle": item("not_visible"),
            }
        },
        "embedded_depictions": ([embedded] if embedded is not None else []),
        "non_target_entities": ([non_target] if non_target is not None else []),
    }


def _sam3d() -> dict:
    return {
        "schema_version": "sam3d-geometry-probe-0.1",
        "bbox": {"source": "dwpose_target_keypoint_bbox"},
        "metrics": {
            "shoulder_out_of_image_plane_deg": 60.0,
            "hip_out_of_image_plane_deg": 62.0,
            "torso_depth_rotation_proxy_deg": 61.0,
            "torso_depth_tilt_deg": 4.0,
            "signed_depth_fraction_diagnostics": {"authority": "diagnostic_only_sign_not_validated"},
        },
    }


class Sam3DSupportTests(unittest.TestCase):
    def test_visible_shoulders_and_hips_qualify_unsigned_torso_depth(self) -> None:
        audit = qualify_sam3d_geometry(_analysis(), _sam3d())
        self.assertEqual(audit["shoulder_depth_rotation"]["support"]["state"], "observed_supported")
        self.assertEqual(audit["hip_depth_rotation"]["support"]["state"], "observed_supported")
        self.assertEqual(audit["torso_depth_rotation"]["authority"], "qualified_3d_geometry")
        self.assertTrue(audit["torso_depth_rotation"]["caption_usable"])
        self.assertFalse(audit["torso_depth_rotation"]["selection_usable"])
        self.assertEqual(audit["torso_axis_out_of_image_plane"]["magnitude_deg"], 4.0)

    def test_invisible_hips_make_combined_metric_partial(self) -> None:
        audit = qualify_sam3d_geometry(_analysis(hips="not_visible"), _sam3d())
        self.assertEqual(audit["shoulder_depth_rotation"]["support"]["state"], "observed_supported")
        self.assertEqual(audit["hip_depth_rotation"]["support"]["state"], "prior_reconstructed")
        self.assertEqual(audit["torso_depth_rotation"]["support_state"], "partially_supported")
        self.assertEqual(audit["torso_depth_rotation"]["authority"], "report_only_partial_image_support")
        self.assertFalse(audit["torso_depth_rotation"]["caption_usable"])

    def test_legacy_analysis_cannot_grant_sam3d_authority(self) -> None:
        legacy = {"schema_version": "2.0", "target_subject": {}, "embedded_depictions": [], "non_target_entities": []}
        audit = qualify_sam3d_geometry(legacy, _sam3d())
        self.assertFalse(audit["landmark_visibility_available"])
        self.assertEqual(
            audit["torso_depth_rotation"]["authority"],
            "report_only_requires_analyze_v2_1_visibility",
        )

    def test_human_portrait_depiction_requires_target_provenance_review(self) -> None:
        audit = qualify_sam3d_geometry(
            _analysis(embedded={"type": "framed_photo", "description": "framed portrait of another man"}),
            _sam3d(),
        )
        self.assertEqual(
            audit["torso_depth_rotation"]["authority"],
            "qualified_geometry_pending_target_provenance",
        )
        self.assertEqual(audit["target_provenance"]["context_risk"], "requires_review")

    def test_generic_media_or_object_hand_reference_does_not_trigger_provenance(self) -> None:
        analysis = _analysis(
            embedded={"description": "blue patterned poster or flag", "type": "poster"},
            non_target={"description": "white ceramic mug", "contact": "held by right hand"},
        )
        audit = qualify_sam3d_geometry(analysis, _sam3d())
        self.assertEqual(audit["target_provenance"]["context_risk"], "no_semantic_multi_subject_risk_detected")
        self.assertEqual(audit["torso_depth_rotation"]["authority"], "qualified_3d_geometry")

    def test_generic_media_located_behind_target_head_does_not_trigger_provenance(self) -> None:
        audit = qualify_sam3d_geometry(
            _analysis(non_target={"description": "decorative poster partially visible behind subject's head"}),
            _sam3d(),
        )
        self.assertEqual(audit["target_provenance"]["context_risk"], "no_semantic_multi_subject_risk_detected")
        self.assertEqual(audit["torso_depth_rotation"]["authority"], "qualified_3d_geometry")

    def test_real_depiction_still_triggers_when_located_behind_target_head(self) -> None:
        audit = qualify_sam3d_geometry(
            _analysis(non_target={"description": "portrait of another woman behind subject's head"}),
            _sam3d(),
        )
        self.assertEqual(audit["target_provenance"]["context_risk"], "requires_review")

    def test_human_form_tattoo_does_not_trigger_bbox_provenance_review(self) -> None:
        audit = qualify_sam3d_geometry(
            _analysis(embedded={"type": "tattoo", "description": "tattoo portrait of a bearded man"}),
            _sam3d(),
        )
        self.assertEqual(audit["target_provenance"]["context_risk"], "no_semantic_multi_subject_risk_detected")
        self.assertEqual(audit["torso_depth_rotation"]["authority"], "qualified_3d_geometry")


if __name__ == "__main__":
    unittest.main()
