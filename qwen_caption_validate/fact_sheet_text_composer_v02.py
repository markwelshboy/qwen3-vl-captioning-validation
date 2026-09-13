from __future__ import annotations

from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v01 as base

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v02.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.2"
SCHEMA_VERSION = "fact-sheet-text-composer-0.2"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.2.2"


def _compact_orientation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    band = base._clean(value.get("orientation_band"))
    yaw = value.get("yaw_magnitude_deg")
    approx = value.get("approx_yaw_deg")
    out: dict[str, Any] = {}
    if band:
        out["camera_orientation"] = band
    if isinstance(yaw, (int, float)):
        out["yaw_magnitude_deg"] = round(float(yaw), 1)
    if isinstance(approx, (int, float)):
        out["approx_yaw_deg"] = int(approx)
    return out or None


def _torso_fact(body: dict[str, Any]) -> dict[str, Any] | None:
    torso = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}
    if not torso.get("available") or not torso.get("composer_eligible"):
        return None

    summary = torso.get("caption_orientation") if isinstance(torso.get("caption_orientation"), dict) else {}
    root = _compact_orientation(torso.get("body_root_orientation"))
    upper = _compact_orientation(torso.get("upper_torso_orientation"))
    mode = base._clean(summary.get("mode"))

    if upper:
        preferred = {
            "camera_orientation": upper.get("camera_orientation"),
            "yaw_magnitude_deg": upper.get("yaw_magnitude_deg"),
            "approx_yaw_deg": upper.get("approx_yaw_deg"),
        }
        preferred = {k: v for k, v in preferred.items() if v is not None}
        if mode == "articulated" and root:
            twist = summary.get("relative_twist_magnitude_deg")
            out: dict[str, Any] = {
                "mode": "articulated",
                "body_root": root,
                "upper_torso": upper,
                "preferred": preferred,
            }
            if isinstance(twist, (int, float)):
                out["relative_twist_magnitude_deg"] = round(float(twist), 1)
            return out
        return {
            "mode": mode or "combined",
            "preferred": preferred,
            "body_root": root,
            "upper_torso": upper,
        }

    # Compatibility fallback for fact sheets without the 4B.2 enrichment.
    orientation = base._clean(torso.get("torso_camera_orientation"))
    if not orientation:
        return None
    out = {"mode": "legacy", "preferred": {"camera_orientation": orientation}}
    yaw = torso.get("torso_yaw_magnitude_deg")
    if isinstance(yaw, (int, float)):
        out["preferred"]["yaw_magnitude_deg"] = round(abs(float(yaw)), 1)
        out["preferred"]["approx_yaw_deg"] = int(5 * round(abs(float(yaw)) / 5.0))
    return out


def main() -> int:
    # Reuse the v0.1 generation/audit CLI, changing only the fact-sheet
    # contract, torso evidence projection, prompt, and output namespace.
    base._torso_fact = _torso_fact
    base.DEFAULT_PROMPT = DEFAULT_PROMPT
    base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
