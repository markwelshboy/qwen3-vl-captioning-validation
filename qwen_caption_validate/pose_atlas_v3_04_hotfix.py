from __future__ import annotations

"""Compatibility hotfix for Pose Atlas v0.4 mesh resolution.

v0.4 calls ``v03._resolve_mesh`` but the helper actually lives in the v0.2
module, which v0.3 imports as ``v03.v02``.  Patch the alias before delegating to
v0.4 so the atlas can resolve retained OBJ meshes without changing any of the
hand/relational-pose logic.
"""

from . import pose_atlas_v3_04 as v04


v04.v03._resolve_mesh = v04.v03.v02._resolve_mesh


def main() -> int:
    return v04.main()


if __name__ == "__main__":
    raise SystemExit(main())
