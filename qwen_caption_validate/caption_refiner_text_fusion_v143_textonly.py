from __future__ import annotations

from .caption_refiner_text_fusion_v14_textonly import _install_text_only_vllm


def main() -> int:
    _install_text_only_vllm()
    from . import caption_refiner_text_fusion_v143 as impl

    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
