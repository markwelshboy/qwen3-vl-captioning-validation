from __future__ import annotations


def main() -> int:
    from .semantic_v3_text_only_bootstrap import install_text_only_vllm

    install_text_only_vllm()
    from .semantic_v3_analyze import main as analyze_main

    return analyze_main()


if __name__ == "__main__":
    raise SystemExit(main())
