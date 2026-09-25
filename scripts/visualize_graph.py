"""Render the current agent graph for viewing in VS Code.

Writes to docs/:
  agent_graph.mmd  Mermaid source (preview with the "Mermaid Chart" or "Markdown Preview Mermaid" extension)
  agent_graph.md   the same diagram in a ```mermaid block (open it and press Ctrl+Shift+V)
  agent_graph.png  rendered image (opens natively in VS Code; needs internet for mermaid.ink)

Usage:
  .venv\\Scripts\\python scripts\\visualize_graph.py            # write all three files
  .venv\\Scripts\\python scripts\\visualize_graph.py --open     # ...and open the PNG/Markdown in VS Code
  .venv\\Scripts\\python scripts\\visualize_graph.py --no-png   # offline: skip the PNG
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph import build_builder  # noqa: E402
from config import settings  # noqa: E402

# Human-in-the-loop nodes are highlighted in the diagram.
INTERRUPT_NODES = ["ask_question", "post_test_and_feedback"]
# Nodes backed by real services today, as opposed to stubs.
REAL_NODES = ["crag_retrieve", "corrective_retrieve", "llm_router", "generate_response"]


def build_mermaid() -> str:
    """Return the graph's Mermaid source with interrupt and real-service nodes highlighted."""
    graph = build_builder().compile().get_graph()
    mermaid = graph.draw_mermaid()
    extra = [
        "\tclassDef interrupt fill:#ffe08a,stroke:#b8860b,stroke-width:2px",
        "\tclassDef real fill:#c8f7c5,stroke:#2e7d32",
        f"\tclass {','.join(INTERRUPT_NODES)} interrupt",
        f"\tclass {','.join(REAL_NODES)} real",
    ]
    return mermaid.rstrip() + "\n" + "\n".join(extra) + "\n"


def render_png(mermaid: str, out: Path) -> bool:
    """Render the Mermaid source to PNG via the mermaid.ink API; return False if offline."""
    from langchain_core.runnables.graph_mermaid import draw_mermaid_png

    try:
        draw_mermaid_png(mermaid, output_file_path=str(out), background_color="white")
        return True
    except Exception as exc:  # network errors, API errors
        print(f"  PNG skipped ({type(exc).__name__}: {exc})")
        return False


def main() -> None:
    """Write the diagram files and optionally open them in VS Code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--open", action="store_true", help="open the result in VS Code")
    parser.add_argument("--no-png", action="store_true", help="skip PNG rendering (offline)")
    args = parser.parse_args()

    docs = settings.docs_dir
    docs.mkdir(parents=True, exist_ok=True)
    mermaid = build_mermaid()

    mmd_path, md_path, png_path = docs / "agent_graph.mmd", docs / "agent_graph.md", docs / "agent_graph.png"
    mmd_path.write_text(mermaid, encoding="utf-8")
    md_path.write_text(
        "# Agent graph\n\n"
        "Yellow = interrupt (waits for the learner). Green = wired to real services. Dashed = conditional edge.\n"
        "Regenerate with `python scripts/visualize_graph.py`.\n\n"
        f"```mermaid\n{mermaid}```\n",
        encoding="utf-8",
    )
    print(f"Wrote {mmd_path}\nWrote {md_path}")
    png_ok = not args.no_png and render_png(mermaid, png_path)
    if png_ok:
        print(f"Wrote {png_path}")

    if args.open:
        code = shutil.which("code")
        if code is None:
            print("VS Code 'code' command not found on PATH; open the files manually.")
        else:
            subprocess.run([code, "--reuse-window", str(png_path if png_ok else md_path)], check=False)


if __name__ == "__main__":
    main()
