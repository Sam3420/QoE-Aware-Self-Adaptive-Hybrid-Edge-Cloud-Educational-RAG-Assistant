"""Extract an NCERT chapter PDF to local text with page markers (content prep; the output is NOT committed).

  .venv\\Scripts\\python scripts\\extract_chapter_text.py <pdf> <topic_id> --first-page 79

Writes knowledge_prep/packages/<topic_id>/chapter.txt (gitignored: the textbook says "not to be republished"), one
block per page, each starting with a marker line ``[p.<printed page>]``. Prints the PDF's SHA-256 for the manifest.
Needs pypdf (requirements.txt, dev section). Pages whose drawings exceed pypdf's default parsing budget are retried
with a larger budget for this run only.
"""
import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pypdf  # noqa: E402
from pypdf._configuration import CURRENT_CONFIGURATION, get_configuration  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402

FOOTER = re.compile(r"^\s*Reprint \d{4}-\d{2}\s*$")


def clean(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if not FOOTER.match(line)]
    return "\n".join(lines).strip()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.disable(logging.WARNING)  # pypdf's font-encoding notices
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("topic_id")
    parser.add_argument("--first-page", type=int, required=True, help="printed page number of the PDF's first page")
    args = parser.parse_args()
    data = args.pdf.read_bytes()
    reader = pypdf.PdfReader(str(args.pdf))
    blocks, retried = [], []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except pypdf.errors.LimitReachedError:
            cfg = get_configuration()
            token = CURRENT_CONFIGURATION.set(cfg.with_overwrites(  # large figures: more parsing and inflating room
                stream_parsing_work_maximum_count=cfg.stream_parsing_work_maximum_count * 20,
                zlib_maximum_output_length=cfg.zlib_maximum_output_length * 20))
            try:
                text = page.extract_text() or ""
            finally:
                CURRENT_CONFIGURATION.reset(token)
            retried.append(args.first_page + i)
        blocks.append(f"[p.{args.first_page + i}]\n{clean(text)}")
    out = PROJECT_ROOT / "knowledge_prep" / "packages" / args.topic_id / "chapter.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"{args.pdf.name}: {len(reader.pages)} pages (p.{args.first_page}-{args.first_page + len(reader.pages) - 1}),"
          f" sha256 {hashlib.sha256(data).hexdigest()}")
    print(f"retried with a larger parsing budget: {retried or 'none'}")
    print(f"wrote {out} ({out.stat().st_size} bytes, gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
