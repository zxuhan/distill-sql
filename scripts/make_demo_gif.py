"""Generate ``assets/demo.gif``: animated terminal-style card showing
three real predictions from the deployed 1.5B q4 student.

The SQL outputs are pulled from ``reports/predictions/distilled_1p5b_q4.jsonl``
(actual model predictions that passed exec-match against gold on Spider
dev). The rendering is synthesized; the data is real.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Output config.
WIDTH = 760
HEIGHT = 460
OUT_PATH = Path("assets/demo.gif")

# Colors: dark navy terminal aesthetic.
BG = (15, 23, 42)
CARD = (30, 41, 59)
ACCENT = (59, 130, 246)
GREEN = (34, 197, 94)
TEXT = (226, 232, 240)
DIM = (148, 163, 184)
SQL_BG = (8, 14, 28)
SQL_COLOR = (147, 197, 253)
KEYWORD = (244, 114, 182)  # pink, for SQL keywords

# Real predictions from the deployed model. SQL strings are verbatim from
# reports/predictions/distilled_1p5b_q4.jsonl; the questions come from
# data/spider/dev.json at the same question_id.
EXAMPLES = [
    {
        "db": "concert_singer",
        "question": "How many singers do we have?",
        "sql": "SELECT COUNT(*) AS number_of_singers\nFROM singer;",
    },
    {
        "db": "pets_1",
        "question": "How much does the youngest dog weigh?",
        "sql": "SELECT weight FROM pets\nWHERE pettype = 'dog'\nORDER BY pet_age ASC LIMIT 1;",
    },
    {
        "db": "employee_hire_evaluation",
        "question": "Sort employee names by their age in ascending order.",
        "sql": "SELECT name FROM employee\nORDER BY age ASC;",
    },
]

# Naive SQL keyword highlight set, kept short on purpose.
_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "ORDER", "BY", "ASC", "DESC", "LIMIT",
    "AS", "COUNT", "GROUP", "JOIN", "ON", "AND", "OR", "NOT", "IN",
    "DISTINCT", "MAX", "MIN", "AVG", "SUM", "HAVING",
}


def _load_font(paths: list[str], size: int) -> ImageFont.FreeTypeFont:
    """Try a list of font paths in order; fall back to PIL default."""
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


SANS_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
SANS_BOLD_PATHS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
MONO_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _draw_sql_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    xy: tuple[int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    """Render an SQL line with naive keyword highlighting."""
    x, y = xy
    i = 0
    n = len(line)
    while i < n:
        # Whitespace passes through unchanged.
        if line[i].isspace():
            ws_end = i
            while ws_end < n and line[ws_end].isspace():
                ws_end += 1
            chunk = line[i:ws_end]
            draw.text((x, y), chunk, fill=SQL_COLOR, font=font)
            bbox = draw.textbbox((x, y), chunk, font=font)
            x = bbox[2]
            i = ws_end
            continue
        # Word.
        word_end = i
        while word_end < n and not line[word_end].isspace():
            word_end += 1
        word = line[i:word_end]
        upper = word.upper().strip(",;()")
        fill = KEYWORD if upper in _SQL_KEYWORDS else SQL_COLOR
        draw.text((x, y), word, fill=fill, font=font)
        bbox = draw.textbbox((x, y), word, font=font)
        x = bbox[2]
        i = word_end


def render_frame(
    db: str,
    question: str,
    sql: str,
    *,
    progress: float = 1.0,
    status: str = "idle",
    fonts: dict,
) -> Image.Image:
    """Render one frame of the demo card.

    progress is 0.0-1.0 fraction of SQL revealed (typing animation).
    status is one of "idle" | "thinking" | "writing" | "done".
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # ---- Header bar ----
    draw.rectangle([0, 0, WIDTH, 44], fill=CARD)
    draw.text((20, 12), "distill-sql", fill=TEXT, font=fonts["sans_bold"])
    draw.text(
        (132, 16), "1.5B 4-bit · 847 MB · on-device",
        fill=DIM, font=fonts["sans_small"],
    )
    # green dot + "live" on the right
    draw.ellipse([WIDTH - 80, 18, WIDTH - 68, 30], fill=GREEN)
    draw.text(
        (WIDTH - 60, 16), "live", fill=GREEN, font=fonts["sans_small_bold"],
    )

    # ---- Database hint ----
    draw.text(
        (20, 58),
        f"database: {db}",
        fill=DIM, font=fonts["sans_small"],
    )

    # ---- Question card ----
    draw.rectangle([20, 80, WIDTH - 20, 144], fill=CARD, outline=ACCENT, width=1)
    draw.text((34, 90), "QUESTION", fill=DIM, font=fonts["sans_label"])
    draw.text((34, 110), question, fill=TEXT, font=fonts["sans_body"])

    # ---- Action indicator (small colored dot + label) ----
    dot_x, dot_y = 22, 165
    if status == "idle":
        action_text = "press generate"
        action_color = ACCENT
    elif status == "thinking":
        action_text = "generating..."
        action_color = GREEN
    elif status == "writing":
        action_text = "streaming sql"
        action_color = GREEN
    else:
        action_text = "done in 1.16s"
        action_color = GREEN
    draw.ellipse([dot_x, dot_y, dot_x + 8, dot_y + 8], fill=action_color)
    draw.text((dot_x + 14, dot_y - 4), action_text, fill=action_color,
              font=fonts["sans_bold_small"])

    # ---- SQL output card ----
    draw.rectangle([20, 188, WIDTH - 20, HEIGHT - 20], fill=SQL_BG, outline=DIM, width=1)
    draw.text((34, 198), "SQL", fill=DIM, font=fonts["sans_label"])

    if status in ("idle",):
        pass
    elif status == "thinking":
        # animated dots
        dots = "." * (int(progress * 4) % 4 + 1)
        draw.text(
            (34, 224), f"thinking{dots}",
            fill=ACCENT, font=fonts["mono"],
        )
    else:
        # writing or done: progressive typing
        n_chars = int(len(sql) * progress)
        partial = sql[:n_chars]
        lines = partial.split("\n")
        for i, line in enumerate(lines):
            _draw_sql_line(draw, line, (34, 224 + i * 26), fonts["mono"])

    return img


def generate_frames(fonts: dict) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for ex in EXAMPLES:
        # 1. idle (question shown, awaiting button) ~1.5s
        for _ in range(15):
            frames.append(render_frame(ex["db"], ex["question"], "",
                                       progress=0.0, status="idle", fonts=fonts))
        # 2. thinking (animated dots) ~0.8s
        for t in range(8):
            frames.append(render_frame(ex["db"], ex["question"], "",
                                       progress=t / 8, status="thinking", fonts=fonts))
        # 3. writing (typing animation) ~1.2s
        for t in range(12):
            frames.append(render_frame(ex["db"], ex["question"], ex["sql"],
                                       progress=(t + 1) / 12, status="writing", fonts=fonts))
        # 4. done (hold complete SQL) ~1.8s
        for _ in range(18):
            frames.append(render_frame(ex["db"], ex["question"], ex["sql"],
                                       progress=1.0, status="done", fonts=fonts))
    return frames


def main() -> int:
    fonts = {
        "sans_bold": _load_font(SANS_BOLD_PATHS, 19),
        "sans_body": _load_font(SANS_PATHS, 15),
        "sans_small": _load_font(SANS_PATHS, 12),
        "sans_small_bold": _load_font(SANS_BOLD_PATHS, 12),
        "sans_label": _load_font(SANS_BOLD_PATHS, 10),
        "sans_bold_small": _load_font(SANS_BOLD_PATHS, 13),
        "mono": _load_font(MONO_PATHS, 14),
    }

    frames = generate_frames(fonts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ~12 fps. 83ms/frame.
    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=83,
        loop=0,
        optimize=True,
    )
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"wrote {OUT_PATH} ({size_mb:.2f} MB, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
