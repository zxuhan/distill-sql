# Recording the demo GIF

A short looping GIF of the live demo above the badges row in the README is the single highest-leverage piece of pitch material left for this project. This document captures the storyboard, recording tools, and final wiring step.

## Storyboard (~12 seconds, looped)

| seconds | action | what's on screen |
|---:|---|---|
| 0–2 | Page load with the URL visible | The Space page on `huggingface.co/spaces/zxuhan7/Distill-SQL` with `concert_singer` preloaded |
| 2–3 | Click **Generate SQL** on the default question (`How many singers are there?`) | Button click visible |
| 3–5 | SQL appears in the output box | `SELECT COUNT(*) FROM singer` (or close variant) |
| 5–6 | Open the dropdown, switch to `flight_2` | Schema and question both swap |
| 6–7 | Click **Generate SQL** on `What is the country of the airline JetBlue Airways?` | Button click |
| 7–9 | SQL appears | `SELECT Country FROM AIRLINES WHERE Airline = ...` |
| 9–12 | Slow zoom on the output box, fade to URL | URL banner |

Both questions are `easy`-difficulty, where the deployed 1.5B distilled scores 85.5% in eval. Success rate during recording should be high. If a generation comes out wrong, hit the button again and re-record that segment.

## Tools

Mac, recommended in order of friction:

1. **[Kap](https://getkap.co/)** — free, opens `.mov`, exports `.gif` directly with one click. Lowest friction.
2. **[Cleanshot X](https://cleanshot.com/)** — paid, better quality and built-in trim plus speed controls.
3. **QuickTime + ffmpeg** — record screen with QuickTime, convert to GIF on the command line.

The QuickTime + ffmpeg path:

```sh
# After QuickTime → File → New Screen Recording → save as demo.mov
ffmpeg -i demo.mov -vf "fps=15,scale=720:-1:flags=lanczos" -loop 0 assets/demo.gif

# Or with palette generation for better colors at smaller file size:
ffmpeg -i demo.mov -vf "fps=15,scale=720:-1:flags=lanczos,palettegen" -y palette.png
ffmpeg -i demo.mov -i palette.png -filter_complex "fps=15,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse" -loop 0 assets/demo.gif
```

Aim for roughly 720px wide at 12-15 fps. Final file should be under 10 MB so GitHub embeds it inline.

## Editing the wait time

The Space runs on a free CPU tier and takes 5-10 seconds per query. That dead time looks bad in a GIF. Two clean ways to handle it:

1. **Cut the wait segments down to about a second each.** Most demo GIFs do this. The viewer never notices because their attention is on the question and the answer, not the wait. In Kap, select the wait segment and trim.
2. **Speed up the wait segments 4-6×.** Looks natural enough as long as the click and the result stay at normal speed. In Cleanshot X you can do this directly; in `ffmpeg` it is `-filter:v "setpts=PTS/4"` over the relevant segment.

Both approaches are standard practice and not deceptive. Every CPU-tier ML demo on the internet does one of them.

## Wire it into the README

Once `assets/demo.gif` is in place and looks right:

```sh
# from the repo root
mkdir -p assets
mv ~/Downloads/demo.gif assets/demo.gif    # or wherever Kap saved it
```

Open `README.md` and find the comment block near the top:

```html
<!--
After recording the demo per docs/recording_demo.md, drop the file at
assets/demo.gif and uncomment the block below to embed it under the badges.

<p align="center">
  <img src="assets/demo.gif" alt="Distill-SQL live demo" width="720">
</p>
-->
```

Remove the `<!--` and `-->` so the `<p align="center">` block becomes live HTML. Commit and push.

## When to record on the local app instead

If the HF Space happens to be down (free Spaces auto-sleep after extended inactivity and the first request after wake-up takes longer), you can run the same Gradio UI locally for recording purposes. The UI is identical and the local M1 will respond in 2-3 seconds:

```sh
ADAPTER_PATH=artifacts/cloud_runs/scaling_1p5b_peft/adapter \
  uv run --with gradio --with peft --with transformers \
    python space/app.py
# opens http://localhost:7860
```

Tradeoff: the URL bar in the GIF says `localhost:7860` instead of `huggingface.co/spaces/...`, which is less convincing for a viewer. Prefer the live recording if the Space is responsive.
