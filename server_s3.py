#!/usr/bin/env python3
"""
Flask collector entrypoint for Render with stroke dataset files storage on S3.
Same HTTP API as server.py; uses auto_save_strokes_s3.get_strokes_store() for S3 storage.

Required env:
  - S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

Install:
  pip install -r requirements-render.txt

Usage:
  python server_s3.py --file syl.txt --host 0.0.0.0 --port $PORT
  gunicorn server_s3:app --bind 0.0.0.0:$PORT
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw
from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS
import random

from auto_save_strokes_s3 import StrokesS3Store, get_strokes_store, parse_strokes_from_text

IMAGE_DIRS = {"single": "single", "stroke": "stroke", "time": "time"}
IMG_SIZE = 128

app = Flask(__name__, static_folder="static")
CORS(app)

TEXT_LINES: list[str] = []
STORE: StrokesS3Store | None = None


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _get_store() -> StrokesS3Store:
    if STORE is None:
        raise RuntimeError("STORE not initialized; run main() or load under gunicorn")
    return STORE


def _configure(lines_path: str) -> None:
    global TEXT_LINES, STORE
    _load_env()
    with open(lines_path, encoding="utf-8") as f:
        TEXT_LINES = [l.strip() for l in f if l.strip()]
    STORE = get_strokes_store()


def sort_key(name):
    return [int(n) for n in re.findall(r"\d+", name)]


def get_line_save_counts(user_name):
    return _get_store().get_line_save_counts(user_name)


def user_progress(user_name):
    return _get_store().user_progress(user_name)


def normalize_strokes(strokes, img_size, padding=10):
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span = max(max_x - min_x, max_y - min_y)
    scale = (img_size - 2 * padding) / span if span > 0 else 1

    normed = [[((x - min_x) * scale, (y - min_y) * scale) for x, y in s] for s in strokes]

    all_x2 = [p[0] for s in normed for p in s]
    all_y2 = [p[1] for s in normed for p in s]
    ox = (img_size - (max(all_x2) - min(all_x2))) / 2 - min(all_x2)
    oy = (img_size - (max(all_y2) - min(all_y2))) / 2 - min(all_y2)
    return [[(x + ox, y + oy) for x, y in s] for s in normed]


def generate_image(strokes, img_size, color_mode):
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)

    if color_mode != "single":
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)

    for i, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue

        if color_mode == "single":
            color = 0
        elif color_mode == "stroke":
            color = tuple(random.randint(0, 255) for _ in range(3))
        elif color_mode == "time":
            color = (
                int(255 * i / max(1, len(strokes))),
                0,
                255 - int(255 * i / max(1, len(strokes))),
            )

        for j in range(1, len(stroke)):
            draw.line([stroke[j - 1], stroke[j]], fill=color, width=2)

    return img


def ensure_image(user_name, txt_fname, color_mode):
    base = os.path.splitext(txt_fname)[0]
    out_dir = os.path.join(IMAGE_DIRS[color_mode], user_name)
    out_path = os.path.join(out_dir, base + ".png")

    if not os.path.exists(out_path):
        text = _get_store().read_stroke_txt(user_name, txt_fname)
        strokes = parse_strokes_from_text(text)
        if strokes:
            strokes = normalize_strokes(strokes, IMG_SIZE)
            img = generate_image(strokes, IMG_SIZE, color_mode)
            os.makedirs(out_dir, exist_ok=True)
            img.save(out_path)

    return out_path


@app.route("/api/lines")
def api_lines():
    return jsonify({"lines": TEXT_LINES})


@app.route("/api/users")
def api_users():
    st = _get_store()
    users = []
    for u in st.list_users():
        info = st.read_user_info(u)
        users.append({"name": u, "progress": st.user_progress(u), "info": info})
    return jsonify({"users": users, "total": len(TEXT_LINES)})


@app.route("/api/users", methods=["POST"])
def create_user():
    data = request.json
    name = data.get("name", "").strip().replace(" ", "_")
    if not name:
        return jsonify({"error": "Name required"}), 400
    info = {
        "name": name,
        "age": data.get("age", ""),
        "sex": data.get("sex", ""),
        "education": data.get("education", ""),
    }
    _get_store().create_user(name, info)
    return jsonify({"ok": True, "name": name, "progress": 0})


@app.route("/api/save", methods=["POST"])
def save_sample():
    data = request.json
    user_name = data.get("user")
    line_index = data.get("index")
    strokes = data.get("strokes")

    if not user_name or line_index is None or not strokes:
        return jsonify({"error": "Missing fields"}), 400

    st = _get_store()
    if not st.user_exists(user_name):
        return jsonify({"error": "Unknown user"}), 404

    base = str(line_index + 1)
    i = 1
    while True:
        fname = f"{base}-{i}.txt"
        if not st.stroke_txt_exists(user_name, fname):
            break
        i += 1

    lines_out = []
    for si, stroke in enumerate(strokes):
        lines_out.append(f"STROKE {si+1}\n")
        for pt in stroke:
            lines_out.append(f"{pt['x']} {pt['y']} {pt['t']:.6f}\n")
        lines_out.append("\n")
    st.write_stroke_txt(user_name, fname, "".join(lines_out))

    return jsonify(
        {
            "ok": True,
            "file": fname,
            "progress": user_progress(user_name),
        }
    )


@app.route("/api/progress/<user_name>")
def api_progress(user_name):
    """Legacy endpoint — kept for compatibility."""
    return jsonify(
        {
            "progress": user_progress(user_name),
            "total": len(TEXT_LINES),
        }
    )


@app.route("/api/line_counts/<user_name>")
def api_line_counts(user_name):
    """
    Returns per-line save counts for the given user.
    Response: { "counts": { "0": 3, "1": 2, "1374": 1, … } }
    Keys are 0-based line indices (as strings).
    """
    st = _get_store()
    if not st.user_exists(user_name):
        return jsonify({"error": "Unknown user"}), 404

    counts = get_line_save_counts(user_name)
    return jsonify({"counts": counts})


@app.route("/api/gallery/<user_name>/<color_mode>", methods=["POST"])
def api_gallery(user_name, color_mode):
    if color_mode not in IMAGE_DIRS:
        return jsonify({"error": "Invalid mode"}), 400

    st = _get_store()
    if not st.user_exists(user_name):
        return jsonify({"error": "Unknown user"}), 404

    data = request.get_json(silent=True) or {}
    known_set = set(data.get("known", []))

    txt_files = sorted(st.list_txt_files(user_name), key=sort_key)

    items = []
    for txt_fname in txt_files:
        ensure_image(user_name, txt_fname, color_mode)

        if txt_fname in known_set:
            continue

        base = os.path.splitext(txt_fname)[0]
        m = re.match(r"^(\d+)-\d+$", base)
        label = ""
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(TEXT_LINES):
                label = f"{idx+1}: {TEXT_LINES[idx]}"

        items.append(
            {
                "txt": txt_fname,
                "img": f"/api/gallery/image/{color_mode}/{user_name}/{base}.png",
                "label": label,
            }
        )

    return jsonify({"items": items})


@app.route("/api/gallery/image/<color_mode>/<user_name>/<filename>")
def api_gallery_image(color_mode, user_name, filename):
    if color_mode not in IMAGE_DIRS:
        abort(404)
    img_dir = os.path.join(IMAGE_DIRS[color_mode], user_name)
    return send_from_directory(img_dir, filename)


@app.route("/api/delete/<user_name>/<fname>", methods=["DELETE"])
def api_delete(user_name, fname):
    """Delete a stroke .txt and all 3 mode images for it."""
    if not re.match(r"^\d+-\d+\.txt$", fname):
        return jsonify({"error": "Invalid filename"}), 400

    st = _get_store()
    if not st.stroke_txt_exists(user_name, fname):
        return jsonify({"error": "File not found"}), 404

    st.delete_stroke_txt(user_name, fname)

    base = os.path.splitext(fname)[0]
    for mode_dir in IMAGE_DIRS.values():
        img_path = os.path.join(mode_dir, user_name, base + ".png")
        if os.path.exists(img_path):
            os.remove(img_path)

    updated_counts = get_line_save_counts(user_name)
    return jsonify(
        {
            "ok": True,
            "progress": user_progress(user_name),
            "lineCounts": updated_counts,
        }
    )


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


def main():
    parser = argparse.ArgumentParser(description="Myanmar HW Collector – Render + S3")
    parser.add_argument("--file", required=True, help="Text file with one syllable per line")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    _configure(args.file)

    print(f"\n  Loaded {len(TEXT_LINES)} lines from {args.file}")
    app.run(host=args.host, port=args.port, debug=False)


# gunicorn: import module as server_s3, never run main()
if __name__ != "__main__":
    _default_syl = os.environ.get("SYLLABLE_FILE", "syl.txt")
    if os.path.isfile(_default_syl):
        _configure(_default_syl)

# no gunicorn: run main() directly
if __name__ == "__main__":
    main()
