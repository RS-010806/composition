"""Synthetic colored-shape images for compositional generalization tests.

A compound is (color, shape) on a 6 x 6 grid. Every image shows one object on a
noisy light background, with jitter in hue, position, size and rotation, so that
neither attribute can be read off a single pixel.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]
SHAPES = ["circle", "square", "triangle", "diamond", "star", "plus"]
RGB = {
    "red": (215, 40, 40),
    "orange": (240, 135, 20),
    "yellow": (225, 205, 30),
    "green": (40, 165, 60),
    "blue": (40, 90, 220),
    "purple": (140, 50, 180),
}
C, S = len(COLORS), len(SHAPES)

# The cross: every color with shape 0, and color 0 with every shape (C + S - 1 = 11 compounds).
CROSS = sorted({(c, 0) for c in range(C)} | {(0, s) for s in range(S)})
# Home shape of each color in the correlated "pretraining" distribution (the diagonal).
HOME = {c: c for c in range(C)}
# Held-out compounds: never shown in pretraining or post-training. Off the cross and off
# the diagonal; each of colors 1..5 and shapes 1..5 is held out exactly twice.
HELD_OUT = sorted({(c, (c % 5) + 1) for c in range(1, 6)} | {(c, ((c + 1) % 5) + 1) for c in range(1, 6)})
ALL = [(c, s) for c in range(C) for s in range(S)]
SEEN = [k for k in ALL if k not in HELD_OUT]

assert len(CROSS) == 11 and len(HELD_OUT) == 10 and not set(CROSS) & set(HELD_OUT)
assert all((c, HOME[c]) not in HELD_OUT for c in range(C))


def _polygon(shape, cx, cy, r, rot):
    def rotate(points):
        cr, sr = math.cos(rot), math.sin(rot)
        return [(cx + x * cr - y * sr, cy + x * sr + y * cr) for x, y in points]

    if shape == "square":
        a = r * 0.82
        return rotate([(-a, -a), (a, -a), (a, a), (-a, a)])
    if shape == "diamond":
        return rotate([(0, -r), (r * 0.7, 0), (0, r), (-r * 0.7, 0)])
    if shape == "triangle":
        return rotate([(r * math.cos(t), r * math.sin(t)) for t in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)])
    if shape == "star":
        pts = []
        for k in range(10):
            t = -math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append((rr * math.cos(t), rr * math.sin(t)))
        return rotate(pts)
    if shape == "plus":
        a, b = r, r * 0.34
        return rotate([(-b, -a), (b, -a), (b, -b), (a, -b), (a, b), (b, b), (b, a), (-b, a),
                       (-b, b), (-a, b), (-a, -b), (-b, -b)])
    raise ValueError(shape)


def render(color, shape, rng, size=32, objects=None):
    """Render one image. `objects` overrides with a list of (color, shape, cx, cy, r)."""
    bg = int(rng.integers(200, 246))
    img = Image.new("RGB", (size * 4, size * 4), (bg, bg, bg))  # draw at 4x, then downsample
    draw = ImageDraw.Draw(img)
    if objects is None:
        r = rng.uniform(0.26, 0.36) * size
        cx = rng.uniform(r + 1, size - r - 1)
        cy = rng.uniform(r + 1, size - r - 1)
        objects = [(color, shape, cx, cy, r)]
    for ci, si, cx, cy, r in objects:
        base = np.array(RGB[COLORS[ci]], dtype=float)
        rgb = tuple(int(v) for v in np.clip(base * rng.uniform(0.85, 1.08) + rng.normal(0, 8, 3), 0, 255))
        rot = 0.0 if SHAPES[si] in ("square", "diamond") else rng.uniform(-0.3, 0.3)
        if SHAPES[si] == "circle":
            draw.ellipse([(cx - r) * 4, (cy - r) * 4, (cx + r) * 4, (cy + r) * 4], fill=rgb)
        else:
            draw.polygon([(x * 4, y * 4) for x, y in _polygon(SHAPES[si], cx, cy, r, rot)], fill=rgb)
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr + rng.normal(0, 0.02, arr.shape).astype(np.float32)
    return np.clip(arr, 0, 1)


def bank(per_compound, seed, size=32):
    """Pre-render `per_compound` images for every compound. Returns X[c, s, i] and labels."""
    rng = np.random.default_rng(seed)
    X = np.zeros((C, S, per_compound, size, size, 3), dtype=np.float32)
    for c in range(C):
        for s in range(S):
            for i in range(per_compound):
                X[c, s, i] = render(c, s, rng, size)
    return X
