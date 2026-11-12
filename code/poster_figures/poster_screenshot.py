"""Figures 5 and 6: the screenshots used on the R!Louisville poster, reused as the figures.

Usage: python poster_screenshot.py <screenshot png> <figure name> [bottom crop in px]
Writes <figure name>.png (300 dpi) and a PDF wrapper to posterstyle.OUT. No data CSV: the figure is a screenshot.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import posterstyle as S  # noqa: E402

src, name = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
if len(sys.argv) > 3:
    im = im.crop((0, 0, im.size[0], int(sys.argv[3])))
os.makedirs(S.OUT, exist_ok=True)
out = os.path.join(S.OUT, name)
im.save(out + ".png", dpi=(300, 300))
fig = plt.figure(figsize=(im.size[0] / 300, im.size[1] / 300))
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im); ax.axis("off")
fig.savefig(out + ".pdf", dpi=300)
plt.close(fig)
for stale in (out + "_data.csv",):
    if os.path.exists(stale):
        os.remove(stale)
print("saved", out, im.size)
