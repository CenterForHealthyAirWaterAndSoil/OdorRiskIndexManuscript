"""Updated poster Figure 6: the public odor-risk forecast site, as live on the day the package was built.

Screenshot: headless Google Chrome, 1600 x 1000 CSS px at 2x device scale, of
https://nwright083.github.io/weather-variable-analysis/ (the deployed site; local, uncommitted changes are not on it).
Usage: python fig6_public_site.py <raw screenshot png>. The empty page area below the map is cropped; a PDF wrapper is
written beside the PNG. No data CSV: the figure is a screenshot.
"""
import os
import sys
from datetime import date

from PIL import Image

import posterstyle as S

src = sys.argv[1]
im = Image.open(src).convert("RGB")
w, h = im.size
scale = w / 1600
im = im.crop((0, 0, w, int(775 * scale)))
os.makedirs(S.OUT, exist_ok=True)
out = os.path.join(S.OUT, "Fig7_public_forecast_site")
im.save(out + ".png", dpi=(300, 300))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(im.size[0] / 300, im.size[1] / 300))
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im); ax.axis("off")
fig.savefig(out + ".pdf", dpi=300)
plt.close(fig)
print("saved", out, im.size, "captured", date.today().isoformat())
