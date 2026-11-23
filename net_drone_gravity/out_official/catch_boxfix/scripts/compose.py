"""Side-by-side video with a label band: python compose.py left.mp4 right.mp4 out.mp4 "left label" "right label" """
import sys, av, numpy as np
from PIL import Image, ImageDraw, ImageFont
a_path, b_path, out_path, la, lb = sys.argv[1:6]
A, B = av.open(a_path), av.open(b_path)
fa = [f.to_ndarray(format="rgb24") for f in A.decode(video=0)]
fb = [f.to_ndarray(format="rgb24") for f in B.decode(video=0)]
fps = A.streams.video[0].average_rate
n = min(len(fa), len(fb))
try:
    font = ImageFont.truetype("arial.ttf", 24)
except Exception:
    font = ImageFont.load_default()
O = av.open(out_path, "w")
st = O.add_stream("libx264", rate=fps)
W, H, BAND = 960, 540, 44
st.width, st.height, st.pix_fmt = 2 * W, H + BAND, "yuv420p"
st.options = {"crf": "20"}
for i in range(n):
    ims = []
    for fr, lab in ((fa[i], la), (fb[i], lb)):
        im = Image.new("RGB", (W, H + BAND), (255, 255, 255))
        im.paste(Image.fromarray(fr).resize((W, H), Image.LANCZOS), (0, BAND))
        ImageDraw.Draw(im).text((12, 8), lab, fill=(10, 10, 10), font=font)
        ims.append(np.asarray(im))
    for p in st.encode(av.VideoFrame.from_ndarray(np.hstack(ims), format="rgb24")):
        O.mux(p)
for p in st.encode():
    O.mux(p)
O.close()
print(n, len(fa), len(fb))
