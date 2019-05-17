import sys
import re
import numpy as np
from PIL import Image
from io import BytesIO
import pymorphy2 as pm2
# import module that will extract recommends


def wa_analyzer(msg: str):
    morphy = pm2.MorphAnalyzer()

    norm_msg_pos = " ".join([morphy.normal_forms(x)[0] for x in re.findall(r"\w+", msg)])

    return norm_msg_pos


def img_analyzer(response):
    img = Image.open(BytesIO(response.content))
    width = 224
    height = 224
    print(img.size)
    img = img.resize((width, height), Image.ANTIALIAS)
    print(img.size)
    img = np.array(img).T
    print(img.shape)
    b = BytesIO()
    Image.fromarray(img.T, mode="RGB").save(b, "JPEG")
    b.seek(0)
    return b
