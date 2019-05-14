import sys
import re
import pymorphy2 as pm2
# import module that will extract recommends


def wa_analyzer(msg: str):
    morphy = pm2.MorphAnalyzer()

    norm_msg_pos = "_".join([morphy.normal_forms(x)[0] for x in re.findall(r"\w+", msg)])

    return norm_msg_pos


def img_analyzer(picture):
    with open("test.jpg", "wb") as img:
        img.write(picture.content)
