import sys
import re
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
import pymorphy2 as pm2
import botTextBrain as TextBrain
# import module that will extract recommends

print("Loading Database...")
data = pd.read_csv("./data/dataset_sample.csv")
data = data.fillna("NaN")
print("Ready to work!")


def wa_analyzer(msg: str):
    indexes = TextBrain.main(msg)
    urls = [data.iloc[idx].image_links for idx in indexes]
    titles = [data.iloc[idx].title for idx in indexes]
    return urls, titles


def img_analyzer(response):
    img = Image.open(BytesIO(response.content))
    width = 224
    height = 224
    img = img.resize((width, height), Image.ANTIALIAS)
    img = np.array(img).T
    # subcategory = <функция, определяющая субкатегорию по фото>
    # urls, titles = wa_analyzer(subcategory)
    return wa_analyzer(subcategory)
