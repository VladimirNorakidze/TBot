import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
import botTextBrain as TextBrain
import wearing

print("Loading Database...")
data = pd.read_csv("./data/dataset_sample.csv")
data = data.fillna("NaN")
print("Ready to work!")


idx = 0
prev_text = ""
indexes = []


def wa_analyzer(text=None):
    global idx, prev_text, indexes
    n = 3
    msg_status = True
    if text is not None:
        indexes = TextBrain.main(request=text)
        prev_text = text
        idx = 0
    if indexes == []:
        msg_status = False
        return "Неа, введи одежду)", False, msg_status
    urls = [data.iloc[i].image_links for i in indexes[idx:idx+n:]]
    titles = [data.iloc[i].title for i in indexes[idx:idx+n:]]
    idx += n
    return urls, titles, msg_status


def img_analyzer(response):
    img = Image.open(BytesIO(response.content))
    width, height = (224, 224)
    img = img.resize((width, height), Image.ANTIALIAS)
    img = np.array(img).T
    subcategory = wearing.wearing_classification(img)
    print(subcategory)
    return wa_analyzer(subcategory)
