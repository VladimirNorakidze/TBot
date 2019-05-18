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
indexes = []


def wa_analyzer(text=None):
    """
    Функция-анализатор описания; возвращающая ссылки, названия и
    product_id товаров, наиболее близких к описанию
    """
    global idx, prev_text, indexes
    n = 3
    msg_status = True
    if text is not None:
        indexes = TextBrain.main(request=text)
        idx = 0
    if indexes == []:
        msg_status = False
        return "Неа, введи одежду)", False, False, msg_status
    urls = [data.iloc[i].image_links for i in indexes[idx:idx+n:]]
    titles = [data.iloc[i].title for i in indexes[idx:idx+n:]]
    product_ids = [data.iloc[i].product_id for i in indexes[idx:idx+n:]]
    idx += n
    return urls, titles, product_ids, msg_status


def img_analyzer(response):
    """Функция-анализатор изображения: определяет подкатегорию по
    изображению и отдает на вход анализатору описания"""
    img = Image.open(BytesIO(response.content))
    width, height = (224, 224)
    img = img.resize((width, height), Image.ANTIALIAS)
    img = np.array(img).T
    subcategory = wearing.wearing_classification(img)
    print(subcategory)
    return wa_analyzer(subcategory)
