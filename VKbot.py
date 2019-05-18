import time
import re
import requests
import io
import numpy as np
from PIL import Image
import emoji
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
import logger
import analyzer as a

TOKEN = "bafa83804e118b05e67670d10ac9993b98369fb6129c353e85efef71dfa0070bf43b3d2b551ed67d9d0e8"  # for hakaton

bot_activation = True
vk_session = vk_api.VkApi(token=TOKEN)
longpoll = VkLongPoll(vk_session)
upload = vk_api.VkUpload(vk_session)

vk = vk_session.get_api()

start_time = time.time()
cache = []


def activating_bot(event, continue_status):
    status = bot_activation
    if "/start" in event.text.lower() and not status:
        msg = '''О, привет, {}).'''.format(vk.users.get(user_id=event.user_id)[0]["first_name"])
        vk.messages.send(user_id=event.user_id, message=msg, random_id=0)
        status = True
        continue_status = True
    elif bot_activation and ("/stop" in event.text.lower()):
        msg = "Пока ✌️ Захочешь поболтать - пиши /start"
        vk.messages.send(user_id=event.user_id, message=msg, random_id=0)
        status = False
        continue_status = True
    return status, continue_status


def check_emoji(text):
    f = lambda x: bool(emoji.get_emoji_regexp().search(x))
    text = emoji.get_emoji_regexp().split(text)
    if not (text[0] or text[-1]):
        if len(text) == 3 and f(text[1]):
            return True
        elif len(text) == 5:
            if f(text[1]) and (text[2] == "\u200d") and f(text[3]):
                return True
        else:
            return False
    else:
        return False


def msg_limiter(msg):
    msg_limit = 3896
    if len(msg) > msg_limit:
        msg = msg[:msg_limit]
    return msg


def photo_sender(user_id, url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    img = np.array(img).T
    byte_img = io.BytesIO()
    Image.fromarray(img.T, mode="RGB").save(byte_img, "JPEG")
    byte_img.seek(0)
    ans = upload.photo_messages(photos=byte_img)[0]
    owner_id, media_id = ans["owner_id"], ans["id"]
    vk.messages.send(user_id=user_id, attachment=f"photo{owner_id}_{media_id}", random_id=0)


def img_proc(url):
    response = requests.get(url)
    return a.img_analyzer(response)


def photo_msg(msg):
    uid = msg.user_id
    vk.messages.send(user_id=uid, message="😮", random_id=0)
    vk.messages.send(user_id=uid, reply_to=msg.message_id, message="Это что... Картинка???", random_id=0)
    vk.messages.send(user_id=uid, message="Смотри че могу)", random_id=0)
    url = vk.messages.getHistoryAttachments(peer_id=event.user_id, media_type="photo",
                                            count=1)["items"][-1]["attachment"]["photo"]["sizes"][-1]["url"]
    urls, titles, _ = img_proc(url)
    time.sleep(1)
    for url in urls:
        photo_sender(user_id=uid, url=url)
    vk.messages.send(user_id=uid, message="Хоба!", random_id=0)
    return titles


def text_msg(msg):
    uid = msg.user_id
    if (msg.text.lower() == 'еще' or msg.text.lower() == 'ещё'):
        urls, titles, msg_status = a.wa_analyzer()
        if msg_status:
            vk.messages.send(user_id=uid, message="Одну секундочку...", random_id=0)
            time.sleep(1)
            for url in urls:
                photo_sender(user_id=uid, url=url)
            vk.messages.send(user_id=uid, message="Хоба!", random_id=0)
        else:
            vk.messages.send(user_id=uid, message=urls, random_id=0)
    elif not ("http" in msg.text):
        urls, titles, _ = a.wa_analyzer(text=msg.text)
        vk.messages.send(user_id=uid, message="Одну секундочку...", random_id=0)
        time.sleep(1)
        for url in urls:
            photo_sender(user_id=uid, url=url)
        vk.messages.send(user_id=uid, message="Хоба!", random_id=0)
    else:
        pattern = r"^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+[.](jpg|jpeg|png|gif)$"
        if re.match(pattern, msg.text, re.IGNORECASE):
            vk.messages.send(user_id=uid, message="Сейчас, скачаю и пришлю", random_id=0)
            time.sleep(1)
            urls, titles, _ = img_proc(url=msg.text)
            for url in urls:
                photo_sender(user_id=uid, url=url)
            vk.messages.send(user_id=uid, message="Хоба!", random_id=0)
        else:
            vk.message.send(user_id=uid, text=r"Я не могу найти тут картинку... Проверьте, пожалуйста, что "
                                              r"ссылка оканчивается на .jpg, .jpeg, .png или .gif...")
            titles = []
    return titles


for event in longpoll.listen():
    continue_status = False
    if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
        bot_activation, continue_status = activating_bot(event, continue_status)
        if continue_status:
            continue
        if bot_activation:
            if check_emoji(event.text):
                name = vk.users.get(user_id=event.user_id)[0]["first_name"]
                ans = f"{name}, я не смогу обработать сообщение состоящее из одного смайла..."
                ans += "Пришли то, что хотел бы увидеть :3"
                vk.messages.send(user_id=event.user_id, message=ans, random_id=0)
                cache, start_time = logger.logger_vk(event, ans, cache, start_time)
            else:
                ans = text_msg(event)
                cache, start_time = logger.logger_vk(event, ans, cache, start_time)
        else:
            service_msg = "Введите /start, чтобы начать"
            vk.messages.send(user_id=event.user_id, message=service_msg, random_id=0)
            cache, start_time = logger.logger_vk(event.text, "Bot is not activated", cache, start_time)
    elif event.type == VkEventType.MESSAGE_NEW and event.to_me and event.attachments["attach1_type"] == "photo":
        bot_activation, continue_status = activating_bot(event, continue_status)
        if continue_status:
            continue
        if bot_activation:
            ans = photo_msg(event)
            cache, start_time = logger.logger_vk(event, ans, cache, start_time)
        else:
            service_msg = "Введите /start, чтобы начать"
            vk.messages.send(user_id=event.user_id, message=service_msg, random_id=0)
            cache, start_time = logger.logger_vk(event.text, "Bot is not activated", cache, start_time)
    elif event.type == VkEventType.MESSAGE_NEW and event.to_me and (event.attachments["attach1_type"] == "doc"
            or event.attachments["attach1_type"] == "sticker"):
        bot_activation, continue_status = activating_bot(event, continue_status)
        if continue_status:
            continue
        if bot_activation:
            service_msg = "Я не понимаю, что это 😭😭😭"
            vk.messages.send(user_id=event.user_id, reply_to=event.message_id, message=service_msg, random_id=0)
            service_msg = "Опиши, что хочешь или пришли фотку("
        else:
            service_msg = "Введите /start, чтобы начать"
        vk.messages.send(user_id=event.user_id, message=service_msg, random_id=0)
