import time
import re
import requests
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
import logger
import analyzer as a

TOKEN = "TOKEN"  # for hakaton

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


def msg_limiter(msg):
    msg_limit = 3896
    if len(msg) > msg_limit:
        msg = msg[:msg_limit]
    return msg


def img_proc(url):
    response = requests.get(url)
    return a.img_analyzer(response)


def photo_msg(msg, from_url=False):
    global cache, start_time
    uid = msg.user_id
    cache, start_time = logger.logger_vk(msg, "", cache, start_time)
    if not from_url:
        vk.messages.send(user_id=uid, message="😮", random_id=0)
        time.sleep(1)
        vk.messages.send(user_id=uid, reply_to=msg.message_id, message="Это что... Картинка???", random_id=0)
        vk.messages.send(user_id=uid, message="Смотри че могу)", random_id=0)
        url = vk.messages.getHistoryAttachments(peer_id=event.user_id, media_type="photo",
                                                count=1)["items"][-1]["attachment"]["photo"]["sizes"][-1]["url"]
    else:
        url = msg.text
    vk.messages.send(user_id=uid, message="Хоба!", random_id=0)
    ans = upload.photo_messages(photos=img_proc(url))[0]
    owner_id, media_id = ans["owner_id"], ans["id"]
    vk.messages.send(user_id=uid, attachment=f"photo{owner_id}_{media_id}", random_id=0)


def text_msg(msg):
    global cache, start_time
    uid = msg.user_id
    if not ("http" in msg.text):
        answer = a.wa_analyzer(msg.text)
        cache, start_time = logger.logger_vk(msg, answer, cache, start_time)
        vk.messages.send(user_id=uid, message=msg_limiter(answer), random_id=0)
    else:
        pattern = r"^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+[.](jpg|jpeg|png|gif)$"
        if re.match(pattern, msg.text, re.IGNORECASE):
            vk.messages.send(user_id=uid, message="Сейчас, скачаю и пришлю", random_id=0)
            photo_msg(msg, from_url=True)
        else:
            vk.message.send(user_id=uid, text=r"Я не могу найти тут картинку... Проверьте, пожалуйста, что "
                                              r"ссылка оканчивается на .jpg, .jpeg, .png или .gif...")


prev_index = None


for event in longpoll.listen():
    continue_status = False
    if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
        bot_activation, continue_status = activating_bot(event, continue_status)
        if continue_status:
            continue
        if bot_activation:
            text_msg(event)
        else:
            service_msg = "Введите /start, чтобы начать"
            vk.messages.send(user_id=event.user_id, message=service_msg, random_id=0)
    elif event.type == VkEventType.MESSAGE_NEW and event.to_me and event.attachments["attach1_type"] == "photo":
        bot_activation, continue_status = activating_bot(event, continue_status)
        if continue_status:
            continue
        if bot_activation:
            photo_msg(event)
        else:
            service_msg = "Введите /start, чтобы начать"
            vk.messages.send(user_id=event.user_id, message=service_msg, random_id=0)
    elif event.type == VkEventType.MESSAGE_NEW and event.to_me and event.attachments["attach1_type"] == "doc":
        bot_activation, continue_status = activating_bot(event, continue_status)
        if continue_status:
            continue
        if bot_activation:
            service_msg = "Я не умею работать с доками("
        else:
            service_msg = "Введите /start, чтобы начать"
        vk.messages.send(user_id=event.user_id, message=service_msg, random_id=0)
