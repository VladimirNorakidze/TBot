import os
import re
import requests
import time
import emoji
import telebot
import logger
import analyzer as a

TOKEN = "TOKEN"

start_time = time.time()
cache = []

bot = telebot.TeleBot(TOKEN)
start_status = False

with open("botpid.txt", "w") as file:
    file.write(str(os.getpid()))


def check_emoji(text):
    """
    Функция, проверяющая, что сообщение состоит не из одного смайлика
    """
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


def img_processing(fileid=None, url=None):
    """
    Функция, получающая url изображения и
    вызывающая функцию-анализатора изображений
    """
    if (fileid is not None) and (url is None):
        file = bot.get_file(file_id=fileid)
        response = requests.get(f"https://api.telegram.org/file/bot{TOKEN}/{file.file_path}")
    else:
        response = requests.get(url)
    return a.img_analyzer(response)


def photo_sender(chat_id, url, product_id):
    """
    Функция отправляющая фото и ссылку на объявление в Юле пользователю
    """
    response = requests.get(url)
    bot.send_photo(chat_id=chat_id, photo=response.content)
    prod_url = 'https://youla.io/p' + str(product_id)
    bot.send_message(chat_id=chat_id, text=prod_url)


def photo_msg(msg):
    """
    Функция, вызываемая при типе входящего сообщения 'фото'
    """
    chatid = msg.chat.id
    bot.send_message(chat_id=chatid, text="😮")
    bot.reply_to(msg, text="Это что... Картинка???")
    bot.send_message(chat_id=chatid, text="Смотри, тебе может еще понравиться это)")
    urls, titles, prod_id, _ = img_processing(fileid=msg.photo[-1].file_id)
    time.sleep(1)
    for i, url in enumerate(urls):
        photo_sender(chat_id=chatid, url=url, product_id=prod_id[i])
    bot.send_message(chat_id=chatid, text="Хоба!")
    return titles


def doc_msg(msg):
    """
    Функция, вызываемая при типе входящего сообщения 'документ'
    """
    chatid = msg.chat.id
    bot.send_message(chat_id=chatid, text="😮")
    time.sleep(1)
    bot.reply_to(msg, text="Вот и секретные докумееееееентики подъехали)")
    bot.send_message(chat_id=chatid, text="Ща верну, секунду")
    urls, titles, prod_id, _ = img_processing(fileid=msg.document.file_id)
    time.sleep(1)
    for i, url in enumerate(urls):
        photo_sender(chat_id=chatid, url=url, product_id=prod_id[i])
    bot.send_message(chat_id=chatid, text="Хоба!")
    return titles


def text_msg(msg):
    """
    Функция, вызываемая при типе входящего сообщения 'текст' или 'url'
    """
    chatid = msg.chat.id
    if (msg.text.lower() == 'еще' or msg.text.lower() == 'ещё'):
        urls, titles, prod_id, msg_status = a.wa_analyzer()
        if msg_status:
            bot.send_message(chat_id=chatid, text="Одну секундочку...")
            time.sleep(1)
            for i, url in enumerate(urls):
                photo_sender(chat_id=chatid, url=url, product_id=prod_id[i])
            bot.send_message(chat_id=chatid, text="Хоба!")
        else:
            bot.send_message(chat_id=chatid, text=urls)
    elif not ("http" in msg.text):
        urls, titles, prod_id,  _ = a.wa_analyzer(msg.text)
        bot.send_message(chat_id=chatid, text="Одну секундочку...")
        time.sleep(1)
        for i, url in enumerate(urls):
            photo_sender(chat_id=chatid, url=url, product_id=prod_id[i])
        bot.send_message(chat_id=chatid, text="Хоба!")
    else:
        pattern = r"^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+[.](jpg|jpeg|png|gif)$"
        if re.match(pattern, msg.text, re.IGNORECASE):
            bot.send_message(chat_id=chatid, text="Сейчас пришлю что-нибудь...")
            time.sleep(1)
            urls, titles, prod_id, _ = img_processing(url=msg.text)
            for i, url in enumerate(urls):
                photo_sender(chat_id=chatid, url=url, product_id=prod_id[i])
        else:
            bot.send_message(chat_id=chatid, text=r"Я не могу найти тут картинку... Проверьте, пожалуйста, что "
                                                  r"ссылка оканчивается на .jpg, .jpeg, .png или .gif...")
            titles = []
    return titles


def main(messages):
    global cache, start_time
    if messages[-1].text != "/stop":
        if start_status:
            for m in messages:
                if m.content_type == "text":
                    if check_emoji(m.text):
                        name = m.from_user.first_name
                        ans = f"{name}, я не смогу обработать сообщение состоящее из одного смайла..."
                        ans += "Пришли то, что хотел бы увидеть :3"
                        bot.send_message(m.chat.id, text=ans)
                    else:
                        ans = text_msg(m)
                        cache, start_time = logger.logger(m, ans, cache, start_time)
                elif m.content_type == "photo":
                    ans = photo_msg(m)
                    cache, start_time = logger.logger(m, ans, cache, start_time)
                elif m.content_type == "document" and ("image" in m.document.mime_type):
                    ans = doc_msg(m)
                    cache, start_time = logger.logger(m, ans, cache, start_time)
                elif m.content_type == "sticker":
                    bot.send_message(m.chat.id, text=m.sticker.emoji)
                else:
                    cache, start_time = logger.logger(m, "Unknown type...", cache, start_time)
                    bot.reply_to(m, text="Я не понимаю, что это 😭😭😭")
                    time.sleep(1)
                    bot.send_message(chat_id=m.chat.id, text="Опиши, что хочешь или пришли фотку(")


@bot.message_handler(commands=["start"])
def echo_start(msg):
    global start_status
    if not start_status:
        start_status = True
        welcome_text = f"Рад приветствовать, {msg.from_user.first_name}. Если я тебя утомлю, напиши /stop."
        bot.reply_to(msg, welcome_text)


@bot.message_handler(commands=["stop"])
def echo_stop(msg):
    global start_status
    start_status = False
    bot.reply_to(msg, f"Покеда, {msg.from_user.first_name} ✌️")


@bot.message_handler(content_types=["text", "doc", "photo", "sticker"])
def echo_messages(msg):
    global cache, start_time
    if not start_status:
        cache, start_time = logger.logger(msg, answer="Bot is not activated", cache=cache, start_time=start_time)
        welcome_text = f"Привет, {msg.from_user.first_name}. Я рекомендательный бот. Для начала общения введите /start."
        bot.send_message(msg.chat.id, welcome_text)


bot.set_update_listener(main)
bot.polling(none_stop=True)


while True:
    pass
