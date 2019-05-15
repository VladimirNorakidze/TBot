import os
import requests
import time
import random
import telebot
import logger
import analyzer as a

TOKEN = "TOKEN"

start_time = time.time()
cache = []

bot = telebot.TeleBot(TOKEN)
start_status = True
with open("botpid.txt", "w") as file:
    file.write(str(os.getpid()))


def text_msg(msg):
    global cache, start_time
    chatid = msg.chat.id
    answer = a.wa_analyzer(msg.text)
    cache, start_time = logger.logger(msg, answer, cache, start_time)
    bot.send_message(chat_id=chatid, text=answer)


def img_proc(fileid):
    file = bot.get_file(file_id=fileid)
    response = requests.get(f"https://api.telegram.org/file/bot{TOKEN}/{file.file_path}")
    return str(a.img_analyzer(response))


def photo_msg(msg):
    global cache, start_time
    chatid = msg.chat.id
    cache, start_time = logger.logger(msg, "", cache, start_time)
    bot.send_message(chat_id=chatid, text="😮")
    time.sleep(1)
    bot.send_message(chat_id=chatid, text="Это что... Картинка???")
    bot.send_message(chat_id=chatid, text="И что с ней делать? Перевести в байты? Ну тогда получи!!!")
    ans = img_proc(msg.photo[-1].file_id)
    time.sleep(1)
    bot.send_message(chat_id=chatid, text=ans)


def doc_msg(msg):
    global cache, start_time
    chatid = msg.chat.id
    cache, start_time = logger.logger(msg, "", cache, start_time)
    bot.send_message(chat_id=chatid, text="😮")
    time.sleep(1)
    bot.send_message(chat_id=chatid, text="Документ, лол")
    ans = img_proc(msg.document.file_id)
    time.sleep(1)
    bot.send_message(chat_id=chatid, text=ans)


def main(messages):
    """
    When new messages arrive TeleBot will call this function.
    """
    global cache, start_time
    if messages[-1].text != "/stop":
        for m in messages:
            if start_status:
                if m.content_type == "text":
                    text_msg(m)
                elif m.content_type == "photo":
                    photo_msg(m)
                elif m.content_type == "document" and "image" in m.document.mime_type:
                    doc_msg(m)
                else:
                    cache, start_time = logger.logger(m, "Unknown type...", cache, start_time)
                    bot.reply_to(m, text="Я не понимаю, что это 😭😭😭")
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


@bot.message_handler()
def echo_messages(msg):
    global cache, start_time
    if not start_status:
        cache, start_time = logger.logger(msg, answer="Bot is not activated", cache=cache, start_time=start_time)
        welcome_text = f"Привет, {msg.from_user.first_name}. Я рекомендательный бот. \
        Для начала общения введите /start."
        bot.send_message(msg.chat.id, welcome_text)


bot.set_update_listener(main)
bot.polling(none_stop=True)


while True:
    pass
