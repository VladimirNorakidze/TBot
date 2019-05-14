import os
import requests
import time
import random
import telebot
import logger
import analyzer

TOKEN = "TOKEN"

start_time = time.time()

answers_for_me = ["Твои шутки - отпад 🤣", "Приветики 😘", "Я так рад тебя видеть 😍", "Блин, клево)",
                  "А расскажи еще что-нибудь", "Лол", "Ахахаха", "Ору 😅", "Го еграть? :3", "👍"]
cache = []

bot = telebot.TeleBot(TOKEN)
start_status = False
with open("botpid.txt", "w") as file:
    file.write(str(os.getpid()))


def main(messages):
    """
    When new messages arrive TeleBot will call this function.
    """
    global cache, start_time
    if messages[-1].text != "/stop":
        for m in messages:
            chatid = m.chat.id
            if start_status:
                if m.content_type == "text":
                    answer = random.choice(answers_for_me)
                    cache, start_time = logger.logger(m, answer, cache, start_time)
                    print(wa.wa_analyzer(m.text))
                    bot.send_message(chat_id=chatid, text=answer)
                elif m.content_type == "photo":
                    cache, start_time = logger.logger(m, "", cache, start_time)
                    bot.send_message(chat_id=chatid, text="😮")
                    time.sleep(1)
                    bot.send_message(chat_id=chatid, text="Это что... Картинка???")
                    # file = bot.get_file(file_id=m.photo[-1].file_id)
                    # picture = requests.get(f"https://api.telegram.org/file/bot{TOKEN}/{file.file_path}")
                    # with open("test.jpg", "wb") as pict:
                    #     pict.write(picture.content)


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
