import os
import time
import random
import multiprocessing as mp
import telebot
import logger
import word_analyzer as wa

TOKEN = "815730867:AAEvONIOpcwNZDdhxzoqNw8-gV_LCV9oT6I"

start_time = time.time()

elite_list = [93500076, 203417470, 318022478]
answers = ["Пошел нахер!", "Иди в жопу", "Сам мудак", "Ты это мне?", "Все, пиздец!",
                   "Засужу!", "ЪУЪ С*КА", "Че ты меня бесишь?", "Паш, я не могу понять, он прикалывается что ли?", "🖕"]
answers_for_me = ["Твои шутки - отпад 🤣", "Приветики 😘", "Я так рад тебя видеть 😍", "Блин, клево)",
                  "А расскажи еще что-нибудь", "Лол", "Ахахаха", "Ору 😅", "Го еграть? :3", "👍"]
cache = []

bot = telebot.TeleBot(TOKEN)
start_status = False
with open("botpid.txt", "w") as file:
    file.write(str(os.getpid()))


def log_to_file(cache):
    with open("log.txt", "a") as file:
        for string in cache:
            file.write(str(string) + "\n")
        return []


def cashing(msg, answer):
    global cache, start_time
    chat_id = msg.chat.id
    user_id = msg.from_user.id
    text = msg.text
    datetime = time.strftime("%d/%m/%y %X", time.localtime())
    res = f"{datetime}," + str(chat_id) + "," + str(user_id) + ",\"" + str(text) + "\",\"" + answer + "\""
    print(res)
    cache.append(res)
    if time.time() - start_time > 10:
        cache = log_to_file(cache)
        start_time = time.time()


def main(messages):
    """
    When new messages arrive TeleBot will call this function.
    """
    if messages[-1].text != "/stop":
        for m in messages:
            chatid = m.chat.id
            uid = m.from_user.id
            if start_status:
                if m.content_type == 'text':
                    if uid in elite_list:
                        answer = random.choice(answers_for_me)
                    else:
                        answer = random.choice(answers)
                    cashing(m, answer)
                    print(wa.main(m.text))
                    bot.send_message(chatid, answer)
    # else:
    #     listen_start(messages)


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
    if not start_status:
        welcome_text = f"Привет, {msg.from_user.first_name}. Я рекомендательный бот. \
        Для начала общения введите /start."
        bot.send_message(msg.chat.id, welcome_text)


bot.set_update_listener(main)
bot.polling(none_stop=True)


while True:  # Don't let the main Thread end.
    pass
