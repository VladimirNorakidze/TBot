import os
import time
FILENAME = "log.csv"


def log_to_file(cache):
    if not os.path.exists(FILENAME):
        head = "datetime,chat_id,user_id,message_type,text_or_file_id,answer\n"
        with open(FILENAME, "w") as f:
            f.write(head)
    with open(FILENAME, "a") as file:
        for string in cache:
            file.write(str(string) + "\n")
        return []


def logger(msg, answer, cache, start_time):
    chat_id = msg.chat.id
    user_id = msg.from_user.id
    if msg.content_type == "text":
        input_msg = msg.text
    elif msg.content_type == "photo":
        input_msg = msg.json["photo"][-1]["file_id"]
    elif msg.content_type == "document" and ("image" in msg.document.mime_type):
        input_msg = msg.document.file_id
    else:
        input_msg = msg.document.mime_type
    datetime = time.strftime("%d.%m.%y %X", time.localtime())
    res = f"[{datetime}]," + str(chat_id) + "," + str(user_id) + \
          f",{msg.content_type},\"" + str(input_msg) + "\",\"" + answer + "\""
    print(res)
    cache.append(res)
    if time.time() - start_time > 10:
        cache = log_to_file(cache)
        start_time = time.time()
    return cache, start_time

