import os
import time
FILENAME_T = "log_Tbot.csv"
FILENAME_VK = "log_VKbot.csv"


def log_to_file(cache, bot_name):
    filename = FILENAME_T if bot_name == "Tbot" else FILENAME_VK
    if not os.path.exists(filename):
        head = "datetime,user_id,message_type,text_or_file_id,answer\n"
        with open(filename, "w") as f:
            f.write(head)
    with open(filename, "a") as file:
        for string in cache:
            file.write(str(string) + "\n")
        return []


def logger(msg, answer, cache, start_time):
    chat_id = msg.chat.id
    user_id = msg.from_user.id
    if msg.content_type == "text":
        input_msg = msg.text
        msg_mime_type = ""
    elif msg.content_type == "photo":
        input_msg = msg.json["photo"][-1]["file_id"]
        msg_mime_type = ""
    elif msg.content_type == "document" and ("image" in msg.document.mime_type):
        input_msg = msg.document.file_id
        msg_mime_type = msg.document.mime_type
    else:
        input_msg = "Unknown type..."
        msg_mime_type = msg.document.mime_type
    datetime = time.strftime("%d.%m.%y %X", time.localtime())
    res = [f"[{datetime}]", str(chat_id), str(user_id), str(msg.content_type),
           str(msg_mime_type), "\"" + str(input_msg) + "\"", str(answer)]
    res = ",".join(res)
    print(res)
    cache.append(res)
    if time.time() - start_time > 10:
        cache = log_to_file(cache, bot_name="Tbot")
        start_time = time.time()
    return cache, start_time


def logger_vk(msg, answer, cache, start_time):
    user_id = msg.user_id
    if msg.text:
        msg_type = "text"
        input_msg = msg.text
    else:
        msg_type = msg.attachments["attach1_type"]
        input_msg = msg.attachments["attach1"]
    datetime = time.strftime("%d.%m.%y %X", time.localtime())
    res = [f"[{datetime}]", str(user_id), str(msg_type),
           "\"" + str(input_msg) + "\"", str(answer)]
    res = ",".join(res)
    print(res)
    cache.append(res)
    if time.time() - start_time > 10:
        cache = log_to_file(cache, bot_name="VK")
        start_time = time.time()
    return cache, start_time