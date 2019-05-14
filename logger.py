import time


def log_to_file(cache):
    with open("log.csv", "a") as file:
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
    else:
        input_msg = "Unknown content type..."
    datetime = time.strftime("%d/%m/%y %X", time.localtime())
    res = f"{datetime}," + str(chat_id) + "," + str(user_id) + \
          f",{msg.content_type},\"" + str(input_msg) + "\",\"" + answer + "\""
    print(res)
    cache.append(res)
    if time.time() - start_time > 10:
        cache = log_to_file(cache)
        start_time = time.time()
    return cache, start_time

