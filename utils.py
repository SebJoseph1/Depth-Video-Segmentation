import json
import time
import os

def log_to_json(type: str, value: [str, list, dict, int, float]):
    to_serialize = {'type': type, 'value': value}
    to_serialize['time'] = time.time()
    to_write = json.dumps(to_serialize) + "\n"
    with open("log_" + str(os.getpid()) + ".txt", "a") as log:
        log.write(to_write)

