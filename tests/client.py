import json
import requests

URL = "http://localhost:8000/api/chat/completion"
MODEL = "blip2zh-chatglm-6b"
MESSAGES = [{"role": "user", "content": "Hello world!"}]


def debug():
    headers = {"accept": "application/json"}
    files = {
        "files": (
            "chat-overview.png",
            open("doc/img/chat-overview.png", "rb"),
            "image/png",
        )
    }
    data = {"data": json.dumps({"name": "foo", "point": 0.13, "is_accepted": False})}
    # params= {"name": "foo", "point": 20.5, "is_accepted": False}
    resp = requests.post(
        "http://localhost:8000/debug",
        headers=headers,
        # params=params,
        data=data,
        files=files,
    )
    print(resp.json())


def generate():
    print("generate")
    headers = {"accept": "application/json"}
    data = {"data": json.dumps({"model": MODEL, "messages": MESSAGES, "stream": False})}
    resp = requests.post(
        URL,
        headers=headers,
        data=data,
        files=None,
        stream=False,
    )
    print(resp.json())


def stream_generate():
    print("stream_generate")
    data = {"data": json.dumps({"model": MODEL, "messages": MESSAGES, "stream": True})}
    resp = requests.post(
        URL, data=data, stream=True
    )
    for line in resp.iter_lines():
        print(line)


def stream_generate_img():
    print("stream_generate_img")
    data = {"data": json.dumps({"model": MODEL, "messages": MESSAGES, "stream": True})}
    files = {
        "image": (
            "chat-overview.png",
            open("doc/img/chat-overview.png", "rb"),
            "image/png",
        )
    }
    resp = requests.post(URL, data=data, files=files, stream=True)
    for line in resp.iter_lines():
        print(line)


# debug()
stream_generate_img()
generate()
stream_generate()
