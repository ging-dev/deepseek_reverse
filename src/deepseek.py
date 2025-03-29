import base64
from contextlib import contextmanager
import json
from curl_cffi.requests import Session
from typing import cast, TypedDict, ContextManager, Iterator, Literal, overload
from ._internal import DeepSeekHash

_deepseek_hash = DeepSeekHash()

CHAT_TEMPLATE = "{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true, is_last_user=false) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}<｜end▁of▁sentence｜>{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{%- set ns.is_first = false -%}{%- set ns.is_last_user = true -%}{{'<｜User｜>' + message['content']}}{% if ns.is_last_user %}{% if not loop.last %}<｜Assistant｜>{% endif %}{% endif %}{%- endif %}{%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{%- endif %}{%- set ns.is_first = false %}{%- set ns.is_tool = false -%}{%- set ns.is_output_first = true %}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if message['content'] is none %}{{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- else %}{{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none)%}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_last_user = false -%}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}"


class Message(TypedDict):
    role: str
    content: str


@overload
def completion(
    token: str, messages: list[Message], stream: Literal[True]
) -> ContextManager[Iterator[str]]: ...


@overload
def completion(
    token: str, messages: list[Message], stream: Literal[False]
) -> ContextManager[str]: ...


@overload
def completion(
    token: str, messages: list[Message], stream: bool
) -> ContextManager[Iterator[str] | str]: ...


@contextmanager
def completion(token: str, messages: list[Message], stream: bool = False):
    from jinja2 import Environment

    prompt = Environment().from_string(CHAT_TEMPLATE).render(messages=messages)
    session = Session(
        base_url="https://chat.deepseek.com/api/v0/",
        impersonate="chrome",
        headers={
            "authorization": f"Bearer {token}",
            "x-client-locale": "en_US",
            "x-app-version": "20241129.1",
            "x-client-version": "1.0.0-always",
            "x-client-platform": "web",
        },
    )

    def do_json_request(url: str, json: dict) -> dict:
        response = session.post(url, json=json)
        result = response.json()
        if error := result["msg"]:
            raise RuntimeError(error)

        return result["data"]["biz_data"]

    session_id = do_json_request(
        "chat_session/create", json={"character_id": None}
    ).get("id")
    challenge = do_json_request(
        "chat/create_pow_challenge", json={"target_path": "/api/v0/chat/completion"}
    ).get("challenge")
    challenge["answer"] = _deepseek_hash.calculate_hash(**challenge)

    try:
        response = session.post(
            "chat/completion",
            json={
                "chat_session_id": session_id,
                "parent_message_id": None,
                "prompt": prompt,
                "ref_file_ids": [],
                "thinking_enabled": False,
                "search_enabled": False,
            },
            headers={
                "x-ds-pow-response:": base64.b64encode(json.dumps(challenge).encode())
            },
            stream=True,
        )

        def generate():
            line: bytes
            for line in response.iter_lines():
                line = line.decode()
                if ":" in line and "{" in line:
                    chunk = json.loads(line.split(":", 1)[1])
                    try:
                        yield cast(str, chunk["choices"][0]["delta"]["content"])
                    except:
                        pass

        if stream:
            yield generate()
        else:
            buffer = ""
            for chunk in generate():
                buffer += chunk
            yield buffer
    finally:
        do_json_request("chat_session/delete", json={"chat_session_id": session_id})
        session.close()
