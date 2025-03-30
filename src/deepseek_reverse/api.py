import base64
import json
import os
from contextlib import contextmanager, asynccontextmanager
from curl_cffi.requests import AsyncSession, Session, Response
from typing import (
    overload,
    Any,
    AsyncIterator,
    AsyncContextManager,
    ContextManager,
    Iterator,
    Literal,
    Optional,
    TypedDict,
)
from ._internal import DeepSeekHash, parse_line
from jinja2 import Environment

CHAT_TEMPLATE = "{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true, is_last_user=false) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}<｜end▁of▁sentence｜>{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{%- set ns.is_first = false -%}{%- set ns.is_last_user = true -%}{{'<｜User｜>' + message['content']}}{% if ns.is_last_user %}{% if not loop.last %}<｜Assistant｜>{% endif %}{% endif %}{%- endif %}{%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{%- endif %}{%- set ns.is_first = false %}{%- set ns.is_tool = false -%}{%- set ns.is_output_first = true %}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if message['content'] is none %}{{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- else %}{{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none)%}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_last_user = false -%}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}"
_prompt = Environment().from_string(CHAT_TEMPLATE)
_deepseek_hash = DeepSeekHash()


class Message(TypedDict):
    role: str
    content: str


@overload
def completion(
    messages: list[Message],
    stream: Literal[True],
    search_enabled: bool = False,
    thinking_enabled: bool = False,
    token: Optional[str] = None,
) -> ContextManager[Iterator[str]]: ...


@overload
def completion(
    messages: list[Message],
    stream: Literal[False],
    search_enabled: bool = False,
    thinking_enabled: bool = False,
    token: Optional[str] = None,
) -> ContextManager[str]: ...


@overload
def completion(
    messages: list[Message],
    stream: bool,
    search_enabled: bool = False,
    thinking_enabled: bool = False,
    token: Optional[str] = None,
) -> ContextManager[Iterator[str] | str]: ...


@contextmanager
def completion(
    messages: list[Message],
    stream: bool,
    search_enabled: bool = False,
    thinking_enabled: bool = False,
    token: Optional[str] = None,
):
    token = token or os.environ.get("DEEPSEEK_TOKEN")
    if not token:
        raise RuntimeError("Missing token")
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

    def do_json_request(url: str, json: dict) -> dict[str, Any]:
        response = session.post(url, json=json)
        result = response.json()
        if error := result["msg"]:
            raise RuntimeError(error)

        return result["data"]["biz_data"]

    session_id: str = do_json_request(
        "chat_session/create", json={"character_id": None}
    )["id"]
    challenge: dict = do_json_request(
        "chat/create_pow_challenge", json={"target_path": "/api/v0/chat/completion"}
    )["challenge"]
    challenge["answer"] = _deepseek_hash.calculate_hash(**challenge)

    try:
        response = session.post(
            "chat/completion",
            json={
                "chat_session_id": session_id,
                "parent_message_id": None,
                "prompt": _prompt.render(messages=messages),
                "ref_file_ids": [],
                "thinking_enabled": thinking_enabled,
                "search_enabled": search_enabled,
            },
            headers={
                "x-ds-pow-response:": base64.b64encode(
                    json.dumps(challenge).encode()
                ).decode()
            },
            stream=True,
        )

        def generate() -> Iterator[str]:
            for line in response.iter_lines():
                if chunk := parse_line(line):
                    yield chunk

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


@overload
def acompletion(
    messages: list[Message],
    stream: Literal[True],
    search_enabled: bool = False,
    thinking_enabled: bool = False,
    token: Optional[str] = None,
) -> AsyncContextManager[AsyncIterator[str]]: ...


@overload
def acompletion(
    messages: list[Message],
    stream: Literal[False],
    search_enabled: bool = False,
    thinking_enabled: bool = False,
    token: Optional[str] = None,
) -> AsyncContextManager[str]: ...


@overload
def acompletion(
    messages: list[Message],
    stream: bool,
    search_enabled: bool = False,
    thinking_enabled: bool = False,
    token: Optional[str] = None,
) -> AsyncContextManager[AsyncIterator[str] | str]: ...


@asynccontextmanager
async def acompletion(
    messages: list[Message],
    stream: bool,
    search_enabled: bool = False,
    thinking_enabled: bool = False,
    token: Optional[str] = None,
):
    token = token or os.environ.get("DEEPSEEK_TOKEN")
    if not token:
        raise RuntimeError("Missing token")
    session: AsyncSession[Response] = AsyncSession(
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

    async def do_json_request(url: str, json: dict) -> dict[str, Any]:
        response = await session.post(url, json=json)
        result = response.json()
        if error := result["msg"]:
            raise RuntimeError(error)

        return result["data"]["biz_data"]

    session_id: str = (
        await do_json_request("chat_session/create", json={"character_id": None})
    )["id"]
    challenge: dict = (
        await do_json_request(
            "chat/create_pow_challenge", json={"target_path": "/api/v0/chat/completion"}
        )
    )["challenge"]
    challenge["answer"] = _deepseek_hash.calculate_hash(**challenge)

    try:
        response = await session.post(
            "chat/completion",
            json={
                "chat_session_id": session_id,
                "parent_message_id": None,
                "prompt": _prompt.render(messages=messages),
                "ref_file_ids": [],
                "thinking_enabled": thinking_enabled,
                "search_enabled": search_enabled,
            },
            headers={
                "x-ds-pow-response:": base64.b64encode(
                    json.dumps(challenge).encode()
                ).decode()
            },
            stream=True,
        )

        async def generate() -> AsyncIterator[str]:
            async for line in response.aiter_lines():
                if chunk := parse_line(line):
                    yield chunk

        if stream:
            yield generate()
        else:
            buffer = ""
            async for chunk in generate():
                buffer += chunk
            yield buffer
    finally:
        await do_json_request(
            "chat_session/delete", json={"chat_session_id": session_id}
        )
        await session.close()


__all__ = ["completion", "acompletion"]
