import json
import struct
from contextlib import suppress
from typing import Optional, cast
from wasmtime import Memory
from wasmtime.loader import store
from .sha3 import memory, __wbindgen_export_0 as export, __wbindgen_add_to_stack_pointer as add_to_stack_pointer, wasm_solve  # type: ignore

memory = cast(Memory, memory).data_ptr(store)


class DeepSeekHash:
    @staticmethod
    def encode_string(text: str):
        encoded = text.encode()
        encoded_len = len(encoded)
        ptr = export(encoded_len, 1)
        for i, char_code in enumerate(encoded):
            memory[ptr + i] = char_code

        return ptr, encoded_len

    @staticmethod
    def calculate_hash(
        algorithm: str,
        challenge: str,
        salt: str,
        difficulty: int,
        expire_at: int,
        **kwargs,
    ) -> Optional[int]:
        if algorithm != "DeepSeekHashV1":
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        prefix = f"{salt}_{expire_at}_"

        try:
            # Allocate stack space
            retptr = add_to_stack_pointer(-16)

            # Get encoded pointers and lengths
            ptr0, len0 = DeepSeekHash.encode_string(challenge)
            ptr1, len1 = DeepSeekHash.encode_string(prefix)

            # Call the WASM function
            wasm_solve(retptr, ptr0, len0, ptr1, len1, float(difficulty))

            # Read 4-byte status and 8-byte float value
            status = struct.unpack("<i", bytes(memory[retptr : retptr + 4]))[0]
            value = struct.unpack("<d", bytes(memory[retptr + 8 : retptr + 16]))[0]

            if status == 0:
                return None

            return int(value)

        finally:
            # Free stack space
            add_to_stack_pointer(16)


def parse_line(line: bytes) -> Optional[str]:
    data = line.decode()
    if "{" in data:
        chunk = json.loads(data.split(":", 1)[1])
        with suppress(KeyError):
            return chunk["choices"][0]["delta"]["content"]

    return None
