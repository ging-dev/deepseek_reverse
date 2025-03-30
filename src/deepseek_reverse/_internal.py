import struct
from pathlib import Path
from wasmtime import Store, Module, Instance, Memory, Func
from typing import cast, Optional


class DeepSeekHash:
    def __init__(self):
        self.store = Store()
        self.wasm_exports = Instance(
            self.store,
            Module.from_file(
                self.store.engine, Path(__file__).resolve().parent / "sha3.wasm"
            ),
            [],
        ).exports(self.store)

    def _get_memory(self):
        return cast(Memory, self.wasm_exports["memory"])

    def _get_func(self, name: str) -> Func:
        return cast(Func, self.wasm_exports[name])

    def encode_string(self, text: str):
        encoded = text.encode()
        encoded_len = len(encoded)
        ptr = self._get_func("__wbindgen_export_0")(self.store, encoded_len, 1)
        memory = self._get_memory().data_ptr(self.store)
        for i, char_code in enumerate(encoded):
            memory[ptr + i] = char_code

        return ptr, encoded_len

    def calculate_hash(
        self,
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
            retptr = self._get_func("__wbindgen_add_to_stack_pointer")(self.store, -16)

            # Get encoded pointers and lengths
            ptr0, len0 = self.encode_string(challenge)
            ptr1, len1 = self.encode_string(prefix)

            # Call the WASM function
            self._get_func("wasm_solve")(
                self.store, retptr, ptr0, len0, ptr1, len1, float(difficulty)
            )

            # Get the return result
            data = self._get_memory().data_ptr(self.store)

            # Read 4-byte status and 8-byte float value
            status = struct.unpack("<i", bytes(data[retptr : retptr + 4]))[0]
            value = struct.unpack("<d", bytes(data[retptr + 8 : retptr + 16]))[0]

            if status == 0:
                return None

            return int(value)

        finally:
            # Free stack space
            self._get_func("__wbindgen_add_to_stack_pointer")(self.store, 16)
