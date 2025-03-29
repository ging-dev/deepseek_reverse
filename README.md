## Example

```python
from deepseek_reverse import completion
def main():
    with completion(
        messages=[
            {"role": "user", "content": "Tình hình GDP Việt Nam 2024"},
        ],
        stream=True,
        search_enabled=True,
        token="...",
    ) as stream:
        for chunk in stream:
            print(chunk, end="", flush=True)
if __name__ == "__main__":
    main()
```
