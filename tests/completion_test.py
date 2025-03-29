def test_completion():
    from deepseek_reverse import completion
    with completion(
        messages=[
            {"role": "user", "content": "Xin chào"},
        ],
        stream=False
    ) as r:
        assert isinstance(r, str)
