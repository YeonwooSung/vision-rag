from pathlib import Path


def read_prompt_from_plain_file(filename: str) -> str:
    filepath = Path(filename)
    with filepath.open(mode="r") as prompt:
        return prompt.read()
