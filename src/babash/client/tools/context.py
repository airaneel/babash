"""Core types and utilities shared across tool modules."""

import base64
import mimetypes
import os
from dataclasses import dataclass
from os.path import expanduser
from typing import Literal, Optional

from pydantic import BaseModel

from ...types_ import Console
from ..bash_state import BashState
from ..encoder import get_default_encoder


@dataclass
class Context:
    bash_state: BashState
    console: Console


MEDIA_TYPES = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ImageData(BaseModel):
    media_type: MEDIA_TYPES
    data: str

    @property
    def dataurl(self) -> str:
        return f"data:{self.media_type};base64," + self.data


default_enc = get_default_encoder()


def expand_user(path: str) -> str:
    if not path or not path.startswith("~"):
        return path
    return expanduser(path)


def truncate_if_over(content: str, max_tokens: Optional[int]) -> str:
    if max_tokens and max_tokens > 0:
        tokens = default_enc.encoder(content)
        n_tokens = len(tokens)
        if n_tokens > max_tokens:
            content = (
                default_enc.decoder(tokens[: max(0, max_tokens - 100)])
                + "\n(...truncated)"
            )
    return content


def read_image_from_shell(file_path: str, context: Context) -> ImageData:
    file_path = expand_user(file_path)

    if not os.path.isabs(file_path):
        file_path = os.path.join(context.bash_state.cwd, file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")

    with open(file_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_type = mimetypes.guess_type(file_path)[0]
        return ImageData(media_type=image_type, data=image_b64)  # type: ignore


def save_out_of_context(content: str, suffix: str) -> str:
    from tempfile import NamedTemporaryFile
    file_path = NamedTemporaryFile(delete=False, suffix=suffix).name
    with open(file_path, "w") as f:
        f.write(content)
    return file_path
