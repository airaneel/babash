"""Deciding whether an image is one the model can actually look at.

Claude's vision accepts exactly four formats — JPEG, PNG, GIF and WebP — with a
hard 8000x8000 pixel ceiling and a 10MB-once-base64-encoded size limit. Anything
else is an `invalid_request_error` from the API, raised far from here, phrased in
terms the agent has no way to act on.

So the checks happen here, before the bytes are sent, and each failure says what
to run instead. babash *is* a shell: an image it can't show is one conversion
away, and the agent can already do the conversion.

Formats are recognised by their leading bytes, never their extension — the same
rule as documents.py, and for the same reason: a `.png` that is really a HEIC is
a silent failure, while a file we decline to send is a loud one.
"""

import struct
from dataclasses import dataclass
from typing import Optional

# The only four Claude's vision accepts.
SUPPORTED = ("png", "jpeg", "gif", "webp")

# The API rejects anything larger outright.
MAX_PIXELS = 8000

# The documented limit is 10MB *after* base64 encoding, which inflates by 4/3.
MAX_RAW_BYTES = 7_500_000


class ImageError(Exception):
    """The image can't be shown as-is — and the message says what to do."""


@dataclass(frozen=True)
class Image:
    data: bytes
    format: str
    width: int
    height: int


def _png_size(data: bytes) -> tuple[int, int]:
    # IHDR is always the first chunk: 8-byte signature, 4-byte length, 4-byte
    # type, then width and height as big-endian uint32.
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _gif_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """JPEG keeps its dimensions in a Start-Of-Frame marker, which sits at no
    fixed offset — the marker segments before it vary in number and length, so
    the only way to find it is to walk the chain."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        # SOF0..SOF15, excluding the DHT/JPG/DAC markers interleaved among them.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        i += 2 + length
    raise ImageError("Malformed JPEG: no frame header found.")


def _webp_size(data: bytes) -> tuple[int, int]:
    chunk = data[12:16]
    if chunk == b"VP8X":
        # 24-bit little-endian, stored as (width - 1).
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    if chunk == b"VP8L":
        (bits,) = struct.unpack("<I", data[21:25])
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    raise ImageError("Malformed WebP: unrecognised chunk header.")


# What each convertible format looks like, and what turns it into something the
# model can see. `sips` ships with macOS; ImageMagick is the Linux answer.
_CONVERTIBLE: dict[str, tuple[bytes, str]] = {
    "BMP": (b"BM", "bmp"),
    "TIFF": (b"II*\x00", "tiff"),
    "TIFF (big-endian)": (b"MM\x00*", "tiff"),
    "ICO": (b"\x00\x00\x01\x00", "ico"),
}


def _convert_advice(kind: str) -> str:
    return (
        f"{kind} is not one of the four formats the model can see "
        f"(PNG, JPEG, GIF, WebP). Convert it in the shell, then read the result:\n"
        f"  macOS:  sips -s format png <file> --out /tmp/converted.png\n"
        f"  Linux:  magick <file> /tmp/converted.png   (or: convert <file> …)\n"
        f"then: read_image(file_path='/tmp/converted.png')"
    )


def _sniff(data: bytes) -> str:
    """The image's format, from its leading bytes. Raises with conversion advice
    for images the model cannot see, rather than letting the API reject them."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"

    # Formats we recognise well enough to say what to do about.
    if data.startswith(b"\x00\x00\x00") and data[4:12] in (b"ftypheic", b"ftypheix"):
        raise ImageError(_convert_advice("HEIC (an iPhone photo)"))
    if data.startswith(b"\x00\x00\x00") and data[4:12] == b"ftypavif":
        raise ImageError(_convert_advice("AVIF"))
    if data.lstrip()[:5] in (b"<svg ", b"<?xml"):
        raise ImageError(
            "SVG is vector text, not an image the model can see. Either read it as "
            "text with read_file, or rasterise it:\n"
            "  magick <file> /tmp/out.png   (or: rsvg-convert <file> -o /tmp/out.png)\n"
            "then: read_image(file_path='/tmp/out.png')"
        )
    for kind, (magic, _) in _CONVERTIBLE.items():
        if data.startswith(magic):
            raise ImageError(_convert_advice(kind))

    raise ImageError(
        "Not an image babash recognises. If it is text, use read_file; if it is a "
        "PDF or Office document, use read_document."
    )


_SIZERS = {
    "png": _png_size,
    "jpeg": _jpeg_size,
    "gif": _gif_size,
    "webp": _webp_size,
}


def load(data: bytes) -> Image:
    """Check an image is one the model can actually look at, and say so if not.

    Every rejection here is one the API would otherwise make itself — but it
    would make it as an opaque `invalid_request_error`, several layers away from
    the agent, with no hint that the fix is one shell command.
    """
    fmt = _sniff(data)
    width, height = _SIZERS[fmt](data)

    if width > MAX_PIXELS or height > MAX_PIXELS:
        raise ImageError(
            f"This image is {width}x{height}, over the {MAX_PIXELS}x{MAX_PIXELS} limit. "
            f"Scale it down first — the model downsamples anything past ~2576px on the "
            f"long edge anyway, so nothing is lost:\n"
            f"  macOS:  sips -Z 2000 <file> --out /tmp/small.png\n"
            f"  Linux:  magick <file> -resize 2000x2000 /tmp/small.png\n"
            f"then: read_image(file_path='/tmp/small.png')"
        )

    if len(data) > MAX_RAW_BYTES:
        raise ImageError(
            f"This image is {len(data) // 1_000_000}MB, over the limit (10MB once "
            f"base64-encoded, which is what the API counts). Compress or scale it:\n"
            f"  macOS:  sips -Z 2000 -s format jpeg <file> --out /tmp/small.jpg\n"
            f"  Linux:  magick <file> -resize 2000x2000 -quality 85 /tmp/small.jpg\n"
            f"then: read_image(file_path='/tmp/small.jpg')"
        )

    return Image(data=data, format=fmt, width=width, height=height)


def describe(image: Image) -> Optional[str]:
    """Anything worth telling the agent about how the model will see this."""
    if max(image.width, image.height) > 2576:
        return (
            f"({image.width}x{image.height} — the model downsamples to 2576px on the "
            f"long edge, so fine detail may be lost. Crop to the region of interest "
            f"if you need to read something small.)"
        )
    return None
