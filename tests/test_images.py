"""Deciding whether an image is one the model can look at.

The fixtures are hand-built byte headers rather than Pillow output: Pillow is the
dependency this module exists in order not to have, and the thing under test is
precisely the header parsing, so writing the headers by hand *is* the test.
(Checked against real Pillow-produced PNG/JPEG/GIF/WebP/BMP/TIFF/ICO files.)
"""

import struct
import zlib

import pytest

from babash.client.images import MAX_PIXELS, ImageError, describe, load


def png(width: int, height: int, pad: int = 0) -> bytes:
    def chunk(tag: bytes, body: bytes) -> bytes:
        payload = tag + body
        return struct.pack(">I", len(body)) + payload + struct.pack(">I", zlib.crc32(payload))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00" * (width * 3 + 1) * height + b"\x00" * pad))
        + chunk(b"IEND", b"")
    )


def jpeg(width: int, height: int) -> bytes:
    # A JPEG's dimensions live in a Start-Of-Frame marker at no fixed offset —
    # the segments before it vary. Put a JFIF APP0 in front so the parser has to
    # walk the chain rather than peek at a constant position.
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x00\x11\x00"
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00" * 7


def webp(width: int, height: int) -> bytes:
    body = b"WEBPVP8 " + struct.pack("<I", 10) + b"\x00" * 6 + struct.pack("<HH", width, height)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_reads_dimensions_from_every_supported_format() -> None:
    for data, fmt in (
        (png(320, 200), "png"),
        (jpeg(320, 200), "jpeg"),
        (gif(320, 200), "gif"),
        (webp(320, 200), "webp"),
    ):
        image = load(data)
        assert image.format == fmt
        assert (image.width, image.height) == (320, 200), fmt


def test_format_comes_from_the_bytes_not_the_name() -> None:
    """read_image takes no extension at all — but the store it reads through
    doesn't care what a file is called either, so a mislabelled file must still
    be read as what it actually is."""
    assert load(jpeg(10, 10)).format == "jpeg"


def test_a_format_the_model_cannot_see_says_how_to_convert_it() -> None:
    """Claude's vision accepts exactly four formats. Everything else is an
    invalid_request_error raised far from here — but babash is a shell, and the
    fix is one command the agent can already run."""
    with pytest.raises(ImageError, match="sips|magick"):
        load(b"BM" + b"\x00" * 64)  # BMP

    with pytest.raises(ImageError, match="HEIC"):
        load(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 32)

    with pytest.raises(ImageError, match="rasterise|read_file"):
        load(b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>')


def test_an_oversized_image_is_refused_with_the_command_that_fixes_it() -> None:
    """Past 8000px the API rejects it outright — and the model downsamples to
    2576px anyway, so scaling down loses nothing."""
    with pytest.raises(ImageError, match="over the 8000x8000 limit"):
        load(png(MAX_PIXELS + 1, 10))


def test_a_large_but_valid_image_warns_about_downsampling() -> None:
    image = load(png(4000, 100))
    note = describe(image)
    assert note is not None
    assert "2576" in note


def test_an_image_within_the_model_resolution_gets_no_note() -> None:
    assert describe(load(png(800, 600))) is None


def test_bytes_that_are_not_an_image_at_all() -> None:
    with pytest.raises(ImageError, match="read_file|read_document"):
        load(b"just some text")
