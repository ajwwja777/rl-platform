"""Lossless, bounded inference transport; image and action arrays remain exact."""
import io
import zlib
import numpy as np


def encode(values, compress=False):
    stream = io.BytesIO()
    np.savez(stream, **values)
    body = stream.getvalue()
    return (zlib.compress(body, 1), "deflate") if compress else (body, "identity")


def decode(body, encoding="identity"):
    if encoding == "deflate":
        decoder = zlib.decompressobj()
        body = decoder.decompress(body, 2 * 1024 * 1024)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError("Invalid compressed inference body")
    elif encoding != "identity":
        raise ValueError("Unsupported encoding")
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}
