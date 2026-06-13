import io
import itertools as it
import os
import tempfile
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from math import ceil, sqrt

import av
import duckdb
import numpy as np
import tqdm
from einops import rearrange
from simplejpeg import decode_jpeg


def decode_arrow_jpeg(x):
    return decode_jpeg(np.frombuffer(x.as_buffer(), dtype=np.uint8))


def encode_episode(episode: duckdb.DuckDBPyRelation, *, fps: int, batch_size: int = 32):
    """
    Encode a video of all the cameras available for a specific episode.

    fps: the recording frequency.
    batch_size: how many frames are decoded from jpeg before encoding to a video stream.
    """
    uuid = duckdb.sql("SELECT DISTINCT uuid FROM episode").fetchone()
    assert uuid
    uuid = uuid[0]
    frames = [f"obs.frames.{k}.rgb.data AS {k}" for k, _ in episode.select("obs.frames").types[0].children]
    episode_length = episode.count("*").fetchone()
    assert episode_length is not None
    episode_length = episode_length[0]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as exc, io.BytesIO() as buffer:
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=fps)
        width = height = None
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "threads": str(os.cpu_count())}
        for batch in tqdm.tqdm(
            episode.select(",".join(frames)).to_arrow_reader(batch_size=batch_size),
            total=ceil(episode_length / batch_size),
            unit_scale=batch_size,
            desc=f"Encoding {uuid}",
            unit="frame",
        ):
            decoded = exc.map(decode_arrow_jpeg, it.chain.from_iterable(batch))
            decoded = np.stack(tuple(decoded))
            decoded = rearrange(decoded, "(n b) h w c -> b n h w c", n=len(frames))
            b, n_images, h, w, c = decoded.shape
            col = ceil(sqrt(n_images))
            row = ceil(n_images / col)
            if width is None and height is None:
                stream.width = width = col * w
                stream.height = height = row * h
            n_slots = row * col
            if n_slots != n_images:
                pad = np.zeros((b, n_slots - n_images, h, w, c), dtype=decoded.dtype)
                decoded = np.concatenate([decoded, pad], axis=1)
            decoded = rearrange(decoded, "b (row col) h w c -> b (row h) (col w) c", row=row, col=col)
            video_frames = exc.map(lambda x: av.VideoFrame.from_ndarray(x, format="rgb24"), decoded)
            packets = (stream.encode(frame) for frame in video_frames)
            for packet in it.chain.from_iterable(packets):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        buffer.seek(0)
        return buffer.read()


def sample_uuid(rel: duckdb.DuckDBPyRelation) -> str:
    _ = rel
    ret = duckdb.sql(
        """
        SELECT uuid
        FROM (SELECT DISTINCT uuid FROM rel)
        USING SAMPLE RESERVOIR(1 ROWS)
        """
    ).fetchone()
    assert ret is not None
    return ret[0]



if __name__ == "__main__":
    ds = duckdb.from_parquet("data") # Change path to dataset here
    print(
        duckdb.sql(
        """
            SELECT count(DISTINCT uuid) AS "successful episodes"
            FROM ds
            WHERE success = true
        """)
    )
    # if you want to see e.g. successful episodes only.
    # uuid = sample_uuid(ds.filter("success=true"))
    # Cf. duckdb docs for more.
    uuid = sample_uuid(ds)
    episode = ds.filter(f"uuid = '{uuid}'").order("step")
    video_bytes = encode_episode(episode, fps=30)
    # Displays the video in your browser. The file is automatically deleted.
    with tempfile.NamedTemporaryFile("wb", suffix=".mp4") as f:
        f.write(video_bytes)
        webbrowser.open(f.name)
        # wait so your browser has enough time to read your video
        time.sleep(1)
