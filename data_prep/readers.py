"""Readers for bedside-monitor waveform exports.

GE Datex-Ohmeda S/5 (iCollect) ``*waves.asc``, as found in 01NVa_Dengue::

    11/01/2021 09:14:27<TAB>300<TAB>Hz<TAB>1610381667
    Time<TAB>ECG1<TAB>Pleth<TAB>Resp
    1610381667.000<TAB>0.707<TAB>-3.030<TAB>1.170
    ...

Line 1: local date/time (format varies: D/M/Y 24 h or M/D/Y 12 h), sampling
rate, unit, epoch seconds. Line 2: channel names (the impedance channel may be
called ``Resp`` or ``Resp100``). Column ``Time`` is epoch seconds.

Clock caveat: in every file checked, the header epoch equals the header
wall-clock time + 7 h, i.e. one of the two clocks is off by the Vietnam UTC
offset. Both are kept; check against the wearable devices (e.g. by HR
cross-correlation) before aligning across devices.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHANNEL_ALIASES = {"ECG1": "ecg", "Pleth": "ppg", "Resp": "resp", "Resp100": "resp"}
GAP_TOLERANCE = 1.5  # a gap is a time step larger than 1.5 sample periods


@dataclass
class Segment:
    """A gap-free stretch of a recording."""
    t0_epoch: float               # epoch seconds of the first sample (header clock)
    fs: float
    channels: dict = field(default_factory=dict)  # name -> float32 array

    @property
    def duration(self):
        return len(next(iter(self.channels.values()))) / self.fs


@dataclass
class Recording:
    path: str
    fs: float
    header_local_time: str        # raw, ambiguous format; kept for reference
    header_epoch: float
    segments: list = field(default_factory=list)


class CorruptFileError(ValueError):
    pass


def read_ge_waves(path, min_segment_s=30.0):
    """Read a GE ``*waves.asc`` file into gap-free segments.

    Raises :class:`CorruptFileError` for unreadable files (e.g. files that are
    entirely NUL bytes, which occur in this dataset).
    """
    with open(path, "rb") as fh:
        head = fh.read(512)
    if not head.strip(b"\x00\r\n\t "):
        raise CorruptFileError(f"{path}: file is empty or NUL bytes")
    lines = head.split(b"\n")
    first = lines[0].decode("latin1").strip().split("\t")
    try:
        fs = float(first[1])
        epoch = float(first[3])
        names = lines[1].decode("latin1").strip().split("\t")
    except (IndexError, ValueError) as err:
        raise CorruptFileError(f"{path}: unexpected header {first!r}") from err

    # float32 for signals keeps a 25-h file at ~1.5 GB in memory
    dtypes = {n: (np.float64 if n == "Time" else np.float32) for n in names}
    kw = dict(sep="\t", skiprows=1, engine="c", encoding="latin1", on_bad_lines="skip")
    try:
        df = pd.read_csv(path, dtype=dtypes, **kw)
    except ValueError:  # non-numeric cells somewhere: coerce them to NaN
        df = pd.read_csv(path, dtype=str, **kw).apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={c: CHANNEL_ALIASES.get(c.strip(), c.strip()) for c in df.columns})
    missing = {"Time", "ecg", "ppg", "resp"} - set(df.columns)
    if missing:
        raise CorruptFileError(f"{path}: missing columns {missing}")
    df = df.dropna(subset=["Time"])
    t = df["Time"].to_numpy()

    # Split at gaps / time reversals
    dt = np.diff(t)
    breaks = np.flatnonzero((dt > GAP_TOLERANCE / fs) | (dt <= 0)) + 1
    bounds = np.concatenate([[0], breaks, [len(t)]])
    rec = Recording(path=str(path), fs=fs, header_local_time=first[0], header_epoch=epoch)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - a) / fs < min_segment_s:
            continue
        chans = {name: df[name].to_numpy()[a:b].astype(np.float32) for name in ("ecg", "ppg", "resp")}
        rec.segments.append(Segment(t0_epoch=float(t[a]), fs=fs, channels=chans))
    return rec
