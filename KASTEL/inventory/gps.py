"""Vehicle GPS track from the lab's recording layout (`location/<rec>_loc.csv`).

Time sync: the recording folder name is the video start in LOCAL time
(e.g. 2024_05_18_02_09_20_847 = 02:09:20.847, Tucson = UTC-7, no DST). Video second t
corresponds to `timestamp_utc_local` = start_epoch_ms + 1000 t. `timestamp_utc_gps` is NOT
used: on the 2024-05-18 recording it reads Jan 2025 (inconsistent clock).
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
import re

import numpy as np

M_PER_DEG_LAT = 111_320.0


def recording_start_ms(video_path: str, utc_offset_h: float = -7.0) -> int:
    m = re.search(r"(\d{4})_(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d{3})", os.path.basename(video_path))
    if not m:
        raise ValueError(f"no YYYY_MM_DD_HH_MM_SS_mmm timestamp in {video_path}")
    y, mo, d, h, mi, s, ms = (int(v) for v in m.groups())
    local = dt.datetime(y, mo, d, h, mi, s, ms * 1000, tzinfo=dt.timezone(dt.timedelta(hours=utc_offset_h)))
    return int(local.timestamp() * 1000)


def find_loc_csv(video_path: str) -> str:
    """camera/<rec>_cam_<rec>.mp4 -> location/<rec>_loc.csv (lab OneDrive layout)."""
    rec_dir = os.path.dirname(os.path.dirname(os.path.abspath(video_path)))
    loc_dir = os.path.join(rec_dir, "location")
    hits = [f for f in os.listdir(loc_dir) if f.endswith("_loc.csv")] if os.path.isdir(loc_dir) else []
    if not hits:
        raise FileNotFoundError(f"no *_loc.csv under {loc_dir}")
    return os.path.join(loc_dir, hits[0])


class GpsTrack:
    def __init__(self, loc_csv: str, start_ms: int, min_speed_for_heading: float = 1.0):
        with open(loc_csv, newline="", encoding="utf-8") as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        self.t = np.array([(int(r["timestamp_utc_local"]) - start_ms) / 1000.0 for r in rows])
        self.lat = np.array([float(r["latitude_dd"]) for r in rows])
        self.lon = np.array([float(r["longitude_dd"]) for r in rows])
        self.speed = np.array([float(r["speed_ms"]) for r in rows])
        bearing = np.array([float(r["bearing_deg"]) for r in rows])
        # GPS bearing is noise when (nearly) stopped, e.g. waiting at a red light: hold last good one
        self.heading = bearing.copy()
        for i in range(1, len(bearing)):
            if self.speed[i] < min_speed_for_heading:
                self.heading[i] = self.heading[i - 1]

    def pose(self, t: float):
        lat = float(np.interp(t, self.t, self.lat))
        lon = float(np.interp(t, self.t, self.lon))
        # interpolate heading on the unit circle (359 -> 1 deg must not pass through 180)
        h = np.radians(self.heading)
        hdg = math.degrees(math.atan2(np.interp(t, self.t, np.sin(h)), np.interp(t, self.t, np.cos(h)))) % 360
        return lat, lon, hdg

    @staticmethod
    def offset(lat, lon, heading_deg, forward_m, right_m):
        """Move (forward, right) metres from (lat, lon) facing heading_deg (clockwise from north)."""
        h = math.radians(heading_deg)
        east = forward_m * math.sin(h) + right_m * math.cos(h)
        north = forward_m * math.cos(h) - right_m * math.sin(h)
        return (lat + north / M_PER_DEG_LAT,
                lon + east / (M_PER_DEG_LAT * math.cos(math.radians(lat))))

    def path(self):
        return [[round(a, 7), round(b, 7)] for a, b in zip(self.lat, self.lon)]


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000 * math.asin(math.sqrt(a))
