"""Load one SIH-26055 config_*.h5 scenario into plain python / numpy structures.

The loader uses the small pure-python HDF5 reader in mini_h5.py (no h5py needed - this is what was used to build and
test everything in this package).  If you prefer h5py, see the `load_config_h5py` reference implementation below;
it is a straight translation of the same layout but has NOT been run by the author.
"""
import numpy as np
from mini_h5 import H5


def _py(v):
    return v.item() if hasattr(v, "item") else v


def load_config(path):
    h = H5(path)
    X = h["data"].read().astype(np.float64)                  # (N,5): ToA[us], Frequency[MHz], PulseWidth[us], AoA[deg], Amplitude[dBm]
    y = h["labels"].read().ravel().astype(int)               # transmitter index of every pulse
    rxa = {k: _py(v) for k, v in h["metadata/receiver"].attrs().items()}
    rxd = {k: h["metadata/receiver/" + k].read() for k in ["start_position_km", "freq_range_mhz", "dwell_centres_mhz", "dwell_times_s"]}
    kids = h.group_children(h["metadata/transmitters"])
    tx = {}
    for name in kids:
        idx = int(name.split("_")[1])
        base = "metadata/transmitters/" + name
        d = {"idx": idx, "function": h[base].attrs()["function"]}
        for sub in ["frequency_config", "position_config", "power_config", "pri_config", "pulse_width_config", "scan_config"]:
            d[sub] = {k: _py(v) for k, v in h[base + "/" + sub].attrs().items()}
        d["freqs_mhz"] = h[base + "/frequency_config/freqs_mhz"].read()
        d["start_position_km"] = h[base + "/position_config/start_position_km"].read()
        d["pris_us"] = h[base + "/pri_config/pris_us"].read()
        d["pws_us"] = h[base + "/pulse_width_config/pws_us"].read()
        tx[idx] = d
    meta = {k: _py(v) for k, v in h["metadata"].attrs().items()}
    return dict(X=X, y=y, rx=rxa, rxd=rxd, tx=tx, meta=meta)


def load_config_h5py(path):          # reference only - untested
    import h5py
    with h5py.File(path, "r") as f:
        X = f["data"][:].astype(np.float64)
        y = f["labels"][:].ravel().astype(int)
        rxa = dict(f["metadata/receiver"].attrs)
        rxd = {k: f["metadata/receiver/" + k][:] for k in ["start_position_km", "freq_range_mhz", "dwell_centres_mhz", "dwell_times_s"]}
        tx = {}
        for name, g in f["metadata/transmitters"].items():
            idx = int(name.split("_")[1])
            d = {"idx": idx, "function": g.attrs["function"]}
            for sub in ["frequency_config", "position_config", "power_config", "pri_config", "pulse_width_config", "scan_config"]:
                d[sub] = dict(g[sub].attrs)
            d["freqs_mhz"] = g["frequency_config/freqs_mhz"][:]
            d["start_position_km"] = g["position_config/start_position_km"][:]
            d["pris_us"] = g["pri_config/pris_us"][:]
            d["pws_us"] = g["pulse_width_config/pws_us"][:]
            tx[idx] = d
        meta = dict(f["metadata"].attrs)
    return dict(X=X, y=y, rx=rxa, rxd=rxd, tx=tx, meta=meta)
