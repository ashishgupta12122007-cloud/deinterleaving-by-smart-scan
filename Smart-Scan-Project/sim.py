"""
Emitter-level "digital twin" for the SIH 26055 scenario files.

Everything here is calibrated from the five provided .h5 files:
  * AoA is the compass bearing (0 deg = +y, clockwise) from receiver to emitter.
  * Emitters move as  p(t) = p0 + v * t * (cos a, sin a)   (a = travel_angle_deg, math convention).
  * A circular-scan beam points at  az(t) = scan_start_angle + 360*rpm/60 * t  (compass, clockwise);
    it hits the receiver when az(t) == bearing(emitter -> receiver).
  * Received peak power  P_pk = Pt[dBm] - FSPL(d, f) + rx_gain   (main-beam peak, measured offset ~ 0 +- 1.5 dB)
  * Off-beam loss follows an empirical pattern  L(|r|/beam_width)  fitted on the pooled data.
  * A pulse is recorded with probability  P_rec(P_pk - L(.))  (two soft logistics, fitted).
The model works with *expected* pulse counts so any dwell schedule can be scored in milliseconds.
"""
import numpy as np

# ------------------------------------------------------------------ empirical antenna pattern
PAT_U = np.array([0, 0.05, 0.1, 0.17, 0.25, 0.4, 0.6, 0.9, 1.25, 1.75, 2.5, 3.5, 5, 7, 10, 14, 20, 1e9])
PAT_L = np.array([0.3, 1.0, 2.0, 3.5, 9.0, 15.0, 20.0, 20.5, 23.6, 28.2, 29.8, 33.1, 35.9, 38.5, 41.5, 43.6, 46.7, 46.7])
FLOOR_OFFSET_DB = 12.0   # far-side-lobe / back-lobe floor = antenna gain + 12 dB below the boresight peak
# soft pulse-recording probability fitted (Poisson likelihood) on clean FixedSingle/Fixed-PRI emitters of the 5 files:
#   P_rec(A) = p1*sig((A-a1)/s1) + p2*sig((A-a2)/s2),  A = modelled received amplitude in dBm
P_REC = dict(p1=0.6218, a1=-85.13, s1=3.2155, p2=0.0489, a2=-110.19, s2=7.2411)


def pattern_loss_db(u, gain_db=None):
    u = np.abs(u)
    L = np.interp(u, PAT_U[:-1], PAT_L[:-1])
    far = 46.7 + 20.0 * np.log10(np.maximum(u, 20.0) / 20.0)      # continues rising ~20 dB/decade
    L = np.where(u > 20.0, far, L)
    if gain_db is not None:
        L = np.minimum(L, gain_db + FLOOR_OFFSET_DB)
    return L


def _phi(z):
    # standard-normal CDF (vectorised, no scipy dependency at import time)
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def _phi_fast(z):
    # logistic approximation of the normal CDF (max abs err ~ 0.01) - fast path for big arrays
    return 1.0 / (1.0 + np.exp(-1.702 * z))


class Emitter:
    def __init__(self, d, rx_pos, rx_gain_db):
        self.d = d
        self.idx = d["idx"]
        self.name = d["function"]
        fc, pc, pw, pri, sc = (d["frequency_config"], d["position_config"], d["power_config"],
                               d["pri_config"], d["scan_config"])
        self.fmode = fc["freq_mode"]
        self.freqs = np.asarray(d["freqs_mhz"], float)
        self.pri_mode = pri["pri_mode"]
        self.pris = np.asarray(d["pris_us"], float)
        self.scan_type = sc["scan_type"]
        self.rpm = float(sc["scan_rate_rpm"])
        self.bw = float(sc["beam_width_deg"])
        self.az0 = float(sc["scan_start_angle"])
        self.p0 = np.asarray(d["start_position_km"], float)
        self.v = float(pc["speed_km_s"])
        self.ang = np.radians(float(pc["travel_angle_deg"]))
        self.pt_dbm = 10 * np.log10(max(pw["power_w"], 1e-6) * 1000.0)
        self.gain = float(pw["gain"])
        self.rx = np.asarray(rx_pos, float)
        self.rx_gain = float(rx_gain_db)
        self.pws = np.asarray(d["pws_us"], float)

    # --------------------------------------------------------------- kinematics
    def pos(self, t):
        t = np.asarray(t, float)
        return (self.p0[0] + self.v * t * np.cos(self.ang), self.p0[1] + self.v * t * np.sin(self.ang))

    def dist_km(self, t):
        x, y = self.pos(t)
        return np.hypot(x - self.rx[0], y - self.rx[1])

    def aoa_deg(self, t):
        x, y = self.pos(t)
        return np.degrees(np.arctan2(x - self.rx[0], y - self.rx[1]))

    def beam_offset_deg(self, t):
        """beam azimuth minus bearing(emitter->receiver), wrapped to [-180,180]."""
        t = np.asarray(t, float)
        brg = self.aoa_deg(t) + 180.0
        az = self.az0 + 360.0 * self.rpm / 60.0 * t
        return (az - brg + 180.0) % 360.0 - 180.0

    # --------------------------------------------------------------- pulse statistics
    @property
    def mean_freq(self):
        return float(self.freqs.mean())

    def pulse_rate_pps(self):
        p = self.pris[self.pris > 0]
        if len(p) == 0:
            return 0.0
        if self.pri_mode == "SwitchDwell":
            return float(np.mean(1e6 / p))
        return float(1e6 / np.mean(p))

    def freq_factor(self, centers, half_bw=500.0, simul_mode="all"):
        """expected number of recorded-carrier pulses per emitted pulse for each receiver centre."""
        c = np.atleast_1d(np.asarray(centers, float))
        f = self.freqs
        lo, hi = c - half_bw, c + half_bw
        if self.fmode == "RandomRange":
            a, b = float(f.min()), float(f.max())
            if b - a < 1e-9:
                return ((f[0] >= lo) & (f[0] <= hi)).astype(float)
            ov = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
            return ov / (b - a)
        inb = ((f[None, :] >= lo[:, None]) & (f[None, :] <= hi[:, None])).sum(1)
        if self.fmode == "FixedMultiSimultaneous":
            return inb.astype(float) if simul_mode == "all" else inb / len(f)
        return inb / len(f)

    def peak_dbm(self, t):
        d = np.maximum(self.dist_km(t), 0.05)
        fspl = 32.44 + 20 * np.log10(d) + 20 * np.log10(max(self.mean_freq, 1.0))
        return self.pt_dbm - fspl + self.rx_gain

    def visibility(self, t):
        """probability that a pulse emitted at time t is above the receive threshold."""
        t = np.asarray(t, float)
        pk = self.peak_dbm(t)
        if self.scan_type == "Omni":
            loss = np.zeros_like(pk)
        else:
            loss = pattern_loss_db(self.beam_offset_deg(t) / max(self.bw, 1e-3), self.gain)
        A = pk - loss
        P = P_REC
        return (P["p1"] / (1.0 + np.exp(-(A - P["a1"]) / P["s1"])) +
                P["p2"] / (1.0 + np.exp(-(A - P["a2"]) / P["s2"])))

    def main_beam_period_s(self):
        return 60.0 / self.rpm if (self.scan_type == "Circular" and self.rpm > 0) else np.inf


class Twin:
    """Vectorised digital twin of one scenario (all emitters, 1 ms time grid)."""

    def __init__(self, D, dt=1e-3, simul_mode="all"):
        self.D = D
        self.rx_pos = np.asarray(D["rxd"]["start_position_km"], float)
        self.rx_gain = D["rx"]["gain_db"]
        self.B = float(D["rx"]["bandwith_mhz"])
        self.T = float(D["rx"]["collection_time_s"])
        self.dt = dt
        self.simul_mode = simul_mode
        self.ids = sorted(D["tx"])
        self.em = [Emitter(D["tx"][k], self.rx_pos, self.rx_gain) for k in self.ids]
        self.tgrid = (np.arange(int(self.T / dt)) + 0.5) * dt
        # expected recorded pulses per second at every 1 ms slot, before frequency selection
        rate = np.array([e.pulse_rate_pps() for e in self.em])
        vis = np.stack([e.visibility(self.tgrid) for e in self.em])          # (E, Tn)
        self.rate = rate
        self.vis = vis
        self.cum = np.concatenate([np.zeros((len(self.em), 1)), np.cumsum(vis * rate[:, None] * dt, axis=1)], axis=1)

    def n_slots(self, t):
        return np.clip(np.round(np.asarray(t) / self.dt).astype(int), 0, self.cum.shape[1] - 1)

    def expected_pulses(self, t0, t1, center):
        """expected recorded pulses for every emitter for a dwell (t0,t1,center)."""
        i0, i1 = self.n_slots(t0), self.n_slots(t1)
        base = self.cum[:, i1] - self.cum[:, i0]
        ff = np.array([e.freq_factor([center], self.B, self.simul_mode)[0] for e in self.em])
        return base * ff

    def score_schedule(self, dwells):
        """dwells: iterable of (t0, dur, center). returns cumulative expected pulses per emitter (E,) and per-dwell (n_dwell,E)."""
        out = []
        for (t0, dur, c) in dwells:
            out.append(self.expected_pulses(t0, t0 + dur, c))
        out = np.array(out)
        return out.sum(0), out


def baseline_schedule(D):
    dc = np.asarray(D["rxd"]["dwell_centres_mhz"], float)
    dt = np.asarray(D["rxd"]["dwell_times_s"], float)
    T = float(D["rx"]["collection_time_s"])
    t = 0.0
    out = []
    while t < T - 1e-9:
        for c, d in zip(dc, dt):
            if t >= T - 1e-9:
                break
            d2 = min(d, T - t)
            out.append((t, d2, float(c)))
            t += d2
    return out
