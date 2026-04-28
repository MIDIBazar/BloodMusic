# live_cellular_stream.py
# Streaming live controller -> ghost particle -> MIDI out (loopMIDI/Ableton)

import math
import time
import heapq
from dataclasses import dataclass
from typing import Dict, Optional
import cv2
import numpy as np
import pandas as pd
import mido
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import json

# Recommended backend for Mido
mido.set_backend("mido.backends.rtmidi")


# =========================================================
# CONFIGURATION
# =========================================================
DEBUG_MIDI = True

# ===== SOURCE MODE =====
USE_BCCD = True   # True = use ZIP BCCD + image_id, False = use EVENTS_CSV

EVENTS_CSV = "events_BloodImage_00000_basic.csv"
BCCD_ZIP_PATH = "blood_dataset.zip"
BCCD_IMAGE_ID = "BloodImage_00000"

# Use partial names so find_port_name() can find the right port
CONTROLLER_INPUT_NAME = "Arturia MiniLab mkII"
MIDI_OUTPUT_NAME = "CellularBloodToAbleton"

WINDOW_SCALE = 1.6   # 1.0 = image size, >1 larger
FULLSCREEN = False   # set True for fullscreen

DT = 0.02  # 50 Hz

VISUALIZE = True
VISUAL_UPDATE_EVERY = 2   # update the view every N frames

SCALE_ORDER = [
    "minor_pentatonic",
    "major_pentatonic",
    "dorian",
    "aeolian",
    "phrygian",
    "mixolydian",
    "lydian",
    "harmonic_minor",
    "melodic_minor",
    "whole_tone",
    "diminished",
    "hirajoshi",
    "in_sen",
    "yo",
    "iwato",
    "double_harmonic",
    "hungarian_minor",
    "persian",
    "blues_minor",
    "blues_major",
]

SCALES: Dict[str, list[int]] = {
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "aeolian":          [0, 2, 3, 5, 7, 8, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "lydian":           [0, 2, 4, 6, 7, 9, 11],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":    [0, 2, 3, 5, 7, 9, 11],
    "whole_tone":       [0, 2, 4, 6, 8, 10],
    "diminished":       [0, 2, 3, 5, 6, 8, 9, 11],
    "hirajoshi":        [0, 2, 3, 7, 8],
    "in_sen":           [0, 1, 5, 7, 10],
    "yo":               [0, 2, 5, 7, 9],
    "iwato":            [0, 1, 5, 6, 10],
    "double_harmonic":  [0, 1, 4, 5, 7, 8, 11],
    "hungarian_minor":  [0, 2, 3, 6, 7, 8, 11],
    "persian":          [0, 1, 4, 5, 6, 8, 11],
    "blues_minor":      [0, 3, 5, 6, 7, 10],
    "blues_major":      [0, 2, 3, 4, 7, 9],
}

HARMONY_MODES = [
    "triad",
    "dyad_open",
    "dyad_close",
    "unison",
    "triad_dense",
    "spread",
    "octaves",
    "fifths",
    "quartal",
    "cluster_light",
    "drone_root",
    "drone_fifth",
]

# default fallback, overwritten by learned actions if available
ACTION_NOTE_MAP = {
    42: "cycle_scale_backward",
    43: "cycle_scale_forward",
    44: "cycle_harmony_backward",
    45: "cycle_harmony_forward",
    48: "reset_particle",
    49: "freeze_toggle",
    50: "all_notes_off",
}

ACTION_SPECS = [
    ("cycle_scale_backward", "Press button for SCALE BACKWARD"),
    ("cycle_scale_forward", "Press button for SCALE FORWARD"),
    ("cycle_harmony_backward", "Press button for HARMONY BACKWARD"),
    ("cycle_harmony_forward", "Press button for HARMONY FORWARD"),
    ("reset_particle", "Press button for RESET PARTICLE"),
    ("freeze_toggle", "Press button for FREEZE"),
    ("all_notes_off", "Press button for ALL NOTES OFF"),
]

CC_MAP = {}

MAPPING_FILE = "midi_mapping.json"
LEARN_MODE = False   # set to False after creating the mapping

PARAM_SPECS = {
    "root_midi":          {"min": 36.0, "max": 60.0,  "type": "int"},
    "octaves":            {"min": 1.0,  "max": 4.0,   "type": "int"},
    "duration_scale":     {"min": 0.4,  "max": 2.5,   "type": "float"},
    "velocity_compress":  {"min": 0.2,  "max": 1.0,   "type": "float"},
    "base_speed":         {"min": 30.0, "max": 140.0, "type": "float"},
    "wave_amp":           {"min": 0.0,  "max": 70.0,  "type": "float"},
    "wave_freq":          {"min": 0.2,  "max": 4.0,   "type": "float"},
    "phase_speed":        {"min": 0.0,  "max": 2.0,   "type": "float"},
    "center_pull":        {"min": 0.0,  "max": 0.4,   "type": "float"},
    "inertia":            {"min": 0.4,  "max": 0.98,  "type": "float"},
    "wbc_repulsion":      {"min": 0.0,  "max": 220.0, "type": "float"},
    "wbc_swirl":          {"min": 0.0,  "max": 260.0, "type": "float"},
    "trigger_radius_rbc": {"min": 20.0, "max": 140.0, "type": "float"},
    "trigger_radius_wbc": {"min": 40.0, "max": 220.0, "type": "float"},
    "retrigger_cooldown": {"min": 0.2,  "max": 6.0,   "type": "float"},
}

PANEL_FIELDS = {
    "MUSIC": [
        "scale_name",
        "root_midi",
        "octaves",
        "harmony_mode",
        "duration_scale",
        "velocity_compress",
    ],
    "FLOW": [
        "base_speed",
        "wave_amp",
        "wave_freq",
        "phase_speed",
        "center_pull",
        "inertia",
    ],
    "FIELD / TRIGGER": [
        "wbc_repulsion",
        "wbc_swirl",
        "trigger_radius_rbc",
        "trigger_radius_wbc",
        "retrigger_cooldown",
    ],
}

FIELD_LABELS = {
    "scale_name": "scale",
    "root_midi": "root",
    "octaves": "octaves",
    "harmony_mode": "harmony",
    "duration_scale": "dur_scale",
    "velocity_compress": "vel_comp",

    "base_speed": "speed",
    "wave_amp": "wave_amp",
    "wave_freq": "wave_freq",
    "phase_speed": "phase",
    "center_pull": "pull",
    "inertia": "inertia",

    "wbc_repulsion": "repulsion",
    "wbc_swirl": "swirl",
    "trigger_radius_rbc": "r_rbc",
    "trigger_radius_wbc": "r_wbc",
    "retrigger_cooldown": "cooldown",
}

PAD_CONTROLLED_FIELDS = {
    "scale_name",
    "harmony_mode",
}

# =========================================================
# HELPERS
# =========================================================
def format_field_value(field_name: str, value):
    if field_name == "root_midi":
        return midi_to_note_name(value)
    if field_name in {"scale_name", "harmony_mode"}:
        return str(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def get_active_controlled_fields(cc_map):
    """
    Return the names of fields that are actually controlled:
    - those mapped to knobs via CC_MAP
    - those controlled by pads
    """
    cc_fields = {name for _, (name, _, _, _) in cc_map.items()}
    return cc_fields | PAD_CONTROLLED_FIELDS


def midi_to_note_name(midi_note: int, include_octave: bool = False) -> str:
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    midi_note = int(midi_note)
    name = note_names[midi_note % 12]

    if include_octave:
        octave = (midi_note // 12) - 1
        return f"{name}{octave}"

    return name


def cycle_list_value(current_value: str, values: list[str], direction: int = 1) -> str:
    if current_value not in values:
        return values[0]
    idx = values.index(current_value)
    idx = (idx + direction) % len(values)
    return values[idx]


def cycle_scale(current_scale: str, direction: int = 1) -> str:
    return cycle_list_value(current_scale, SCALE_ORDER, direction=direction)


def cycle_harmony_mode(current_mode: str, direction: int = 1) -> str:
    return cycle_list_value(current_mode, HARMONY_MODES, direction=direction)


def list_ports():
    print("\nINPUT PORTS:")
    for n in mido.get_input_names():
        print(" -", n)

    print("\nOUTPUT PORTS:")
    for n in mido.get_output_names():
        print(" -", n)


def cc_to_range(value: int, vmin: float, vmax: float, mode: str = "float"):
    x = vmin + (value / 127.0) * (vmax - vmin)
    if mode == "int":
        return int(round(x))
    return float(x)


def clamp_midi_note(n: int) -> int:
    return max(0, min(127, int(n)))


def get_canvas_size(events: pd.DataFrame):
    # Try to recover width/height from the maximum bounding box values
    W = int(max(events["xmax"].max(), events["cx"].max()))
    H = int(max(events["ymax"].max(), events["cy"].max()))
    return W, H


def get_main_wbc_center(events: pd.DataFrame) -> Optional[np.ndarray]:
    if (events["label"] == "WBC").any():
        row = events[events["label"] == "WBC"].sort_values("area", ascending=False).iloc[0]
        return np.array([float(row["cx"]), float(row["cy"])], dtype=float)
    return None


def save_midi_mappings(cc_map, action_note_map, path=MAPPING_FILE):
    data = {
        "cc_map": {
            str(cc): {
                "name": name,
                "min": vmin,
                "max": vmax,
                "type": mode
            }
            for cc, (name, vmin, vmax, mode) in cc_map.items()
        },
        "action_note_map": {
            str(note): action
            for note, action in action_note_map.items()
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Mapping saved to {path}")


def load_midi_mappings(path=MAPPING_FILE):
    if not Path(path).exists():
        return {}, {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cc_raw = raw.get("cc_map", {})
    action_raw = raw.get("action_note_map", {})

    cc_map = {
        int(cc): (spec["name"], spec["min"], spec["max"], spec["type"])
        for cc, spec in cc_raw.items()
    }

    action_note_map = {
        int(note): action
        for note, action in action_raw.items()
    }

    print(f"[OK] Mapping loaded from {path}")
    return cc_map, action_note_map


def learn_action_notes(in_port, action_specs):
    """
    Separate learn mode for buttons/pads (note_on).
    """
    learned = {}
    used_notes = set()

    print("\n=== ACTION/PAD LEARN MODE ===")
    print("Press the requested button/pad for each action.\n")

    for action_name, prompt in action_specs:
        print(f"\n[LEARN ACTION] {prompt}")
        assigned = False
        t0 = time.time()

        while time.time() - t0 < 8.0:
            for msg in in_port.iter_pending():
                if msg.type == "note_on" and msg.velocity > 0:
                    note = msg.note

                    if note in used_notes:
                        print(f"  [WARN] NOTE {note} already used, ignored.")
                        continue

                    learned[note] = action_name
                    used_notes.add(note)
                    print(f"  [OK] {action_name} <- NOTE {note}")
                    assigned = True
                    break

            if assigned:
                break

            time.sleep(0.01)

        if not assigned:
            print(f"  [SKIP] No button assigned to {action_name}")

    print("\n=== ACTION LEARN COMPLETED ===")
    return learned


def print_action_mapping_table(action_note_map):
    print("\n=== ACTION / PAD MAPPING TABLE ===")
    print(f"{'NOTE':<8}{'ACTION':<28}")
    print("-" * 40)

    for note, action in sorted(action_note_map.items()):
        print(f"{note:<8}{action:<28}")

    print("-" * 40)
    print()


class BCCD:
    """Minimal BCCD loader from a ZIP archive."""
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.zf = zipfile.ZipFile(self.zip_path, "r")
        self._names = self.zf.namelist()

    def list_image_ids(self):
        ids = []
        for n in self._names:
            if n.startswith("BCCD/JPEGImages/") and n.endswith(".jpg"):
                ids.append(Path(n).stem)
        return sorted(ids)

    def load_image(self, image_id: str):
        img_bytes = self.zf.read(f"BCCD/JPEGImages/{image_id}.jpg")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def load_annotations(self, image_id: str):
        xml_bytes = self.zf.read(f"BCCD/Annotations/{image_id}.xml")
        root = ET.fromstring(xml_bytes)

        size = root.find("size")
        W = int(size.findtext("width"))
        H = int(size.findtext("height"))

        rows = []
        for i, obj in enumerate(root.findall("object")):
            label = obj.findtext("name")
            bnd = obj.find("bndbox")
            xmin = int(bnd.findtext("xmin"))
            ymin = int(bnd.findtext("ymin"))
            xmax = int(bnd.findtext("xmax"))
            ymax = int(bnd.findtext("ymax"))

            bw = max(1, xmax - xmin)
            bh = max(1, ymax - ymin)
            area = bw * bh
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0

            rows.append({
                "cell_id": i,
                "label": label,
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "bbox_w": bw, "bbox_h": bh,
                "cx": cx, "cy": cy,
                "x_norm": cx / W,
                "y_norm": cy / H,
                "area": area,
                "aspect": bw / bh
            })

        return pd.DataFrame(rows), (W, H)


def robust_minmax(x: np.ndarray, lo_p: float = 5, hi_p: float = 95):
    if len(x) == 0:
        return x, 0.0, 1.0

    lo = float(np.percentile(x, lo_p))
    hi = float(np.percentile(x, hi_p))
    x_clip = np.clip(x, lo, hi)
    denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    return (x_clip - lo) / denom, lo, hi


def build_events_basic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    area_norm, _, _ = robust_minmax(out["area"].to_numpy())
    out["area_norm"] = area_norm

    pts = out[["cx", "cy"]].to_numpy()
    N = len(out)

    if N <= 1:
        sparsity = np.zeros(N)
    else:
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        k = min(5, N - 1)
        d_sorted = np.sort(d, axis=1)[:, 1:k+1]
        knn_mean = d_sorted.mean(axis=1)
        sparsity, _, _ = robust_minmax(knn_mean)

    out["sparsity_norm"] = sparsity

    # pitch base
    scale = [0, 3, 5, 7, 10]   # minor pentatonic
    base = 48                  # C3
    octaves = 2
    steps_total = len(scale) * octaves

    idx = np.clip((out["y_norm"].to_numpy() * (steps_total - 1)).astype(int), 0, steps_total - 1)
    octv = idx // len(scale)
    deg = idx % len(scale)
    out["midi_note"] = base + 12 * octv + np.take(scale, deg)

    vel = 35 + 60 * np.sqrt(out["area_norm"].to_numpy())
    vel += np.where(out["label"].to_numpy() == "WBC", 12, 0)
    vel -= np.where(out["label"].to_numpy() == "Platelets", 8, 0)
    out["velocity"] = np.clip(vel, 1, 127).astype(int)

    dur = 0.8 + 2.8 * out["sparsity_norm"].to_numpy()
    dur += 0.6 * np.where(out["label"].to_numpy() == "WBC", 1, 0)
    out["duration_s"] = dur

    out["pan"] = (out["x_norm"] * 2 - 1).astype(float)

    out["track"] = np.where(
        out["label"] == "RBC", "pad",
        np.where(out["label"] == "WBC", "lead", "shimmer")
    )

    out["start_s"] = 0.0
    return out


# =========================================================
# STATE
# =========================================================

@dataclass
class LiveState:
    # particle / flow
    base_speed: float = 72.0
    wave_amp: float = 26.0
    wave_freq: float = 1.6
    phase_speed: float = 0.45
    center_pull: float = 0.10
    inertia: float = 0.88

    # WBC perturbation
    wbc_repulsion: float = 55.0
    wbc_swirl: float = 120.0
    wbc_sigma: float = 95.0

    # trigger radii
    trigger_radius_rbc: float = 70.0
    trigger_radius_wbc: float = 125.0
    trigger_radius_platelet: float = 45.0

    # retrigger / cooldown
    retrigger_cooldown: float = 2.0

    # musical mapping
    scale_name: str = "minor_pentatonic"
    root_midi: int = 48
    octaves: int = 2
    duration_scale: float = 1.0
    velocity_compress: float = 0.72

    # harmony
    harmony_mode: str = "unison"

    # engine state
    frozen: bool = False


# =========================================================
# GHOST ENGINE
# =========================================================

class GhostParticleEngine:
    def __init__(self, events_df: pd.DataFrame, state: LiveState):
        self.state = state
        self.events = events_df.copy().reset_index(drop=True)

        required = {
            "cell_id", "label", "cx", "cy", "x_norm", "y_norm",
            "xmin", "ymin", "xmax", "ymax",
            "area", "velocity", "duration_s", "track"
        }
        missing = required - set(self.events.columns)
        if missing:
            raise ValueError(f"Missing event CSV columns: {missing}")

        self.W, self.H = get_canvas_size(self.events)
        self.wbc_center = get_main_wbc_center(self.events)

        self.events["triggered"] = False
        self.events["active_note"] = -1
        self.events["last_trigger_time"] = -1e9

        self.note_off_heap = []  # (off_time, note, channel)
        self.sim_time = 0.0

        self.reset_particle(clear_triggers=True)

    def reset_particle(self, clear_triggers: bool = True):
        self.x = -20.0
        self.y = self.H / 2.0
        self.vx = self.state.base_speed
        self.vy = 0.0
        self.sim_time = 0.0

        if clear_triggers:
            self.events["triggered"] = False
            self.events["active_note"] = -1
            self.events["last_trigger_time"] = -1e9

    def flow_vector(self, x, y, t):
        s = self.state

        vx = s.base_speed

        phase = 2 * math.pi * s.wave_freq * (x / max(self.W, 1)) + s.phase_speed * t
        vy = s.wave_amp * math.sin(phase)

        vy += -s.center_pull * (y - self.H / 2.0)

        if self.wbc_center is not None:
            dx = x - self.wbc_center[0]
            dy = y - self.wbc_center[1]
            r2 = dx * dx + dy * dy + 1e-9
            r = math.sqrt(r2)

            g = math.exp(-r2 / (2.0 * (s.wbc_sigma ** 2)))

            ux = dx / r
            uy = dy / r

            tx = -uy
            ty = ux

            vx += s.wbc_repulsion * g * ux
            vy += s.wbc_repulsion * g * uy

            vx += s.wbc_swirl * g * tx
            vy += s.wbc_swirl * g * ty

        return vx, vy

    def row_to_note(self, row) -> int:
        scale = SCALES[self.state.scale_name]
        steps_total = len(scale) * max(1, self.state.octaves)

        idx = int(np.clip(row["y_norm"] * (steps_total - 1), 0, steps_total - 1))
        octv = idx // len(scale)
        deg = idx % len(scale)

        note = self.state.root_midi + 12 * octv + scale[deg]
        return clamp_midi_note(note)
    
    def row_to_scale_position(self, row):
        """
        Return:
        - global index in the expanded scale
        - degree in the scale
        - relative octave
        """
        scale = SCALES[self.state.scale_name]
        steps_total = len(scale) * max(1, self.state.octaves)

        idx = int(np.clip(row["y_norm"] * (steps_total - 1), 0, steps_total - 1))
        octv = idx // len(scale)
        deg = idx % len(scale)
        return idx, deg, octv

    def scale_note_from_degree_offset(self, row, degree_offset: int, octave_extra: int = 0):
        """
        Construct a MIDI note while staying inside the current scale.
        """
        scale = SCALES[self.state.scale_name]
        _, deg, octv = self.row_to_scale_position(row)

        total_degree = deg + degree_offset
        new_octv = octv + octave_extra + (total_degree // len(scale))
        new_deg = total_degree % len(scale)

        note = self.state.root_midi + 12 * new_octv + scale[new_deg]
        return clamp_midi_note(note)

    def row_to_harmony_notes(self, row):
        """
        Return a list of tuples:
        (note, channel, velocity_scale)

        Harmony patterns vary by:
        - harmony_mode
        - cell label
        """
        label = row["label"]
        mode = self.state.harmony_mode
        root_note = self.row_to_note(row)

        if mode == "unison":
            if label == "Platelets":
                pattern = [(0, 0, 0.70)]
            else:
                pattern = [(0, 0, 1.00)]

        elif mode == "dyad_open":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (4, 1, 0.80)]
            elif label == "Platelets":
                pattern = [(0, 0, 0.60), (6, 1, 0.40)]
            else:
                pattern = [(0, 0, 1.00), (4, 1, 0.68)]

        elif mode == "dyad_close":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (2, 1, 0.85)]
            elif label == "Platelets":
                pattern = [(0, 0, 0.55), (1, 1, 0.35)]
            else:
                pattern = [(0, 0, 1.00), (2, 1, 0.70)]

        elif mode == "triad":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (2, 1, 0.82), (4, 2, 0.68)]
            elif label == "Platelets":
                pattern = [(0, 0, 0.55), (4, 1, 0.40), (6, 2, 0.28)]
            else:
                pattern = [(0, 0, 1.00), (2, 1, 0.72), (4, 2, 0.55)]

        elif mode == "triad_dense":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (1, 1, 0.80), (2, 2, 0.65)]
            elif label == "Platelets":
                pattern = [(0, 0, 0.50), (1, 1, 0.35), (3, 2, 0.22)]
            else:
                pattern = [(0, 0, 1.00), (1, 1, 0.65), (2, 2, 0.48)]

        elif mode == "spread":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (4, 1, 0.75), (8, 2, 0.58)]
            elif label == "Platelets":
                pattern = [(0, 0, 0.48), (7, 2, 0.30)]
            else:
                pattern = [(0, 0, 1.00), (4, 1, 0.62), (7, 2, 0.42)]

        elif mode == "octaves":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (0, 1, 0.78), (0, 2, 0.55)]
            elif label == "Platelets":
                pattern = [(0, 1, 0.45), (0, 2, 0.28)]
            else:
                pattern = [(0, 0, 1.00), (0, 1, 0.65)]

        elif mode == "fifths":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (4, 1, 0.78), (8, 2, 0.55)]
            elif label == "Platelets":
                pattern = [(4, 1, 0.36), (8, 2, 0.25)]
            else:
                pattern = [(0, 0, 1.00), (4, 1, 0.62)]

        elif mode == "quartal":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (3, 1, 0.78), (6, 2, 0.56)]
            elif label == "Platelets":
                pattern = [(3, 1, 0.34), (6, 2, 0.22)]
            else:
                pattern = [(0, 0, 1.00), (3, 1, 0.62), (6, 2, 0.40)]

        elif mode == "cluster_light":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (1, 1, 0.70), (2, 2, 0.52)]
            elif label == "Platelets":
                pattern = [(1, 1, 0.28), (2, 2, 0.18)]
            else:
                pattern = [(0, 0, 1.00), (1, 1, 0.52)]

        elif mode == "drone_root":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (0, 1, 0.55)]
            elif label == "Platelets":
                pattern = [(0, 2, 0.25)]
            else:
                pattern = [(0, 0, 1.00)]

        elif mode == "drone_fifth":
            if label == "WBC":
                pattern = [(0, 0, 1.00), (4, 1, 0.55)]
            elif label == "Platelets":
                pattern = [(4, 2, 0.25)]
            else:
                pattern = [(0, 0, 1.00), (4, 1, 0.45)]

        else:
            pattern = [(0, 0, 1.00)]

        notes = []
        for degree_offset, channel, vel_scale in pattern:
            if degree_offset == 0:
                note = root_note
            else:
                note = self.scale_note_from_degree_offset(row, degree_offset)

            notes.append((note, channel, vel_scale))

        return notes

    def row_to_velocity(self, row) -> int:
        v = float(row["velocity"])
        v = 64 + (v - 64) * self.state.velocity_compress
        return int(np.clip(round(v), 20, 110))

    def row_to_duration(self, row) -> float:
        base = float(row["duration_s"]) * self.state.duration_scale

        if row["track"] == "pad":
            return max(base, 1.6)
        if row["track"] == "lead":
            return max(base, 2.4)
        return max(base, 0.18)

    def row_to_channel(self, row) -> int:
        return 0

    def trigger_radius_for_row(self, row) -> float:
        if row["label"] == "WBC":
            return self.state.trigger_radius_wbc
        if row["label"] == "Platelets":
            return self.state.trigger_radius_platelet
        return self.state.trigger_radius_rbc

    def point_to_bbox_distance(self, x, y, row) -> float:
        """
        Point to bounding box distance.
        Returns 0 if the point is inside the box.
        """
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

        dx = max(xmin - x, 0.0, x - xmax)
        dy = max(ymin - y, 0.0, y - ymax)

        return math.sqrt(dx * dx + dy * dy)

    def all_notes_off(self, out_port):
        for ch in [0, 1, 2]:
            out_port.send(mido.Message("control_change", channel=ch, control=123, value=0))
            out_port.send(mido.Message("control_change", channel=ch, control=120, value=0))
        self.note_off_heap.clear()
        self.events["active_note"] = -1

    def process_note_offs(self, out_port, now_s: float):
        while self.note_off_heap and self.note_off_heap[0][0] <= now_s:
            _, note, channel = heapq.heappop(self.note_off_heap)
            print(f"[NOTE OFF] note={note} ch={channel}")
            out_port.send(mido.Message("note_off", note=note, velocity=0, channel=channel))

    def step(self, out_port, dt: float = DT):
        self.process_note_offs(out_port, self.sim_time)

        if self.state.frozen:
            return

        fx, fy = self.flow_vector(self.x, self.y, self.sim_time)

        self.vx = self.state.inertia * self.vx + (1.0 - self.state.inertia) * fx
        self.vy = self.state.inertia * self.vy + (1.0 - self.state.inertia) * fy

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.y = float(np.clip(self.y, 0, self.H))

        # soft wrap: when it exits to the right, restart
        if self.x > self.W + 40:
            self.x = -20.0
            self.y = self.H / 2.0

        # check all cells using bbox distance + cooldown
        for idx in range(len(self.events)):
            row = self.events.iloc[idx]
            radius = self.trigger_radius_for_row(row)

            dist = self.point_to_bbox_distance(self.x, self.y, row)
            can_retrigger = (self.sim_time - self.events.at[idx, "last_trigger_time"]) >= self.state.retrigger_cooldown

            if dist <= radius and can_retrigger:
                base_vel = self.row_to_velocity(row)
                dur = self.row_to_duration(row)
                harmony_notes = self.row_to_harmony_notes(row)

                print(
                    f"[TRIGGER] cell_id={row['cell_id']} "
                    f"label={row['label']} track={row['track']} "
                    f"mode={self.state.harmony_mode} "
                    f"x={self.x:.1f} y={self.y:.1f} dist={dist:.2f} radius={radius:.2f}"
                )

                for note, ch, vel_scale in harmony_notes:
                    vel = int(np.clip(round(base_vel * vel_scale), 15, 110))

                    print(
                        f"    -> SEND note={note} vel={vel} ch={ch}"
                    )

                    out_port.send(mido.Message("note_on", note=note, velocity=vel, channel=ch))
                    heapq.heappush(self.note_off_heap, (self.sim_time + dur, note, ch))

                self.events.at[idx, "triggered"] = True
                self.events.at[idx, "active_note"] = harmony_notes[0][0]
                self.events.at[idx, "last_trigger_time"] = self.sim_time

        self.sim_time += dt


# =========================================================
# MIDI CONTROL MAPPING
# =========================================================

def learn_cc_mapping(in_port, param_specs):
    """
    Interactive procedure:
    for each parameter, wait for a MIDI CC control and assign it.
    """
    learned = {}
    used_ccs = set()

    print("\n=== MIDI LEARN MODE ===")
    print("Move a knob/fader for each requested parameter.")
    print("If you want to skip a parameter, press Enter and do not move anything for a few seconds.\n")

    for param_name, spec in param_specs.items():
        print(f"\n[LEARN] Assign control to: {param_name}")
        print(f"        Range: {spec['min']} -> {spec['max']} ({spec['type']})")
        print("        Move a knob/fader NOW...")

        assigned = False
        t0 = time.time()

        while time.time() - t0 < 8.0:  # 8 second timeout
            for msg in in_port.iter_pending():
                if msg.type == "control_change":
                    cc = msg.control

                    if cc in used_ccs:
                        print(f"  [WARN] CC {cc} already used, ignored.")
                        continue

                    learned[cc] = (param_name, spec["min"], spec["max"], spec["type"])
                    used_ccs.add(cc)

                    print(f"  [OK] {param_name} <- CC {cc}")
                    assigned = True
                    break

            if assigned:
                break

            time.sleep(0.01)

        if not assigned:
            print(f"  [SKIP] No control assigned to {param_name}")

    print("\n=== LEARN COMPLETED ===")
    return learned


def print_cc_mapping_table(cc_map):
    print("\n=== CC MAPPING TABLE ===")
    print(f"{'CC':<6}{'PARAMETER':<24}{'RANGE':<20}{'TYPE':<10}")
    print("-" * 60)

    for cc_num, (name, vmin, vmax, mode) in sorted(cc_map.items()):
        print(f"{cc_num:<6}{name:<24}{f'{vmin} -> {vmax}':<20}{mode:<10}")

    print("-" * 60)
    print()


def handle_cc(msg, state: LiveState):
    if msg.control not in CC_MAP:
        print(f"[UNMAPPED CC] control={msg.control} value={msg.value}")
        return

    name, vmin, vmax, mode = CC_MAP[msg.control]
    old_value = getattr(state, name)
    new_value = cc_to_range(msg.value, vmin, vmax, mode)
    setattr(state, name, new_value)

    if isinstance(new_value, float):
        print(f"[CC {msg.control:02d}] {name}: {old_value:.3f} -> {new_value:.3f}")
    else:
        print(f"[CC {msg.control:02d}] {name}: {old_value} -> {new_value}")


def handle_note_control(msg, state: LiveState, engine: GhostParticleEngine, out_port):
    if msg.velocity == 0:
        return

    if msg.note not in ACTION_NOTE_MAP:
        return

    action = ACTION_NOTE_MAP[msg.note]

    if action == "cycle_scale_forward":
        old_scale = state.scale_name
        state.scale_name = cycle_scale(state.scale_name, direction=1)
        print(f"[ACTION] scale: {old_scale} -> {state.scale_name}")

    elif action == "cycle_scale_backward":
        old_scale = state.scale_name
        state.scale_name = cycle_scale(state.scale_name, direction=-1)
        print(f"[ACTION] scale: {old_scale} -> {state.scale_name}")

    elif action == "cycle_harmony_forward":
        old_mode = state.harmony_mode
        state.harmony_mode = cycle_harmony_mode(state.harmony_mode, direction=1)
        print(f"[ACTION] harmony_mode: {old_mode} -> {state.harmony_mode}")

    elif action == "cycle_harmony_backward":
        old_mode = state.harmony_mode
        state.harmony_mode = cycle_harmony_mode(state.harmony_mode, direction=-1)
        print(f"[ACTION] harmony_mode: {old_mode} -> {state.harmony_mode}")

    elif action == "reset_particle":
        engine.all_notes_off(out_port)
        engine.reset_particle(clear_triggers=True)
        print("[ACTION] reset_particle")

    elif action == "freeze_toggle":
        state.frozen = not state.frozen
        print(f"[ACTION] frozen -> {state.frozen}")

    elif action == "all_notes_off":
        engine.all_notes_off(out_port)
        print("[ACTION] all_notes_off")


def debug_print_midi_message(msg):
    if msg.type == "control_change":
        print(f"[RAW CC] channel={msg.channel} control={msg.control} value={msg.value}")
    elif msg.type == "note_on":
        print(f"[RAW NOTE ON] channel={msg.channel} note={msg.note} velocity={msg.velocity}")
    elif msg.type == "note_off":
        print(f"[RAW NOTE OFF] channel={msg.channel} note={msg.note} velocity={msg.velocity}")
    elif msg.type == "pitchwheel":
        print(f"[RAW PITCH] channel={msg.channel} pitch={msg.pitch}")
    elif msg.type == "aftertouch":
        print(f"[RAW AFTERTOUCH] channel={msg.channel} value={msg.value}")
    elif msg.type == "polytouch":
        print(f"[RAW POLYTOUCH] channel={msg.channel} note={msg.note} value={msg.value}")
    else:
        print(f"[RAW OTHER] {msg}")


def process_controller_messages(in_port, state: LiveState, engine: GhostParticleEngine, out_port):
    for msg in in_port.iter_pending():
        if DEBUG_MIDI:
            debug_print_midi_message(msg)

        if msg.type == "control_change":
            handle_cc(msg, state)

        elif msg.type == "note_on":
            handle_note_control(msg, state, engine, out_port)


def find_port_name(query: str, names: list[str]) -> str:
    matches = [n for n in names if query.lower() in n.lower()]
    if not matches:
        raise OSError(f"No port found containing: {query!r}\nAvailable: {names}")
    if len(matches) > 1:
        print(f"[WARN] Multiple ports match {query!r}, using the first: {matches[0]}")
    return matches[0]


# =========================================================
# VISUALIZER (OpenCV)
# =========================================================

class OpenCVVisualizer:
    def __init__(self, img, events_df, engine, window_name="Blood Music Live", trail_len=160):
        self.engine = engine
        self.events_df = events_df.copy()
        self.window_name = window_name
        self.trail_len = trail_len
        self.trail = []

        if img is not None:
            rgb = np.array(img)
            self.base_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            W = int(max(events_df["xmax"].max(), events_df["cx"].max()))
            H = int(max(events_df["ymax"].max(), events_df["cy"].max()))
            self.base_frame = np.full((H, W, 3), 245, dtype=np.uint8)

        self.H, self.W = self.base_frame.shape[:2]

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if FULLSCREEN:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(
                self.window_name,
                int(self.W * WINDOW_SCALE),
                int(self.H * WINDOW_SCALE)
            )

    def _cell_color_bgr(self, label):
        if label == "WBC":
            return (90, 90, 255)      # red-magenta
        if label == "Platelets":
            return (80, 220, 255)     # light yellow/cyan
        return (255, 220, 120)        # RBC -> soft blue

    def _draw_transparent_circle(self, frame, center, radius, color, alpha, thickness=-1):
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_panel(self, frame, x, y, w, h, alpha=0.34):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
    def _draw_panel_text(self, frame, lines, x, y, title=None, line_h=14):
        yy = y
        if title is not None:
            cv2.putText(
                frame, title, (x, yy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 2, cv2.LINE_AA
            )
            yy += 20

        for line in lines:
            cv2.putText(
                frame, line, (x, yy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1, cv2.LINE_AA
            )
            yy += line_h

    def _draw_trail(self, frame):
        if len(self.trail) < 2:
            return

        overlay = frame.copy()
        n = len(self.trail)

        for i in range(1, n):
            p0 = self.trail[i - 1]
            p1 = self.trail[i]

            frac = i / max(1, n - 1)
            thickness = max(1, int(1 + 4 * frac))
            color = (255, int(180 * frac), int(80 * frac))  # white -> warm aqua

            cv2.line(
                overlay,
                (int(p0[0]), int(p0[1])),
                (int(p1[0]), int(p1[1])),
                color,
                thickness,
                lineType=cv2.LINE_AA
            )

        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def _draw_cells(self, frame):
        overlay = frame.copy()

        for _, row in self.events_df.iterrows():
            x = int(row["cx"])
            y = int(row["cy"])
            label = row["label"]

            # passive cells: very subtle
            cv2.circle(overlay, (x, y), 4, (210, 210, 210), -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        # active/triggered cells with decay based on last trigger time
        sim_t = self.engine.sim_time
        for idx, row in self.engine.events.iterrows():
            dt = sim_t - float(row["last_trigger_time"])
            if dt < 0:
                continue

            glow = max(0.0, 1.0 - dt / 1.2)
            if glow <= 0:
                continue

            x = int(row["cx"])
            y = int(row["cy"])
            label = row["label"]
            color = self._cell_color_bgr(label)

            r_outer = int(14 + 14 * glow)
            r_inner = int(5 + 6 * glow)

            self._draw_transparent_circle(frame, (x, y), r_outer, color, alpha=0.10 * glow)
            self._draw_transparent_circle(frame, (x, y), r_inner, color, alpha=0.35 * glow)
            cv2.circle(frame, (x, y), max(2, int(2 + 2 * glow)), color, -1, lineType=cv2.LINE_AA)

    def _draw_particle(self, frame, state):
        x = int(self.engine.x)
        y = int(self.engine.y)

        # main trigger circle
        radius = int(max(8, state.trigger_radius_rbc))
        self._draw_transparent_circle(frame, (x, y), radius, (255, 255, 255), alpha=0.06)

        # particle glow
        self._draw_transparent_circle(frame, (x, y), 18, (255, 255, 255), alpha=0.18)
        self._draw_transparent_circle(frame, (x, y), 9, (255, 245, 200), alpha=0.35)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def _draw_hud(self, frame, state):
        active_fields = get_active_controlled_fields(CC_MAP)

        margin = 12
        panel_w = 220
        panel_h = 102

        panels = {
            "MUSIC": (margin, margin),
            "FLOW": (self.W - panel_w - margin, margin),
            "FIELD / TRIGGER": (margin, self.H - panel_h - margin),
        }

        for panel_title, fields in PANEL_FIELDS.items():
            visible_fields = [f for f in fields if f in active_fields]
            if not visible_fields:
                continue

            x, y = panels[panel_title]
            self._draw_panel(frame, x, y, panel_w, panel_h, alpha=0.34)

            lines = []
            for field_name in visible_fields:
                raw_value = getattr(state, field_name)
                val = format_field_value(field_name, raw_value)
                label = FIELD_LABELS.get(field_name, field_name)
                lines.append(f"{label:<10} {val}")

            self._draw_panel_text(
                frame,
                lines,
                x + 10,
                y + 18,
                title=panel_title,
                line_h=14
            )

    def update(self, state):
    # if the window was closed manually, stop
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                return False
        except cv2.error:
            return False

        frame = self.base_frame.copy()

        self.trail.append((self.engine.x, self.engine.y))
        if len(self.trail) > self.trail_len:
            self.trail = self.trail[-self.trail_len:]

        self._draw_cells(frame)
        self._draw_trail(frame)
        self._draw_particle(frame, state)
        self._draw_hud(frame, state)

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            return False
        return True

    def close(self):
        cv2.destroyWindow(self.window_name)


def load_source_data():
    """
    Return:
    - events dataframe
    - img (PIL image or None)
    """
    if USE_BCCD:
        print(f"[SOURCE] Using BCCD zip: {BCCD_ZIP_PATH}")
        print(f"[SOURCE] Image ID: {BCCD_IMAGE_ID}")

        bccd = BCCD(BCCD_ZIP_PATH)
        img = bccd.load_image(BCCD_IMAGE_ID)
        df_cells, _ = bccd.load_annotations(BCCD_IMAGE_ID)
        events = build_events_basic(df_cells)

        return events, img

    else:
        print(f"[SOURCE] Using CSV events: {EVENTS_CSV}")
        events = pd.read_csv(EVENTS_CSV)
        img = None
        return events, img


# =========================================================
# MAIN
# =========================================================

def main():
    global CC_MAP, ACTION_NOTE_MAP

    print("Available MIDI ports:")
    list_ports()

    events, img = load_source_data()

    state = LiveState()
    engine = GhostParticleEngine(events, state)

    print("\nResolving MIDI ports...")
    input_names = mido.get_input_names()
    output_names = mido.get_output_names()

    in_name = find_port_name(CONTROLLER_INPUT_NAME, input_names)
    out_name = find_port_name(MIDI_OUTPUT_NAME, output_names)

    print("Using controller input :", in_name)
    print("Using MIDI output      :", out_name)

    print("\nOpening ports...")
    in_port = mido.open_input(in_name)
    out_port = mido.open_output(out_name)

    if LEARN_MODE or not Path(MAPPING_FILE).exists():
        print("\n[INFO] MIDI Learn active: live visualizer will not open.")
        CC_MAP = learn_cc_mapping(in_port, PARAM_SPECS)
        ACTION_NOTE_MAP = learn_action_notes(in_port, ACTION_SPECS)
        save_midi_mappings(CC_MAP, ACTION_NOTE_MAP, MAPPING_FILE)
    else:
        CC_MAP, loaded_action_map = load_midi_mappings(MAPPING_FILE)
        if loaded_action_map:
            ACTION_NOTE_MAP = loaded_action_map

    print_cc_mapping_table(CC_MAP)
    print_action_mapping_table(ACTION_NOTE_MAP)

    visualizer = None
    if VISUALIZE and not LEARN_MODE:
        visualizer = OpenCVVisualizer(img, engine.events, engine)

    print("Streaming started. Ctrl+C to exit.\n")

    engine.all_notes_off(out_port)

    frame_count = 0

    try:
        while True:
            process_controller_messages(in_port, state, engine, out_port)
            engine.step(out_port, dt=DT)

            if frame_count % 50 == 0:
                n_trig = int(engine.events["triggered"].sum())
                print(
                    f"[STATUS] sim_time={engine.sim_time:.2f} "
                    f"x={engine.x:.1f} y={engine.y:.1f} "
                    f"triggered={n_trig}/{len(engine.events)} "
                    f"r_rbc={state.trigger_radius_rbc:.1f} "
                    f"r_wbc={state.trigger_radius_wbc:.1f} "
                    f"cooldown={state.retrigger_cooldown:.2f}"
                )

            if visualizer is not None and (frame_count % VISUAL_UPDATE_EVERY == 0):
                keep_running = visualizer.update(state)
                if not keep_running:
                    print("\nVisualizer closed by user.")
                    break

            frame_count += 1
            time.sleep(DT)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        engine.all_notes_off(out_port)
        out_port.close()
        in_port.close()

        if visualizer is not None:
            visualizer.close()


if __name__ == "__main__":
    main()
