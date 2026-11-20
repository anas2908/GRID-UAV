import math
import re

import numpy as np
from scipy.spatial.transform import Rotation

NUM_BINS = 99
ACTION_RANGES = {
    "forward": (0.0, 5.0),
    "down": (-5.0, 5.0),
    "yaw": (-1.1, 1.1),
}
LABEL_KEYS = (("fwd", "forward"), ("down", "down"), ("yaw", "yaw"))


def quantize(value, axis):
    low, high = ACTION_RANGES[axis]
    value = np.clip(value, low, high)
    return int((value - low) / (high - low) * (NUM_BINS - 1))


def dequantize(index, axis):
    low, high = ACTION_RANGES[axis]
    index = max(0, min(NUM_BINS - 1, index))
    return index / (NUM_BINS - 1) * (high - low) + low


def action_text(label, land):
    text = " ".join(f"{quantize(label[key], axis):02d}" for key, axis in LABEL_KEYS)
    return text + " LAND" if land else text


def prompt_text(instruction):
    return f"{instruction.strip()}\nAction: "


def parse_action(text):
    output = text.split("Action:")[-1]
    numbers = re.findall(r"\d+", output)
    forward = down = yaw = 0.0
    valid = len(numbers) >= 3
    if valid:
        bins = [int(value) for value in numbers[-3:]]
        forward = dequantize(bins[0], "forward")
        down = dequantize(bins[1], "down")
        yaw = dequantize(bins[2], "yaw")
    land = "LAND" in text or (forward < 0.01 and abs(down) < 0.01 and abs(yaw) < 0.01)
    return {"fwd": forward, "down": down, "yaw": yaw, "land": land, "valid": valid}


def direction_hint(position, orientation, target):
    if isinstance(orientation, dict):
        orientation = [orientation.get("x", 0), orientation.get("y", 0),
                       orientation.get("z", 0), orientation.get("w", 1)]
    offset = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    if float(np.linalg.norm(offset[:2])) < 0.01:
        return ""
    body = Rotation.from_quat(orientation).inv().apply(offset)
    angle = math.degrees(math.atan2(float(body[1]), float(body[0])))
    if -15 <= angle <= 15:
        return "straight ahead "
    if 15 < angle <= 60:
        return "forward-right "
    if 60 < angle <= 120:
        return "to your right "
    if 120 < angle <= 180:
        return "to your right rear "
    if -60 <= angle < -15:
        return "forward-left "
    if -120 <= angle < -60:
        return "to your left "
    if -180 <= angle < -120:
        return "to your left rear "
    return ""


def object_description(instruction):
    marker, suffix = "degrees from you.", " Please control"
    if marker not in instruction or suffix not in instruction:
        return instruction.strip()
    return instruction.split(marker, 1)[1].split(suffix, 1)[0].strip()


def navigation_prompt(hint, description):
    return f"<image>\nFly {hint}and find the target. {description}\nAction: "
