"""Vision core tests on synthetic images (no camera, no model weights).

Run:  python -m pytest src/poker_vision/test -q
"""
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poker_vision.core.cards import class_to_code, pair_corners          # noqa: E402
from poker_vision.core.detector import Detection, MockBackend            # noqa: E402
from poker_vision.core.pot import PotEstimator                           # noqa: E402
from poker_vision.core.reader import CardReader, PotReader               # noqa: E402
from poker_vision.core.zones import Zone                                  # noqa: E402
from poker_vision.core.table_frame import MarkerSpec, TableFrame, aruco_dictionary  # noqa: E402

DICT = aruco_dictionary("DICT_4X4_50")


def render_table(H_table_to_px, markers, shape=(1200, 1600)):
    """White table with ArUco markers drawn where H puts them (perspective-correct)."""
    img = np.full(shape + (3,), 230, np.uint8)
    for m in markers:
        side = 400
        tag = cv2.aruco.generateImageMarker(DICT, m.id, side)
        tag = cv2.copyMakeBorder(tag, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
        src = np.float32([[0, 0], [side, 0], [side, side], [0, side]])          # TL TR BR BL
        dst = cv2.perspectiveTransform(m.corners_table().reshape(-1, 1, 2), H_table_to_px).reshape(-1, 2)
        M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        warped = cv2.warpPerspective(cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR), M, (shape[1], shape[0]),
                                     borderValue=(255, 255, 255))
        mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), M, (shape[1], shape[0]))
        # white quiet zone around the tag
        quiet = cv2.dilate(mask, np.ones((25, 25), np.uint8))
        img[quiet > 0] = 255
        img[mask > 0] = warped[mask > 0]
    return img


def tilted_camera():
    """Table metres -> pixels: 2500 px/m, rotated 12 deg, with perspective."""
    a = math.radians(12)
    s = 2500.0
    H = np.array([[s * math.cos(a), s * math.sin(a), 800.0],
                  [s * math.sin(a), -s * math.cos(a), 600.0],
                  [0.15, 0.25, 1.0]])
    return H


def table_cfg(markers, mode="homography"):
    return {"mode": mode, "dictionary": "DICT_4X4_50", "smoothing": 0.0,
            "markers": [{"id": m.id, "size": m.size, "x": m.x, "y": m.y,
                         "yaw_deg": math.degrees(m.yaw)} for m in markers]}


# --------------------------------------------------------------------------- table frame
def test_homography_recovers_table_points_under_perspective():
    markers = [MarkerSpec(0, 0.05), MarkerSpec(1, 0.05, 0.20, 0.0), MarkerSpec(2, 0.05, 0.0, -0.15)]
    H = tilted_camera()
    img = render_table(H, markers)
    tf = TableFrame(table_cfg(markers))
    assert tf.update(img)
    assert tf.visible_ids == [0, 1, 2]
    truth = np.array([[0.10, -0.05], [-0.08, 0.06], [0.18, -0.12], [0.0, 0.0]])
    px = cv2.perspectiveTransform(truth.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(tf.to_table(px) - truth, axis=1)
    assert err.max() < 0.003, err                       # < 3 mm across the table
    back = tf.to_pixel(truth)
    assert np.abs(back - px).max() < 2.0


def test_single_marker_far_points_degrade_gracefully():
    markers = [MarkerSpec(0, 0.05)]
    H = tilted_camera()
    tf = TableFrame(table_cfg(markers))
    assert tf.update(render_table(H, markers))
    truth = np.array([[0.15, -0.10]])
    px = cv2.perspectiveTransform(truth.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.linalg.norm(tf.to_table(px) - truth) < 0.02  # one 5 cm tag extrapolated 18 cm: ~cm level


def test_scale_mode_matches_notebook_formula_top_down():
    s = 2000.0
    H = np.array([[s, 0, 700.0], [0, -s, 500.0], [0, 0, 1.0]])        # straight down, aligned
    markers = [MarkerSpec(0, 0.05)]
    img = render_table(H, markers)
    tf_scale = TableFrame(table_cfg(markers, "scale"))
    tf_h = TableFrame(table_cfg(markers))
    assert tf_scale.update(img) and tf_h.update(img)
    # notebook: m_per_px from perimeter, offset from marker centre, y flipped
    corners, ids, _ = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters()).detectMarkers(img)
    m_per_px = 0.05 / (cv2.arcLength(corners[0], True) / 4.0)
    centre = np.mean(corners[0][0], axis=0)
    target = (333.0, 1175.0 - 400)
    nb = ((target[0] - centre[0]) * m_per_px, -(target[1] - centre[1]) * m_per_px)
    # sub-pixel corner refinement makes ours slightly different from the raw-corner notebook value
    assert np.allclose(tf_scale.to_table([target])[0], nb, atol=0.003)
    assert np.allclose(tf_h.to_table([target])[0], nb, atol=0.003)


def test_yawed_marker_and_offset_origin():
    markers = [MarkerSpec(5, 0.06, 0.30, 0.10, math.radians(90))]
    H = tilted_camera()
    tf = TableFrame(table_cfg(markers))
    assert tf.update(render_table(H, markers))
    truth = np.array([[0.32, 0.12]])
    px = cv2.perspectiveTransform(truth.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.linalg.norm(tf.to_table(px) - truth) < 0.003


def test_no_marker_not_ready():
    tf = TableFrame(table_cfg([MarkerSpec(0, 0.05)]))
    assert not tf.update(np.full((480, 640, 3), 200, np.uint8))
    assert not tf.ready
    with pytest.raises(RuntimeError):
        tf.to_table([[1, 2]])


# --------------------------------------------------------------------------- cards
@pytest.mark.parametrize("name,code", [("AC", "Ac"), ("5D", "5d"), ("10H", "Th"), ("TS", "Ts"),
                                       ("qh", "Qh"), ("ace_of_spades", "As"), ("10-hearts", "Th"),
                                       ("chip_5", None), ("JQ", None), ("X", None)])
def test_class_names(name, code):
    assert class_to_code(name) == code


def test_pairing_uses_table_distance_and_keeps_lone_corners():
    def det(name, cx, cy, conf=0.9):
        return Detection(name, conf, np.array([cx - 20, cy - 30, cx + 20, cy + 30], float))
    dets = [det("AC", 100, 100), det("AC", 250, 300),          # one card, two corners (0.1 m apart)
            det("5D", 900, 100),                               # lone corner
            det("9C", 1200, 900), det("9C", 1250, 950, 0.7),   # same card again, close
            det("chip_5", 50, 50)]                             # not a card
    to_table = lambda px: np.asarray(px) / 2500.0               # 2500 px per metre
    cards = {c.code: c for c in pair_corners(dets, to_table=to_table)}
    assert set(cards) == {"Ac", "5d", "9c"}
    assert cards["Ac"].corners == 2 and np.allclose(cards["Ac"].pixel, [175, 200])
    assert np.allclose(cards["Ac"].table, [0.07, 0.08])
    assert cards["5d"].corners == 1
    # far apart in metres -> not merged; duplicate code keeps the 2-corner reading
    far = pair_corners([det("KH", 0, 0), det("KH", 2000, 0)], to_table=to_table)
    assert len(far) == 1 and far[0].corners == 1


def test_notebook_example_pairs():
    # four corner boxes from the notebook output pattern: AC twice, 5D once
    dets = [Detection("AC", 0.9, np.array([280, 860, 320, 920.])), Detection("AC", 0.8, np.array([310, 890, 350, 950.])),
            Detection("5D", 0.9, np.array([230, 380, 280, 430.]))]
    cards = pair_corners(dets)
    assert [c.code for c in cards] == ["5d", "Ac"]


# --------------------------------------------------------------------------- pot
def chip(name, x, y, conf=0.9):
    poly = np.array([[x, y], [x + 300, y], [x + 300, y + 12], [x, y + 12]], float)
    return Detection(name, conf, np.array([x, y, x + 300, y + 12], float), poly)


def test_pot_instance_counts_small_stakes_and_ignores_black():
    cfg = {"count_mode": "instance", "class_values": {"White": 1, "Red": 5, "Blue": 10}}
    est = PotEstimator(cfg)
    dets = [chip("Blue", 230, 560), chip("Blue", 230, 580), chip("Blue", 230, 600),       # 30
            chip("White", 930, 560), chip("White", 930, 580),                            # 2
            chip("Red", 1240, 560), chip("Red", 1240, 580), chip("Red", 1240, 600),      # 15
            chip("Black", 600, 560)]                                                     # ignored
    r = est.estimate(dets)
    assert r.value == 47 and r.counts == {"Blue": 3, "White": 2, "Red": 3}
    assert est.estimate([]).value == 0 and est.estimate([]).confidence == 1.0


def test_pot_stack_height_mode():
    est = PotEstimator({"count_mode": "stack_height", "px_per_chip": 6.0, "class_values": {"Red": 5}})
    d = Detection("Red", 0.8, np.array([0, 100, 40, 130.]))            # 30 px tall -> 5 chips
    assert est.estimate([d]).value == 25


def test_pot_reader_crops_roi():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    mock = MockBackend({"mock_detections": [
        {"name": "Blue", "xyxy": [230, 560, 530, 572]},          # inside roi
        {"name": "Red", "xyxy": [1240, 560, 1540, 572]},          # inside roi
        {"name": "White", "xyxy": [100, 100, 200, 112]}]})       # background, outside roi
    reader = PotReader(mock, {"class_values": {"White": 1, "Red": 5, "Blue": 10}, "roi": [150, 400, 1700, 750]})
    r = reader.read(frame)
    assert r.success and r.reading.value == 15 and r.crop == (150, 400, 1700, 750)
    assert all(150 <= d.xyxy[0] for d in r.detections)          # boxes back in full-image pixels


# --------------------------------------------------------------------------- on-demand card reads
def _scene():
    markers = [MarkerSpec(0, 0.05), MarkerSpec(1, 0.05, 0.25, 0.0)]
    H = tilted_camera()
    return H, render_table(H, markers), markers


def _corner_boxes(H, name, cx, cy, diag=(0.015, 0.03)):
    c = cv2.perspectiveTransform(np.array([[[cx - diag[0], cy + diag[1]]], [[cx + diag[0], cy - diag[1]]]]),
                                 H).reshape(-1, 2)
    return [{"name": name, "xyxy": [float(u - 12), float(v - 20), float(u + 12), float(v + 20)], "conf": 0.9}
            for u, v in c]


def test_card_reader_crops_zone_filters_and_counts():
    H, img, markers = _scene()
    zones = {"board": Zone("board", 0.04, 0.24, -0.16, -0.02), "bot_hole": Zone("bot_hole", -0.10, 0.0, -0.16, -0.06)}
    dets = (_corner_boxes(H, "AH", 0.08, -0.08) + _corner_boxes(H, "10D", 0.15, -0.08)
            + _corner_boxes(H, "QS", 0.21, -0.09) + _corner_boxes(H, "2C", -0.05, -0.11))   # 2C is in bot_hole
    tf = TableFrame(table_cfg(markers))
    reader = CardReader(tf, MockBackend({"mock_detections": dets}), zones, {"crop_margin_m": 0.02})
    r = reader.read([img], "board", 3)
    assert r.success, r.message
    assert [c.code for c in r.cards] == ["Ah", "Qs", "Td"]
    x0, y0, x1, y1 = r.crop
    assert (x1 - x0) * (y1 - y0) < img.shape[0] * img.shape[1] / 4      # a real crop, not the table
    near = {c.code: c.table for c in r.cards}
    assert np.linalg.norm(near["Ah"] - [0.08, -0.08]) < 0.005
    bot = reader.read([img], "bot_hole", 2)
    assert not bot.success and [c.code for c in bot.cards] == ["2c"] and "expected 2" in bot.message
    assert not reader.read([img], "nowhere").success


def test_conflicting_corners_merge_diagonally_not_sideways():
    H, img, markers = _scene()
    zones = {"board": Zone("board", 0.04, 0.24, -0.16, -0.02)}
    nine = _corner_boxes(H, "9S", 0.10, -0.08)
    nine[1]["name"], nine[1]["conf"] = "9C", 0.68          # bottom corner misread, as in the sample image
    side = [_corner_boxes(H, "7D", 0.16, -0.08)[0], _corner_boxes(H, "7H", 0.22, -0.08)[0]]  # neighbours, same row
    tf = TableFrame(table_cfg(markers))
    reader = CardReader(tf, MockBackend({"mock_detections": nine + side}), zones, {})
    r = reader.read([img], "board", 0)
    codes = {c.code: c for c in r.cards}
    assert codes["9s"].conflict == "9s/9c" and codes["9s"].corners == 2
    assert "7d" in codes and "7h" in codes and not codes["7d"].conflict      # side by side: not merged
    strict = CardReader(tf, MockBackend({"mock_detections": nine}), zones, {"conflict_policy": "reject"})
    assert not strict.read([img], "board", 1).success


def test_card_reader_without_marker_fails_cleanly():
    tf = TableFrame(table_cfg([MarkerSpec(0, 0.05)]))
    reader = CardReader(tf, MockBackend({}), {"board": Zone("board", 0, 1, 0, 1)}, {})
    r = reader.read([np.full((480, 640, 3), 200, np.uint8)], "board", 3)
    assert not r.success and "marker" in r.message


def test_multi_shot_vote():
    from poker_vision.core.cards import Card
    a = Card("Ah", np.zeros(2), 0.9, 2, np.zeros(2))
    b = Card("Kd", np.zeros(2), 0.9, 2, np.zeros(2))
    voted = CardReader._vote([[a, b], [a], [a, b]])
    assert [c.code for c in voted] == ["Ah", "Kd"]
    assert [c.code for c in CardReader._vote([[a, b], [a], [a]])] == ["Ah"]
