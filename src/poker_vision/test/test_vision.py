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
def test_pot_stack_height_roi_and_rounding():
    def stack(name, x, top, bottom):
        poly = np.array([[x, top], [x + 40, top], [x + 40, bottom], [x, bottom]], float)
        return Detection(name, 0.8, np.array([x, top, x + 40, bottom], float), poly)
    cfg = {"count_mode": "stack_height", "px_per_chip": 6.0, "round_to": 5,
           "class_values": {"chip_5": 5, "chip_25": 25},
           "roi": [[0, 0], [500, 0], [500, 500], [0, 500]]}
    est = PotEstimator(cfg)
    r = est.estimate([stack("chip_5", 10, 100, 130),     # 5 chips -> 25
                      stack("chip_25", 100, 100, 112),   # 2 chips -> 50
                      stack("chip_25", 800, 100, 160),   # outside ROI
                      stack("dealer_button", 50, 50, 60)])
    assert r.value == 75 and len(r.stacks) == 2
    assert est.estimate([]).value == 0


def test_pot_instance_mode():
    est = PotEstimator({"count_mode": "instance", "class_values": {"chip": 10}})
    d = Detection("chip", 0.9, np.array([0, 0, 10, 10.]))
    assert est.estimate([d, d, d]).value == 30


# --------------------------------------------------------------------------- full overhead pipeline
def test_overhead_pipeline_with_mock_model():
    markers = [MarkerSpec(0, 0.05), MarkerSpec(1, 0.05, 0.25, 0.0)]
    H = tilted_camera()
    img = render_table(H, markers)
    # where two corners of the Ah would appear for a card centred at (0.10, -0.08)
    corners_t = np.array([[0.085, -0.045], [0.115, -0.115]])
    c_px = cv2.perspectiveTransform(corners_t.reshape(-1, 1, 2), H).reshape(-1, 2)
    mock = MockBackend({"mock_detections": [
        {"name": "AH", "xyxy": [c_px[0][0] - 15, c_px[0][1] - 25, c_px[0][0] + 15, c_px[0][1] + 25]},
        {"name": "AH", "xyxy": [c_px[1][0] - 15, c_px[1][1] - 25, c_px[1][0] + 15, c_px[1][1] + 25]},
    ]})
    tf = TableFrame(table_cfg(markers))
    assert tf.update(img)
    cards = pair_corners(mock(img), to_table=tf.to_table)
    assert len(cards) == 1 and cards[0].code == "Ah" and cards[0].corners == 2
    assert np.linalg.norm(cards[0].table - [0.10, -0.08]) < 0.005
