"""
Drawing for the camera detectors' debug images.

The two Ultralytics nodes draw with this; the DeepStream nodes carry their own
drawing, which this matches -- people blue, balls yellow, anything the node did
not publish red with the reason, skeletons yellow with green joints -- so an
image reads the same whichever detector produced it.

No ROS here: the nodes wrap the JPEG bytes in a `CompressedImage` themselves.
"""

import math

import cv2

# BGR, as OpenCV draws.
COLOUR_PERSON = (255, 0, 0)
COLOUR_BALL = (0, 220, 220)
COLOUR_REJECTED = (0, 0, 255)
COLOUR_BONE = (0, 255, 255)
COLOUR_JOINT = (0, 255, 0)

# Standard YOLO pose skeleton connections, as indices into the 17 COCO joints.
SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Head/Face
    (5, 6),  # Shoulders
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # Arms
    (11, 12),
    (5, 11),
    (6, 12),  # Torso/Hips
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # Legs
]


def draw_box(image, corners, colour, label):
    """Draw one `(xmin, ymin, xmax, ymax)` box with a filled label on its top edge."""
    x1, y1, x2, y2 = (int(round(v)) for v in corners)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )
    # Pushed down into the image when the box touches the top edge.
    label_y = max(y1, text_h + baseline + 4)
    cv2.rectangle(
        image,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 8, label_y + 2),
        colour,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 4, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def visible_joints(keypoints, min_conf):
    """
    Return which of `(x, y, conf)` joints count as found.

    A joint the model did not find comes back as `(0, 0)` from Ultralytics and
    as `NaN` from DeepStream; both are missing, whatever their confidence says.
    `conf` may be `None` for a model exported without visibility scores.
    """
    visible = []
    for x, y, conf in keypoints:
        if not (math.isfinite(x) and math.isfinite(y)) or (x == 0.0 and y == 0.0):
            visible.append(False)
        elif conf is None:
            visible.append(True)
        else:
            visible.append(math.isfinite(conf) and conf > min_conf)
    return visible


def draw_skeleton(image, keypoints, min_conf):
    """Draw bones between found joints, then each joint with its confidence."""
    visible = visible_joints(keypoints, min_conf)
    for p1, p2 in SKELETON_CONNECTIONS:
        if p1 >= len(keypoints) or p2 >= len(keypoints):
            continue
        if visible[p1] and visible[p2]:
            cv2.line(
                image,
                (int(keypoints[p1][0]), int(keypoints[p1][1])),
                (int(keypoints[p2][0]), int(keypoints[p2][1])),
                COLOUR_BONE,
                2,
            )
    for (x, y, conf), found in zip(keypoints, visible):
        if not found:
            continue
        point = (int(x), int(y))
        cv2.circle(image, point, 4, COLOUR_JOINT, -1)
        if conf is None:
            continue
        text_pos = (point[0] + 6, point[1] - 6)
        label = f"{conf:.2f}"
        # Black under white, so the number reads on any background.
        for colour, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(
                image,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                colour,
                thickness,
                cv2.LINE_AA,
            )


def encode_jpeg(image):
    """Return the image as JPEG bytes, or `None` if OpenCV refuses it."""
    ok, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes() if ok else None
