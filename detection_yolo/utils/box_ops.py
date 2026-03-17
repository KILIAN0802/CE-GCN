import numpy as np

def expand_box(xyxy, expand_ratio, img_w, img_h):
    """
    xyxy: [x1, y1, x2, y2]
    expand_ratio: scale box (ví dụ 1.4 để phóng rộng)
    """

    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    new_w = w * expand_ratio
    new_h = h * expand_ratio

    nx1 = max(0, cx - new_w / 2)
    ny1 = max(0, cy - new_h / 2)
    nx2 = min(img_w - 1, cx + new_w / 2)
    ny2 = min(img_h - 1, cy + new_h / 2)

    return [int(nx1), int(ny1), int(nx2), int(ny2)]


def merge_boxes(boxes):
    """
    Gộp nhiều bounding boxes thành 1 bbox lớn.
    boxes: list [[x1,y1,x2,y2], ...]
    """

    if len(boxes) == 0:
        return None

    boxes = np.array(boxes)
    x1 = np.min(boxes[:, 0])
    y1 = np.min(boxes[:, 1])
    x2 = np.max(boxes[:, 2])
    y2 = np.max(boxes[:, 3])

    return [int(x1), int(y1), int(x2), int(y2)]
