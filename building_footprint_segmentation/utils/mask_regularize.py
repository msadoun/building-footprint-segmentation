"""Regularize soft binary building masks into sharp polygonal footprints."""

from __future__ import annotations

import cv2
import numpy as np


def _to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array.squeeze()
    if array.dtype != np.uint8:
        array = (array > 0.5).astype(np.uint8) * 255
    else:
        array = (array > 0).astype(np.uint8) * 255
    return array


def _normalize_rect_angle(angle_deg: float) -> float:
    """Map OpenCV minAreaRect angle into roughly [-45, 45]."""
    angle = float(angle_deg)
    while angle < -45:
        angle += 90
    while angle > 45:
        angle -= 90
    return angle


def _orthogonalize_contour(contour: np.ndarray) -> np.ndarray:
    """
    Snap a contour toward right angles using the building's dominant orientation.
    """
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return contour

    rect = cv2.minAreaRect(points)
    center = rect[0]
    angle = _normalize_rect_angle(rect[2])

    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.transform(points.reshape(1, -1, 2), rotation).reshape(-1, 2)

    perimeter = cv2.arcLength(rotated.astype(np.float32), True)
    epsilon = max(1.0, 0.01 * perimeter)
    approx = cv2.approxPolyDP(rotated.astype(np.float32), epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        box = cv2.boxPoints(rect).astype(np.float32)
        return box.reshape(-1, 1, 2)

    ortho = [approx[0].copy()]
    for point in approx[1:]:
        previous = ortho[-1]
        delta = point - previous
        if abs(delta[0]) >= abs(delta[1]):
            ortho.append(np.array([point[0], previous[1]], dtype=np.float32))
        else:
            ortho.append(np.array([previous[0], point[1]], dtype=np.float32))

    # Close the ring with an orthogonal final segment.
    first = ortho[0]
    last = ortho[-1]
    delta = first - last
    if abs(delta[0]) >= abs(delta[1]):
        ortho.append(np.array([first[0], last[1]], dtype=np.float32))
    else:
        ortho.append(np.array([last[0], first[1]], dtype=np.float32))

    ortho = np.asarray(ortho, dtype=np.float32)
    inverse = cv2.getRotationMatrix2D(center, -angle, 1.0)
    restored = cv2.transform(ortho.reshape(1, -1, 2), inverse).reshape(-1, 2)
    return np.round(restored).astype(np.int32).reshape(-1, 1, 2)


def regularize_binary_mask(
    mask: np.ndarray,
    min_area: int = 64,
    morph_kernel: int = 3,
    use_min_area_rect_fallback: bool = True,
) -> np.ndarray:
    """
    Convert a soft/noisy binary mask into sharp building-like polygons.

    Steps:
      1. Morphological close/open
      2. Contour extraction
      3. Polygon simplification + orthogonalization
      4. Rasterize filled polygons back to a binary mask
    """
    binary = _to_uint8_mask(mask)
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel, morph_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(binary)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(1.0, 0.01 * perimeter), True)

        try:
            polygon = _orthogonalize_contour(approx)
        except Exception:
            if use_min_area_rect_fallback:
                polygon = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
                polygon = polygon.reshape(-1, 1, 2)
            else:
                polygon = approx

        if polygon is None or len(polygon) < 3:
            if use_min_area_rect_fallback:
                polygon = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
                polygon = polygon.reshape(-1, 1, 2)
            else:
                continue

        cv2.fillPoly(output, [polygon], 255)

    return (output > 0).astype(np.uint8) * 255


def regularize_probability_mask(
    probability: np.ndarray,
    threshold: float = 0.20,
    min_area: int = 64,
    morph_kernel: int = 3,
) -> np.ndarray:
    """Threshold a probability map, then regularize to sharp polygons."""
    binary = (np.asarray(probability) >= threshold).astype(np.uint8) * 255
    return regularize_binary_mask(
        binary, min_area=min_area, morph_kernel=morph_kernel
    )
