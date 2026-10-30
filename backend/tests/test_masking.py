import cv2
import numpy as np
import pytest
from fakes import polygon_mask, rotated_rect

from backend import masking


@pytest.fixture
def noise() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)


def near(mask: np.ndarray, pixels: int = 2) -> np.ndarray:
    """마스크 경계 근처까지 넓힌 마스크. 경계 픽셀은 안쪽으로 칠지 바깥으로 칠지 래스터화에 따라 다르다."""
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a != b).any(axis=2)


@pytest.mark.parametrize("angle", [0, 20, -35])
def test_only_inside_changes(noise, angle):
    polygon = rotated_rect(160, 120, 180, 40, angle)
    result = masking.blur(noise, [polygon])
    inside = polygon_mask(polygon, (320, 240))
    diff = changed(result, noise)
    assert not diff[~near(inside)].any()
    assert diff[inside].mean() > 0.95


def test_does_not_modify_input(noise):
    original = noise.copy()
    result = masking.blur(noise, [rotated_rect(100, 100, 80, 30, 0)])
    assert np.array_equal(noise, original)
    assert result is not noise


def test_no_polygons_returns_same_pixels(noise):
    assert np.array_equal(masking.blur(noise, []), noise)


def test_polygon_partly_outside_image(noise):
    polygon = [[-50, -20], [100, -20], [100, 40], [-50, 40]]
    result = masking.blur(noise, [polygon])
    inside = polygon_mask(polygon, (320, 240))
    diff = changed(result, noise)
    assert diff[0:39, 0:99].mean() > 0.95
    assert not diff[~near(inside)].any()


def test_text_becomes_unreadable():
    # 높이 40의 글자 획(4px 간격 세로 줄무늬)이 거의 평평해진다
    image = np.full((200, 400, 3), 255, np.uint8)
    image[80:120, 100:300][:, ::8] = 0
    image[80:120, 100:300][:, 1::8] = 0
    polygon = [[96, 76], [304, 76], [304, 124], [96, 124]]
    before = image[84:116, 110:290].std()
    after = masking.blur(image, [polygon])[84:116, 110:290].std()
    assert after < 0.2 * before


def test_kernel_uses_short_side_of_tilted_polygon():
    # 20° 기울어진 200×40: 수평 외접 사각형 높이(약 106)가 아니라 짧은 변(40) 기준
    polygon = rotated_rect(200, 200, 200, 40, 20)
    size = masking.kernel_size(polygon)
    assert size % 2 == 1
    assert 40 <= size <= 43


def test_kernel_has_minimum():
    assert masking.kernel_size(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)) == 3


def test_rgba_image(noise):
    rgba = np.dstack([noise, np.full(noise.shape[:2], 255, np.uint8)])
    result = masking.blur(rgba, [rotated_rect(160, 120, 100, 30, 0)])
    assert result.shape == rgba.shape
    assert (result[:, :, 3] == 255).all()
