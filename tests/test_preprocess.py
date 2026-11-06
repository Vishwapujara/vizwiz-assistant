import io

from PIL import Image

from vizwiz.serve.preprocess import load_image, prepare_image, target_size, visual_tokens


def test_target_size_caps_pixels_and_rounds_to_32():
    w, h = target_size(1968, 2624, 262_144)
    assert w % 32 == 0 and h % 32 == 0
    assert w * h <= 262_144
    assert abs(w / h - 1968 / 2624) < 0.05  # aspect ratio kept


def test_never_upscales_and_native_rounds_down():
    assert target_size(640, 480, 1_048_576) == (640, 480)
    assert target_size(650, 490, None) == (640, 480)


def test_visual_tokens():
    assert visual_tokens(640, 480) == 20 * 15


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (400, 200), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    loaded = load_image(buf.getvalue())
    assert loaded.size == (200, 400)
    assert prepare_image(buf.getvalue(), None).size == (192, 384)
