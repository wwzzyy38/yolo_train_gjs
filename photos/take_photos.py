"""Capture 640x480 photos from a selected camera."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2


DEFAULT_CAMERA_INDEX = 0
DEFAULT_OUTPUT_DIR = Path("photos")
WINDOW_NAME = "Camera - press Enter to take a photo, q to quit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture 640x480 photos from a camera."
    )
    parser.add_argument(
        "--camera",
        "--camera-index",
        dest="camera_index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Camera index (default: {DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Photo output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not try to show a window; capture photos by pressing Enter in the terminal.",
    )
    return parser.parse_args()


def make_photo_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    photo_path = output_dir / f"photo_{timestamp}.jpg"
    counter = 1
    while photo_path.exists():
        photo_path = output_dir / f"photo_{timestamp}_{counter}.jpg"
        counter += 1
    return photo_path


def save_photo(frame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    photo_path = make_photo_path(output_dir)
    if not cv2.imwrite(str(photo_path), frame):
        raise RuntimeError(f"保存照片失败: {photo_path}")
    print(f"已拍照: {photo_path}", flush=True)
    return photo_path


def try_open_window() -> bool:
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        return True
    except cv2.error:
        return False


def capture_with_gui(capture: cv2.VideoCapture, output_dir: Path) -> None:
    print("摄像头画面已打开，按回车拍照，按 q 退出。", flush=True)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("读取摄像头画面失败")

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (10, 13):
                save_photo(frame, output_dir)
            elif key == ord("q"):
                break
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def capture_without_gui(capture: cv2.VideoCapture, output_dir: Path) -> None:
    print("图形化窗口不可用，进入终端模式。按回车拍照，输入 q 后回车退出。", flush=True)
    while True:
        command = input("> ")
        if command.strip().lower() == "q":
            break

        success, frame = capture.read()
        if not success:
            raise RuntimeError("读取摄像头画面失败")
        save_photo(frame, output_dir)


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头，编号: {args.camera_index}")

    # Set both properties explicitly; some cameras may use the nearest supported mode.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.2)

    try:
        if not args.no_gui and try_open_window():
            try:
                capture_with_gui(capture, args.output_dir)
            except cv2.error:
                print("图形化窗口不可用，切换到终端模式。", file=sys.stderr)
                capture_without_gui(capture, args.output_dir)
        else:
            capture_without_gui(capture, args.output_dir)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
