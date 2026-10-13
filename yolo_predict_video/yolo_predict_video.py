"""Run Ultralytics YOLO inference on a video and save an annotated copy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import cv2


# These defaults can also be changed directly in this file.
DEFAULT_VIDEO = Path("input/input.mp4")
DEFAULT_MODEL = Path("best.pt")
DEFAULT_OUTPUT = Path("output/predicted.mp4")
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".m4v", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use an Ultralytics YOLO model to draw predictions on a video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"Input video path (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help=f"YOLO model path or Ultralytics model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output video path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0 or cpu (default: Ultralytics decides)",
    )
    return parser.parse_args()


def resolve_video_path(video: Path) -> Path:
    """Resolve an explicitly supplied video or the only video in a directory."""
    if video.is_file():
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"不支持的视频格式: {video}")
        return video

    if video.is_dir():
        candidates = sorted(
            path for path in video.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"目录中没有找到视频文件: {video}")
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"目录中找到多个视频，请用 --video 指定一个: {names}")

    raise FileNotFoundError(f"输入视频不存在: {video}")


def create_writer(output: Path, capture: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("无法读取输入视频的宽高")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output}")
    return writer


def predict_video(
    video_path: Path,
    model_path: str,
    output_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
) -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "未安装 ultralytics，请先运行: pip install ultralytics"
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频: {video_path}")

    writer = create_writer(output_path, capture)
    model = YOLO(model_path)
    results: Iterator = model.predict(
        source=str(video_path),
        stream=True,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    try:
        for result in results:
            writer.write(result.plot())
    finally:
        writer.release()
        capture.release()

    print(f"预测完成，输出文件: {output_path}")


def main() -> None:
    args = parse_args()
    if not 0 < args.conf <= 1:
        raise ValueError("--conf 必须在 (0, 1] 范围内")
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须大于 0")

    video_path = resolve_video_path(args.video)
    predict_video(
        video_path=video_path,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
