"""Nox fork: the plain `look` / `listen` commands (wrappers live in ~/.local/bin)."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys

from .errors import SensoryError, error_result


def _kind_for_look(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or ""
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name:format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=30, check=False,
    )
    try:
        info = json.loads(probe.stdout or "{}")
    except ValueError:
        info = {}
    duration = float((info.get("format") or {}).get("duration") or 0)
    return "video" if info.get("streams") and duration > 0.5 else "image"


def main(command: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=command)
    parser.add_argument("path")
    parser.add_argument("-q", "--question", default="")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--detail", choices=("auto", "quick", "detailed"), default="auto")
    parser.add_argument("--language", default="zh-CN")
    parser.add_argument("--json", action="store_true", help="full structured report")
    if command == "look":
        parser.add_argument("--no-audio", action="store_true", help="video: picture only")
    else:
        parser.add_argument("--music", action="store_true", help="listen as music")
    args = parser.parse_args(argv)

    from .api import default_services, sense, stage

    try:
        services = default_services()
        source = stage(args.path, services)
    except FileNotFoundError:
        print(f"{command}: 找不到文件 {args.path}", file=sys.stderr)
        return 2
    options: dict[str, object] = {
        "question": args.question,
        "detail": args.detail,
        "language": args.language,
    }
    if command == "look":
        kind = _kind_for_look(str(source))
        if kind == "video":
            options.update(
                start_seconds=args.start,
                end_seconds=args.end,
                include_audio=not args.no_audio,
            )
        elif args.start is not None or args.end is not None:
            print(f"{command}: 图片没有 --start/--end", file=sys.stderr)
            return 2
    else:
        kind = "music" if args.music else "audio"
        options.update(start_seconds=args.start, end_seconds=args.end)
    try:
        result = sense(kind, str(source), **options)  # type: ignore[arg-type]
    except SensoryError as error:
        print(json.dumps(error_result(error), ensure_ascii=False), file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"{command}: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["text"])
    return 0
