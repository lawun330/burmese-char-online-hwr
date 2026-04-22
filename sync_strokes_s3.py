#!/usr/bin/env python3
"""
Manual sync stroke dataset files to/from S3.

Why S3?:
  - Strokes are already plain .txt files on disk; S3 keeps that shape with minimal change.

Required env:
  - S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

Install:
  pip install -r requirements-s3.txt

Usage:
  python sync_strokes_s3.py upload --dataset dataset
  python sync_strokes_s3.py download --dataset dataset
  python sync_strokes_s3.py list
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from auto_save_strokes_s3 import get_s3_client, normalize_s3_prefix


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=False)


def _bucket() -> str:
    b = os.environ.get("S3_BUCKET", "").strip()
    if not b:
        raise SystemExit("Set S3_BUCKET environment variable")
    return b


def _prefix() -> str:
    return normalize_s3_prefix(os.environ.get("S3_PREFIX", ""))


def _s3_key(rel_path: str) -> str:
    """rel_path uses forward slashes, no leading slash."""
    return _prefix() + rel_path.replace("\\", "/")


def cmd_upload(args: argparse.Namespace) -> None:
    from botocore.exceptions import ClientError

    root = Path(args.dataset).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    s3 = get_s3_client()
    bucket = _bucket()
    uploaded = 0
    skipped = 0

    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".txt"):
                continue
            local = Path(dirpath) / name
            try:
                rel = local.relative_to(root)
            except ValueError:
                continue
            key = _s3_key(str(rel).replace(os.sep, "/"))

            if args.skip_existing:
                try:
                    s3.head_object(Bucket=bucket, Key=key)
                    skipped += 1
                    continue
                except ClientError as e:
                    status = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
                    code = e.response.get("Error", {}).get("Code", "")
                    if status == 404 or code in ("404", "NotFound", "NoSuchKey"):
                        pass  # upload
                    else:
                        raise

            extra = {"ContentType": "text/plain; charset=utf-8"}
            s3.upload_file(str(local), bucket, key, ExtraArgs=extra)
            uploaded += 1
            if args.verbose:
                print(f"upload {local} -> s3://{bucket}/{key}")

    print(f"upload done: {uploaded} files, skipped {skipped} (already present)")


def cmd_download(args: argparse.Namespace) -> None:
    root = Path(args.dataset).resolve()
    root.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client()
    bucket = _bucket()
    prefix = _s3_key("")  # dataset-relative root under bucket prefix

    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :] if key.startswith(prefix) else key
            if not rel.endswith(".txt"):
                continue
            local = root / rel.replace("/", os.sep)
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local))
            downloaded += 1
            if args.verbose:
                print(f"download s3://{bucket}/{key} -> {local}")

    print(f"download done: {downloaded} files into {root}")


def cmd_list(_args: argparse.Namespace) -> None:
    s3 = get_s3_client()
    bucket = _bucket()
    prefix = _prefix()

    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            print(obj["Key"])
            n += 1
    print(f"total: {n} objects under prefix {prefix!r}")


def main() -> None:
    _load_env()

    p = argparse.ArgumentParser(description="Upload/download stroke .txt files to S3")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("upload", help="Upload local dataset tree to S3")
    u.add_argument("--dataset", default="dataset", help="Local dataset root (default: dataset)")
    u.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip upload if object already exists (HEAD check)",
    )
    u.add_argument("-v", "--verbose", action="store_true")
    u.set_defaults(func=cmd_upload)

    d = sub.add_parser("download", help="Download objects from S3 into local dataset tree")
    d.add_argument("--dataset", default="dataset", help="Local dataset root (default: dataset)")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=cmd_download)

    l = sub.add_parser("list", help="List object keys under S3_PREFIX")
    l.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
