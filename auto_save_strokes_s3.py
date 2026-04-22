"""
Auto-save stroke dataset files to S3 via Flask web app.

Why S3?:
  - Strokes are already plain .txt files on disk; S3 keeps that shape with minimal change.

Required env:
  - S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

Install:
  pip install -r requirements-s3.txt
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from botocore.exceptions import ClientError


def normalize_s3_prefix(prefix: str) -> str:
    p = (prefix or "").strip().replace("\\", "/")
    if p and not p.endswith("/"):
        p += "/"
    return p


def parse_strokes_from_text(content: str) -> List[List[Tuple[float, float]]]:
    strokes: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("STROKE"):
            if current:
                strokes.append(current)
                current = []
            continue
        parts = line.split()
        if len(parts) >= 2:
            current.append((float(parts[0]), float(parts[1])))
    if current:
        strokes.append(current)
    return strokes


def _sort_key(name: str) -> List[int]:
    return [int(n) for n in re.findall(r"\d+", name)]


class StrokesS3Store:
    def __init__(self, client: Any, bucket: str, prefix: str = "") -> None:
        self.client = client
        self.bucket = bucket
        self.prefix = normalize_s3_prefix(prefix)

    def _key(self, user_name: str, fname: str) -> str:
        return f"{self.prefix}{user_name}/{fname}"

    def _user_prefix(self, user_name: str) -> str:
        return f"{self.prefix}{user_name}/"

    def _head_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            status = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
            if status == 404 or code in ("404", "NotFound", "NoSuchKey"):
                return False
            raise

    def list_users(self) -> List[str]:
        """
        Mirror local server.py: list dataset/<user>/ as top-level names under S3_PREFIX.
        """
        users: set[str] = set()
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes") or []:
                p = cp["Prefix"]
                if not p.startswith(self.prefix):
                    continue
                inner = p[len(self.prefix) :].rstrip("/")
                if inner and "/" not in inner:
                    users.add(inner)
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if not key.startswith(self.prefix):
                    continue
                rel = key[len(self.prefix) :]
                if "/" not in rel:
                    continue
                user, rest = rel.split("/", 1)
                if user and (rest.endswith(".txt") or rest == "user_info.json"):
                    users.add(user)
        return sorted(users)

    def user_exists(self, user_name: str) -> bool:
        """True if user folder exists: user_info.json and/or any stroke .txt."""
        if self._head_exists(self._key(user_name, "user_info.json")):
            return True
        p = self._user_prefix(user_name)
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=p, MaxKeys=1)
        return bool(resp.get("Contents"))

    def create_user(self, user_name: str, info: Dict[str, Any]) -> None:
        key = self._key(user_name, "user_info.json")
        body = json.dumps(info, ensure_ascii=False)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json; charset=utf-8",
        )

    def get_line_save_counts(self, user_name: str) -> Dict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for f in self.list_txt_files(user_name):
            m = re.match(r"^(\d+)-\d+\.txt$", f)
            if m:
                line_1based = int(m.group(1))
                idx_0based = line_1based - 1
                counts[str(idx_0based)] += 1
        return dict(counts)

    def user_progress(self, user_name: str) -> int:
        written = set()
        for f in self.list_txt_files(user_name):
            m = re.match(r"^(\d+)-\d+\.txt$", f)
            if m:
                written.add(int(m.group(1)))
        return len(written)

    def list_txt_files(self, user_name: str) -> List[str]:
        files: List[str] = []
        p = self._user_prefix(user_name)
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=p):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if not key.startswith(p):
                    continue
                fname = key[len(p) :]
                if fname.endswith(".txt") and fname != "user_info.json":
                    files.append(fname)
        return sorted(files, key=_sort_key)

    def stroke_txt_exists(self, user_name: str, fname: str) -> bool:
        return self._head_exists(self._key(user_name, fname))

    def write_stroke_txt(self, user_name: str, fname: str, body: str) -> None:
        key = self._key(user_name, fname)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

    def read_stroke_txt(self, user_name: str, fname: str) -> str:
        key = self._key(user_name, fname)
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read().decode("utf-8")

    def delete_stroke_txt(self, user_name: str, fname: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(user_name, fname))

    def read_user_info(self, user_name: str) -> Dict[str, Any]:
        key = self._key(user_name, "user_info.json")
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError:
            return {}


def get_s3_client():
    try:
        import boto3
    except ImportError as e:
        print("Install deps: pip install -r requirements-s3.txt", file=sys.stderr)
        raise e

    endpoint = os.environ.get("S3_ENDPOINT_URL") or None
    region = os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    return boto3.client("s3", region_name=region, endpoint_url=endpoint)


def get_strokes_store() -> StrokesS3Store:
    bucket = os.environ.get("S3_BUCKET", "").strip()
    if not bucket:
        raise SystemExit("Set required env for stroke storage")
    prefix = normalize_s3_prefix(os.environ.get("S3_PREFIX", ""))
    return StrokesS3Store(get_s3_client(), bucket, prefix)
