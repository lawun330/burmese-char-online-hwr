"""
File Checker
- Help: python file_checker.py --help
- E.g.: python file_checker.py --dataset dataset
- E.g.: python file_checker.py --dataset dataset --user user_name
"""


#!/usr/bin/env python3
import os
from collections import defaultdict
import argparse

EXPECTED = {1, 2, 3, 4}


def check_user_dir(user_dir, label=None):
    """
    return a list of issue lines (missing/extra, or fatal messages).
    empty list means this user folder passes the filename rules.
    """
    tag = label or user_dir
    if not os.path.isdir(user_dir):
        return [f"{tag}: not a directory"]

    files = [f for f in os.listdir(user_dir) if f.endswith(".txt")]
    groups = defaultdict(set)
    for f in files:
        try:
            prefix, num = f.replace(".txt", "").split("-")
            groups[prefix].add(int(num))
        except ValueError:
            continue  # skip invalid filenames

    if not groups and not files:
        return [f"{tag}: no .txt stroke files"]

    issues = []
    for prefix, nums in sorted(groups.items(), key=lambda x: int(x[0])):
        missing = sorted(EXPECTED - nums)
        extra = sorted(nums - EXPECTED)
        if missing:
            issues.append(f"{tag}  line {prefix}: missing {missing}")
        if extra:
            issues.append(f"{tag}  line {prefix}: extra {extra}")
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Check stroke filenames per line (expect copies 1–4) under dataset/<user>/."
    )
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="dataset root directory (default: dataset)",
    )
    parser.add_argument(
        "--user",
        default=None,
        metavar="NAME",
        help="only check this user folder under the dataset root; omit to check every subfolder",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(root):
        print(f"dataset root not found: {root}")
        return

    if args.user:
        name = args.user.strip().replace(" ", "_")
        user_path = os.path.join(root, name)
        issues = check_user_dir(user_path, label=name)
        if issues:
            print("\n".join(issues))
        else:
            print("OK")
        return

    subs = [
        d
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]
    if not subs:
        print(f"No user folders under {root}")
        return

    all_issues = []
    for name in subs:
        all_issues.extend(check_user_dir(os.path.join(root, name), label=name))

    if all_issues:
        print("\n".join(all_issues))
    else:
        print("OK")


if __name__ == "__main__":
    main()
