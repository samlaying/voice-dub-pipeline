"""Command line entry point.

    python3 -m dub init --job jobs/demo --url https://...
    python3 -m dub run  --job jobs/demo --stages s1,s2
    python3 -m dub run  --job jobs/demo --stages s3,s4,s5,s6
    python3 -m dub status --job jobs/demo
"""

import argparse
import sys
from pathlib import Path

from . import stages
from .job import STAGES, STAGE_NAMES, Job


def cmd_init(args):
    root = Path(args.job).resolve()
    if (root / "manifest.json").exists() and not args.force:
        raise SystemExit(f"{root} already has a manifest (use --force to reset)")
    job = Job.create(root, url=args.url)
    job.data["source"]["language"] = args.language
    job.data["source"]["target_language"] = args.target
    job.save()
    print(f"created {root}")
    print(f"  source language : {args.language}")
    print(f"  target language : {args.target}")
    if args.url:
        print(f"  url             : {args.url}")
    return 0


def cmd_run(args):
    root = Path(args.job).resolve()
    if (root / "manifest.json").exists():
        job = Job.load(root)
    elif args.url:
        job = Job.create(root, url=args.url)
    else:
        raise SystemExit(f"{root} has no manifest and no --url was given")

    if args.url and not job.url:
        job.data["source"]["url"] = args.url
        job.save()

    selected = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {', '.join(unknown)} (expected {', '.join(STAGES)})")

    failed = None
    for stage in selected:
        if job.done(stage) and not args.force:
            print(f"[skip] {stage} {STAGE_NAMES[stage]} already done")
            continue
        print(f"\n== {stage} {STAGE_NAMES[stage]}")
        job.mark(stage, "running")
        try:
            info = stages.REGISTRY[stage](job) or {}
        except SystemExit as exc:
            job.mark(stage, "failed", error=str(exc))
            print(f"[fail] {stage}: {exc}")
            failed = stage
            break
        job.mark(stage, "done", **{key: value for key, value in info.items()})
        print(f"[ok] {stage}")

    if failed:
        return 1
    print("\nall requested stages finished")
    return 0


def cmd_status(args):
    job = Job.load(Path(args.job).resolve())
    print(f"job  {job.data.get('id')}")
    print(f"url  {job.url or '-'}")
    for stage in STAGES:
        record = job.data.get("stages", {}).get(stage, {})
        status = record.get("status", "pending")
        extra = record.get("error") or ""
        print(f"  {stage} {STAGE_NAMES[stage]:16} {status:8} {extra}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dub", description="Source video -> translated dubbed video")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create a job directory")
    init.add_argument("--job", required=True)
    init.add_argument("--url", default="")
    init.add_argument("--language", default="en", help="source language, e.g. en")
    init.add_argument("--target", default="zh", help="target language, e.g. zh")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    run = sub.add_parser("run", help="run stages")
    run.add_argument("--job", required=True)
    run.add_argument("--stages", default="all", help="comma separated, e.g. s1,s2 or all")
    run.add_argument("--url", default="")
    run.add_argument("--force", action="store_true", help="re-run stages already marked done")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="show stage progress")
    status.add_argument("--job", required=True)
    status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
