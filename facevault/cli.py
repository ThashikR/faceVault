"""Command line: python -m facevault <command> --help"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from . import attacks, legacy_rns, metrics, pipeline, sharing, vault
from .audit import AuditLog


def _cmd_setup(args):
    paths = pipeline.setup_keys(args.keys, args.threshold, args.officers)
    print(f"Vault created in {args.keys}")
    print(f"Any {args.threshold} of these {args.officers} share files can unlock faces:")
    for path in paths:
        print(f"  {path}")
    print("Give each file to a different person. The full private key was not saved anywhere.")


def _cmd_protect(args):
    from .detect import FaceMatcher, Watchlist, lock_detector

    keys = Path(args.keys)
    detector = lock_detector(args.detector)
    watchlist = None
    if args.watchlist:
        watchlist = Watchlist(detector, FaceMatcher())
        for name in watchlist.add_folder(args.watchlist):
            print(f"watch-list: no face found in {name}, skipped")

    rgb = vault.load_image(args.image)
    context = {"camera": args.camera, "source": Path(args.image).name}
    outcome = pipeline.protect_image(rgb, pipeline.read_key(keys / pipeline.VAULT_PUBLIC), detector, context, watchlist,
                                     signing_private=pipeline.read_key(keys / pipeline.CAMERA_KEY))
    vault.save_png(args.out, outcome.protected, outcome.header)
    pipeline.record_capture(keys, args.out, args.camera)
    print(f"{len(outcome.faces)} face(s) found, {len(outcome.header['boxes'])} region(s) locked -> {args.out}")
    print("image signed by the camera and added to the capture log")

    if outcome.matches:
        path = pipeline.send_alert(pipeline.make_alert(outcome, Path(args.out).name), keys, args.outbox)
        labels = ", ".join(match["label"] for match in outcome.matches)
        print(f"watch-list match ({labels}); encrypted alert written to {path}")


def _cmd_reveal(args):
    keys = Path(args.keys)
    protected, header = vault.load_png(args.image)
    shares = [sharing.Share.from_json(Path(path).read_text(encoding="utf-8")) for path in args.share]
    try:
        result = pipeline.reveal_image(protected, header, shares, args.reason, keys / pipeline.AUDIT_LOG, Path(args.image).name,
                                       trusted_camera=pipeline.read_key(keys / pipeline.CAMERA_PUBLIC))
    except (ValueError, vault.TamperedError) as error:
        print(f"DENIED: {error}")
        return 1
    Image.fromarray(result.image).save(args.out)
    print(f"faces unlocked -> {args.out}")
    if not result.background_intact:
        print("WARNING: the area outside the faces was edited after the image was protected.")
    return 0


def _cmd_read_alert(args):
    print(json.dumps(pipeline.read_alert(args.alert, args.keys), indent=2))


def _cmd_audit(args):
    log = AuditLog(Path(args.keys) / pipeline.AUDIT_LOG)
    for entry in log.entries():
        print(f"{entry['seq']:>3}  {entry['time']}  {entry['action']:<13}  {', '.join(entry['actors'])}  {entry['target']}  {entry['reason']}")
    ok, bad = log.verify()
    print("log chain: intact" if ok else f"log chain: BROKEN at entry {bad}")
    return 0 if ok else 1


def _cmd_captures(args):
    """List every image the camera locked and say whether each file is still there, unchanged."""
    log_ok, rows = pipeline.check_captures(args.keys, args.folder)
    for entry, status in rows:
        print(f"{entry['seq']:>3}  {entry['time']}  {', '.join(entry['actors'])}  {entry['target']:<30}  {status.upper() if status != 'ok' else 'ok'}")
    print("capture log: intact and signed by the camera" if log_ok else "capture log: EDITED, or holds an entry the camera did not sign")
    problems = [status for _, status in rows if status != "ok"]
    if problems:
        print(f"{problems.count('missing')} image(s) missing, {problems.count('altered')} altered")
    return 0 if log_ok and not problems else 1


def _cmd_legacy(args):
    """Run the 2024 scheme on an image and show what an outsider gets from its output."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rgb = vault.load_image(args.image)
    shares = legacy_rns.encode(rgb)
    for index, share in enumerate(shares):
        Image.fromarray(share).save(out / f"encrypted_{index}.png")
        Image.fromarray(attacks.stretch_share(share)).save(out / f"encrypted_{index}_brightened.png")
    recovered, moduli = attacks.recover_without_key(shares)
    Image.fromarray(recovered).save(out / "recovered_without_key.png")
    print(f"moduli read from the files: {moduli}")
    print(f"image rebuilt with no key: PSNR {metrics.psnr(rgb, recovered):.2f} dB, MSE {metrics.mse(rgb, recovered):.2f}")
    print(f"pixel values the scheme destroys (255): {attacks.lost_pixel_fraction(rgb) * 100:.2f}%")
    print(f"files written to {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="facevault", description="Lock faces in images; unlock them only with enough key shares.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="create a vault and split its key between officers")
    p.add_argument("--keys", default="keys")
    p.add_argument("--threshold", type=int, default=2, help="shares needed to unlock (default 2)")
    p.add_argument("--officers", type=int, default=3, help="shares created (default 3)")
    p.set_defaults(run=_cmd_setup)

    p = sub.add_parser("protect", help="lock every face in an image")
    p.add_argument("image")
    p.add_argument("-o", "--out", required=True, help="output PNG")
    p.add_argument("--keys", default="keys")
    p.add_argument("--camera", default="camera-01")
    p.add_argument("--watchlist", help="folder of face images; a match sends an encrypted alert")
    p.add_argument("--outbox", default="outbox")
    p.add_argument("--detector", choices=["yunet", "scrfd"], default="yunet",
                   help="scrfd misses fewer faces in crowds; research-only model, see scripts/fetch_research_models.py")
    p.set_defaults(run=_cmd_protect)

    p = sub.add_parser("reveal", help="unlock a protected image with officers' shares")
    p.add_argument("image")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--share", action="append", required=True, help="a share file; repeat for each officer")
    p.add_argument("--reason", required=True, help="why the faces are being unlocked (goes in the audit log)")
    p.add_argument("--keys", default="keys")
    p.set_defaults(run=_cmd_reveal)

    p = sub.add_parser("read-alert", help="verify and decrypt an alert file")
    p.add_argument("alert")
    p.add_argument("--keys", default="keys")
    p.set_defaults(run=_cmd_read_alert)

    p = sub.add_parser("audit", help="print the unlock log and check it was not altered")
    p.add_argument("--keys", default="keys")
    p.set_defaults(run=_cmd_audit)

    p = sub.add_parser("captures", help="check that every image the camera locked is still there and unchanged")
    p.add_argument("--keys", default="keys")
    p.add_argument("--folder", default=".", help="folder holding the protected images")
    p.set_defaults(run=_cmd_captures)

    p = sub.add_parser("legacy", help="show why the 2024 residue scheme is not encryption")
    p.add_argument("image")
    p.add_argument("-o", "--out", default="legacy_out")
    p.set_defaults(run=_cmd_legacy)

    args = parser.parse_args(argv)
    return args.run(args) or 0


if __name__ == "__main__":
    sys.exit(main())
