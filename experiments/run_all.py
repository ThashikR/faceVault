"""Measure the legacy residue scheme and FaceVault on real face images.

    python experiments/run_all.py              # 1000 LFW images, about ten minutes on a laptop CPU
    python experiments/run_all.py --images 100 # quick check

Needs the models (scripts/fetch_models.py) and LFW in data/ (see README).
Writes results/results.json, results/tables.md and results/figures/*.png.
Every number in the README comes from this script.
"""

import argparse
import itertools
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from facevault import attacks, detect, hybrid, legacy_rns, metrics, obfuscate, research_models, sharing, vault  # noqa: E402

LFW_DIR = ROOT / "data" / "lfw_home" / "lfw_funneled"
WIDER_DIR = ROOT / "data" / "wider_face"
RESULTS = ROOT / "results"

# Confidence at which an observer counts a detection as a face.
OBSERVER_THRESHOLD = 0.8
FIGURES = RESULTS / "figures"


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(statistics.fmean(values)) if values else None


def png_bytes(rgb):
    return len(cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1])


def median_ms(function, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(times))


def lfw_paths():
    if not LFW_DIR.exists():
        sys.exit(f"LFW not found at {LFW_DIR}. See the README for the one-line download.")
    return sorted(LFW_DIR.glob("*/*.jpg"))


# --- 1. the legacy residue scheme -------------------------------------------

def run_legacy(paths, observer):
    moduli = legacy_rns.LEGACY_MODULI
    n = len(paths)
    psnrs, mses, lost = [], [], []
    moduli_inferred = moduli_inferred_500 = damaged = 0
    original_detected = recovered_detected = 0
    raw_detected = [0] * len(moduli)
    bright_detected = [0] * len(moduli)
    share_entropy = [[] for _ in moduli]
    share_corr = [[] for _ in moduli]
    original_entropy, original_corr = [], []
    original_png = shares_png = 0

    for path in paths:
        rgb = vault.load_image(path)
        shares = legacy_rns.encode(rgb)
        restored = legacy_rns.decode(shares)

        error = metrics.mse(rgb, restored)
        mses.append(error)
        if error > 0:
            damaged += 1
            psnrs.append(metrics.psnr(rgb, restored))
        lost.append(attacks.lost_pixel_fraction(rgb))

        try:
            recovered, inferred = attacks.recover_without_key(shares)
        except ValueError:   # the moduli read off the files were wrong and not coprime
            recovered, inferred = None, None
        if inferred == moduli and np.array_equal(recovered, restored):
            moduli_inferred += 1

        # The same attack on files written as the 2024 app wrote them, first 500 pixels unaltered.
        as_app = legacy_rns.encode(rgb, unaltered=legacy_rns.UNALTERED_PIXELS)
        try:
            recovered_500, inferred_500 = attacks.recover_without_key(as_app)
        except ValueError:
            recovered_500, inferred_500 = None, None
        if inferred_500 == moduli and np.array_equal(recovered_500, legacy_rns.decode(as_app)):
            moduli_inferred_500 += 1

        original_detected += bool(observer.detect(rgb))
        recovered_detected += recovered is not None and bool(observer.detect(recovered))
        original_entropy.append(metrics.entropy(rgb))
        original_corr.append(metrics.adjacent_correlation(rgb[:, :, 0]))
        original_png += png_bytes(rgb)
        for i, share in enumerate(shares):
            raw_detected[i] += bool(observer.detect(share))
            bright_detected[i] += bool(observer.detect(attacks.stretch_share(share)))
            share_entropy[i].append(metrics.entropy(share))
            share_corr[i].append(metrics.adjacent_correlation(share[:, :, 0]))
            shares_png += png_bytes(share)

    return {
        "moduli": list(moduli),
        "images": n,
        "images_damaged_by_round_trip_pct": 100.0 * damaged / n,
        "round_trip_mse_mean": mean(mses),
        "round_trip_psnr_db_mean_over_damaged_images": mean(psnrs),
        "pixel_values_lost_pct_mean": 100.0 * mean(lost),
        "pixel_values_lost_pct_max": 100.0 * max(lost),
        "keyless_recovery_success_pct": 100.0 * moduli_inferred / n,
        "keyless_recovery_with_500_unaltered_pixels_pct": 100.0 * moduli_inferred_500 / n,
        "face_detected_pct": {
            "original": 100.0 * original_detected / n,
            "recovered_without_key": 100.0 * recovered_detected / n,
            **{f"share_mod_{m}_as_stored": 100.0 * raw_detected[i] / n for i, m in enumerate(moduli)},
            **{f"share_mod_{m}_brightened": 100.0 * bright_detected[i] / n for i, m in enumerate(moduli)},
        },
        "entropy_bits": {"original": mean(original_entropy), **{f"share_mod_{m}": mean(share_entropy[i]) for i, m in enumerate(moduli)}},
        "adjacent_correlation": {"original": mean(original_corr), **{f"share_mod_{m}": mean(share_corr[i]) for i, m in enumerate(moduli)}},
        "storage_ratio_png_shares_vs_original": shares_png / original_png,
    }


# --- 2. FaceVault ------------------------------------------------------------

def run_facevault(paths, detector, observer, matcher, rng):
    private, public = hybrid.generate_keypair()
    wrong_private, _ = hybrid.generate_keypair()
    n = len(paths)
    no_face = exact = background_changed = protected_detected = 0
    bit_flip_caught = context_caught = wrong_key_caught = 0
    protect_ms, reveal_ms, detect_ms = [], [], []
    roi_entropy, roi_corr, npcrs, uacis, area, similarity = [], [], [], [], [], []
    pooled = np.zeros(256, dtype=np.int64)
    original_png = protected_png = header_bytes = 0
    used = 0

    for path in paths:
        rgb = vault.load_image(path)
        start = time.perf_counter()
        faces = detector.detect(rgb)
        detect_ms.append((time.perf_counter() - start) * 1000.0)
        if not faces:
            no_face += 1
            continue
        used += 1
        boxes = [face.box for face in faces]
        context = {"camera": "lfw", "source": path.name}

        start = time.perf_counter()
        protected, header = vault.protect(rgb, boxes, public, context)
        protect_ms.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        result = vault.reveal(protected, header, private)
        reveal_ms.append((time.perf_counter() - start) * 1000.0)

        exact += np.array_equal(result.image, rgb) and result.background_intact
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for x, y, w, h in header["boxes"]:
            mask[y:y + h, x:x + w] = True
        background_changed += int(np.any(protected[~mask] != rgb[~mask]))
        area.append(float(mask.mean()))
        protected_detected += bool(observer.detect(protected))

        # The same image with one bit changed, protected again: how different is the ciphertext?
        x, y, w, h = header["boxes"][0]
        nudged = rgb.copy()
        nudged[y, x, 0] ^= 1
        second, _ = vault.protect(nudged, boxes, public, context)
        first_roi, second_roi = protected[y:y + h, x:x + w], second[y:y + h, x:x + w]
        npcrs.append(metrics.npcr(first_roi, second_roi))
        uacis.append(metrics.uaci(first_roi, second_roi))
        roi_entropy.append(metrics.entropy(first_roi))
        roi_corr.append(metrics.adjacent_correlation(first_roi[:, :, 0]))
        pooled += np.bincount(protected[mask].ravel(), minlength=256)

        # Tampering and wrong-key attempts must all be refused.
        flipped = protected.copy()
        flipped[y + int(rng.integers(h)), x + int(rng.integers(w)), int(rng.integers(3))] ^= 1 << int(rng.integers(8))
        for attempt, counter in ((lambda: vault.reveal(flipped, header, private), "bit"),
                                 (lambda: vault.reveal(protected, {**header, "context": {**context, "camera": "other"}}, private), "ctx"),
                                 (lambda: vault.reveal(protected, header, wrong_private), "key")):
            try:
                attempt()
            except vault.TamperedError:
                if counter == "bit":
                    bit_flip_caught += 1
                elif counter == "ctx":
                    context_caught += 1
                else:
                    wrong_key_caught += 1

        # Strongest case for an attacker: they are told exactly where the face is.
        face = faces[0]
        similarity.append(matcher.similarity(matcher.embed(rgb, face), matcher.embed(protected, face)))

        original_png += png_bytes(rgb)
        protected_png += png_bytes(protected)
        header_bytes += len(json.dumps(header))

    probs = pooled[pooled > 0] / pooled.sum()
    return {
        "images": n,
        "lock_threshold": detect.LOCK_THRESHOLD,
        "images_with_a_face": used,
        "images_with_no_face_found": no_face,
        "bit_exact_reveal_pct": 100.0 * exact / used,
        "images_with_any_change_outside_faces": background_changed,
        "face_detected_in_protected_pct": 100.0 * protected_detected / used,
        "face_area_pct_of_image_mean": 100.0 * mean(area),
        "ciphertext_entropy_bits_per_region_mean": mean(roi_entropy),
        "ciphertext_entropy_bits_pooled": float(-(probs * np.log2(probs)).sum()),
        "ciphertext_adjacent_correlation_mean": mean(roi_corr),
        "ciphertext_adjacent_correlation_mean_abs": mean([abs(c) for c in roi_corr]),
        "npcr_pct_mean": mean(npcrs),
        "uaci_pct_mean": mean(uacis),
        "rejected_pct": {
            "one_bit_flipped_in_face_region": 100.0 * bit_flip_caught / used,
            "context_changed_in_header": 100.0 * context_caught / used,
            "wrong_private_key": 100.0 * wrong_key_caught / used,
        },
        "sface_similarity_original_vs_protected": {
            "mean": mean(similarity),
            "max": float(max(similarity)),
            "pct_at_or_above_match_threshold": 100.0 * float(np.mean(np.array(similarity) >= detect.MATCH_THRESHOLD)),
            "match_threshold": detect.MATCH_THRESHOLD,
        },
        "time_ms_median": {
            "detect": float(statistics.median(detect_ms)),
            "protect": float(statistics.median(protect_ms)),
            "reveal": float(statistics.median(reveal_ms)),
        },
        "storage_ratio_png_protected_vs_original": protected_png / original_png,
        "header_bytes_mean": header_bytes / used,
    }


# --- 2b. how low should the lock threshold be? ------------------------------

def run_lock_threshold(paths, observer):
    """Lock at several detector confidences; count images where the observer still finds a face."""
    _, public = hybrid.generate_keypair()
    rows = []
    for threshold in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        locker = detect.FaceDetector(threshold)
        leftover, inside, area, regions = 0, 0, [], []
        for path in paths:
            rgb = vault.load_image(path)
            protected, header = vault.protect(rgb, [face.box for face in locker.detect(rgb)], public)
            found = observer.detect(protected)
            leftover += bool(found)
            # Is any detection centred on a locked region, i.e. a face seen through the noise?
            inside += any(bx <= x + w / 2 <= bx + bw and by <= y + h / 2 <= by + bh
                          for x, y, w, h in (face.box for face in found) for bx, by, bw, bh in header["boxes"])
            regions.append(len(header["boxes"]))
            area.append(sum(w * h for _, _, w, h in header["boxes"]) / (rgb.shape[0] * rgb.shape[1]))
        rows.append({
            "lock_threshold": threshold,
            "images_with_face_still_found": leftover,
            "images_with_face_still_found_pct": 100.0 * leftover / len(paths),
            "images_with_detection_inside_a_locked_region": inside,
            "regions_locked_per_image_mean": mean(regions),
            "area_locked_pct_mean": 100.0 * mean(area),
        })
    return rows


# --- 2c. blur, pixelation and a black box against FaceVault -----------------

def obfuscation_methods(public):
    """(label, function(image, boxes) -> image, can the original be recovered?)"""
    return [
        ("No protection", lambda image, boxes: image, None),
        ("Gaussian blur, 15 px", lambda image, boxes: obfuscate.blur(image, boxes, 15), False),
        ("Gaussian blur, 45 px", lambda image, boxes: obfuscate.blur(image, boxes, 45), False),
        ("Pixelation, 8 px blocks", lambda image, boxes: obfuscate.pixelate(image, boxes, 8), False),
        ("Pixelation, 16 px blocks", lambda image, boxes: obfuscate.pixelate(image, boxes, 16), False),
        ("Black box", obfuscate.black_box, False),
        ("FaceVault", lambda image, boxes: vault.protect(image, boxes, public, margin=0)[0], True),
    ]


def photo_pairs(all_paths, people, seed):
    """(photo A, photo B) of the same person, for `people` randomly chosen people."""
    by_person = {}
    for path in all_paths:
        by_person.setdefault(path.parent.name, []).append(path)
    names = sorted(name for name, items in by_person.items() if len(items) >= 2)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(names), size=min(people, len(names)), replace=False))
    return [(by_person[names[i]][0], by_person[names[i]][1]) for i in chosen]


FALSE_ACCEPT_RATE = 0.001   # a second threshold per recogniser: 1 pair of different people in 1,000 accepted


def run_obfuscation(paths, pairs, detector, observer, matcher, second=None):
    """Hide the same face regions with each method; ask a detector and recognisers what is left."""
    _, public = hybrid.generate_keypair()
    methods = obfuscation_methods(public)
    recognisers = {"sface": matcher}
    if second is not None:
        recognisers["arcface"] = second

    def hidden_versions(rgb, faces):
        boxes = vault.prepare_boxes([face.box for face in faces], rgb.shape)
        return boxes, [hide(rgb, boxes) for _, hide, _ in methods]

    # Part 1: the attacker holds the very same photo, unprotected.
    detected = [0] * len(methods)
    same_photo = [[] for _ in methods]
    used = 0
    for path in paths:
        rgb = vault.load_image(path)
        faces = detector.detect(rgb)
        if not faces:
            continue
        used += 1
        original = matcher.embed(rgb, faces[0])
        boxes, versions = hidden_versions(rgb, faces)
        for index, hidden in enumerate(versions):
            detected[index] += any(bx <= x + w / 2 <= bx + bw and by <= y + h / 2 <= by + bh
                                   for x, y, w, h in (found.box for found in observer.detect(hidden))
                                   for bx, by, bw, bh in boxes)
            # The recogniser is told exactly where the face is: the strongest case for an attacker.
            same_photo[index].append(matcher.similarity(original, matcher.embed(hidden, faces[0])))

    # Part 2: the attacker holds a different photo of the same person, as a watch-list would.
    # Matching the hidden face against a WRONG person shows what chance alone produces.
    enrolled = {name: [] for name in recognisers}
    probes = []
    for enrolled_path, probe_path in pairs:
        enrolled_rgb, probe_rgb = vault.load_image(enrolled_path), vault.load_image(probe_path)
        enrolled_faces, probe_faces = detector.detect(enrolled_rgb), detector.detect(probe_rgb)
        if enrolled_faces and probe_faces:
            for name, model in recognisers.items():
                enrolled[name].append(model.embed(enrolled_rgb, enrolled_faces[0]))
            probes.append((probe_rgb, probe_faces))
    people = len(probes)

    # Thresholds: the one recommended with SFace, and for each recogniser the
    # value that accepts 1 in 1,000 pairs of different enrolled people.
    thresholds = {"sface_recommended": detect.MATCH_THRESHOLD}
    for name, model in recognisers.items():
        impostors = [model.similarity(a, b) for a, b in itertools.combinations(enrolled[name], 2)]
        thresholds[f"{name}_far_0.1pct"] = float(np.quantile(impostors, 1.0 - FALSE_ACCEPT_RATE))

    other_photo = {name: [[] for _ in methods] for name in recognisers}
    wrong_person = {name: [[] for _ in methods] for name in recognisers}
    for person, (probe_rgb, probe_faces) in enumerate(probes):
        _, versions = hidden_versions(probe_rgb, probe_faces)
        for index, hidden in enumerate(versions):
            for name, model in recognisers.items():
                feature = model.embed(hidden, probe_faces[0])
                other_photo[name][index].append(model.similarity(enrolled[name][person], feature))
                wrong_person[name][index].append(model.similarity(enrolled[name][(person + 1) % people], feature))

    def matched(scores, threshold=detect.MATCH_THRESHOLD):
        return 100.0 * float(np.mean(np.array(scores) >= threshold))

    def by_setting(scores, index):
        result = {"sface_recommended": matched(scores["sface"][index])}
        for name in recognisers:
            result[f"{name}_far_0.1pct"] = matched(scores[name][index], thresholds[f"{name}_far_0.1pct"])
        return result

    return [{
        "method": label,
        "images": used,
        "people": people,
        "face_detected_in_hidden_region_pct": 100.0 * detected[index] / used,
        "matched_to_same_photo_pct": matched(same_photo[index]),
        "matched_to_other_photo_pct": matched(other_photo["sface"][index]),
        "matched_to_wrong_person_pct": matched(wrong_person["sface"][index]),
        "similarity_to_same_photo_mean": mean(same_photo[index]),
        "similarity_to_other_photo_mean": mean(other_photo["sface"][index]),
        "similarity_to_wrong_person_mean": mean(wrong_person["sface"][index]),
        "recognised_pct": by_setting(other_photo, index),
        "wrong_person_pct": by_setting(wrong_person, index),
        "similarity_mean": {name: {"right_person": mean(other_photo[name][index]), "wrong_person": mean(wrong_person[name][index])}
                            for name in recognisers},
        "thresholds": thresholds,
        "original_recoverable": recoverable,
    } for index, (label, _, recoverable) in enumerate(methods)]


# --- 2d. crowded scenes: how many faces does the lock step miss? -------------

CROWD_THRESHOLDS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
SIZE_BUCKETS = (("under 16 px", 0, 16), ("16 to 31 px", 16, 32), ("32 to 63 px", 32, 64),
                ("64 to 127 px", 64, 128), ("128 px and over", 128, 10 ** 9))
LOCKED, VISIBLE = 0.9, 0.1   # share of a face's annotated box that must be / must not be encrypted


def wider_annotations():
    """Yield (image path, [(x, y, w, h), ...]) for the WIDER FACE validation set, leaving out faces marked invalid."""
    listing = WIDER_DIR / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    if not listing.exists():
        sys.exit(f"WIDER FACE not found at {WIDER_DIR}. See the README for the download.")
    lines = listing.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        name, count = lines[i].strip(), int(lines[i + 1])
        faces = []
        for row in lines[i + 2:i + 2 + count]:
            x, y, w, h, _blur, _expression, _illumination, invalid = (int(v) for v in row.split()[:8])
            if not invalid and w > 0 and h > 0:
                faces.append((x, y, w, h))
        yield WIDER_DIR / "WIDER_val" / "images" / name, faces
        i += 2 + max(count, 1)   # an image with no faces still has one placeholder row


def crowd_variants():
    """name -> (label, function returning every face found at the lowest lock confidence)."""
    lowest = min(CROWD_THRESHOLDS)
    yunet = detect.FaceDetector(lowest)
    variants = {
        "one_scale": ("YuNet, one pass", yunet.detect),
        "two_scale": ("YuNet, two passes", lambda rgb: yunet.detect(rgb) + yunet.detect_enlarged(rgb, 2)),
    }
    if research_models.SCRFD.exists():
        variants["scrfd_640"] = ("SCRFD-10G", research_models.ScrfdDetector(lowest, long_side=640).detect)
    return variants


def run_crowd(limit=None):
    """Lock WIDER FACE images and count the annotated faces that end up encrypted.

    Each detector runs once at the lowest confidence; the higher confidences
    are the same detections filtered by score. 'Two passes' adds a pass on a
    copy enlarged 2x. SCRFD-10G is a stronger, research-only detector.
    """
    finders = crowd_variants()
    variants = tuple(finders)
    counts = {v: {t: {name: [0, 0, 0] for name, _, _ in SIZE_BUCKETS} for t in CROWD_THRESHOLDS} for v in variants}
    area = {v: {t: [] for t in CROWD_THRESHOLDS} for v in variants}
    regions = {v: {t: [] for t in CROWD_THRESHOLDS} for v in variants}
    times = {v: [] for v in variants}
    images = faces_total = 0

    for path, truth in wider_annotations():
        if limit and images >= limit:
            break
        rgb = vault.load_image(path)
        height, width = rgb.shape[:2]
        images += 1
        faces_total += len(truth)

        for variant, (_, find) in finders.items():
            start = time.perf_counter()
            found = find(rgb)
            times[variant].append((time.perf_counter() - start) * 1000.0)
            for threshold in CROWD_THRESHOLDS:
                boxes = vault.prepare_boxes([face.box for face in found if face.score >= threshold], rgb.shape)
                mask = np.zeros((height, width), dtype=bool)
                for x, y, w, h in boxes:
                    mask[y:y + h, x:x + w] = True
                area[variant][threshold].append(float(mask.mean()))
                regions[variant][threshold].append(len(boxes))
                for x, y, w, h in truth:
                    patch = mask[max(y, 0):y + h, max(x, 0):x + w]
                    covered = float(patch.mean()) if patch.size else 0.0
                    bucket = next(name for name, low, high in SIZE_BUCKETS if low <= h < high)
                    tally = counts[variant][threshold][bucket]
                    tally[0] += 1
                    tally[1] += covered >= LOCKED
                    tally[2] += covered < VISIBLE

    def share(tallies):
        faces = sum(t[0] for t in tallies)
        return {"faces": faces,
                "locked_pct": 100.0 * sum(t[1] for t in tallies) / max(faces, 1),
                "visible_pct": 100.0 * sum(t[2] for t in tallies) / max(faces, 1)}

    def summary(variant, threshold, minimum_height=0):
        return share([counts[variant][threshold][name] for name, low, _ in SIZE_BUCKETS if low >= minimum_height])

    default = detect.LOCK_THRESHOLD
    target_area = mean(area[variants[0]][default])
    matched = {v: min(CROWD_THRESHOLDS, key=lambda t: abs(mean(area[v][t]) - target_area)) for v in variants}
    matched[variants[0]] = default
    return {
        "dataset": "WIDER FACE validation set",
        "images": images,
        "faces": faces_total,
        "locked_means_covered_at_least": LOCKED,
        "visible_means_covered_less_than": VISIBLE,
        "variants": {v: finders[v][0] for v in variants},
        "detect_ms_median": {v: float(statistics.median(times[v])) for v in variants},
        "by_size_at_lock_threshold": {
            "lock_threshold": default,
            "rows": [{"face_height": name, **{v: share([counts[v][default][name]]) for v in variants}} for name, _, _ in SIZE_BUCKETS],
        },
        "matched_area": {
            "explanation": "each detector at the confidence whose locked image area is closest to the first detector's at the default lock confidence",
            "lock_threshold": matched,
            "area_locked_pct": {v: 100.0 * mean(area[v][matched[v]]) for v in variants},
            "faces_32px_and_over": {v: summary(v, matched[v], 32) for v in variants},
            "rows": [{"face_height": name, **{v: share([counts[v][matched[v]][name]]) for v in variants}} for name, _, _ in SIZE_BUCKETS],
        },
        "by_lock_threshold": [{
            "lock_threshold": t,
            **{v: {"all_faces": summary(v, t),
                   "faces_32px_and_over": summary(v, t, 32),
                   "area_locked_pct": 100.0 * mean(area[v][t]),
                   "regions_per_image": mean(regions[v][t])} for v in variants},
        } for t in CROWD_THRESHOLDS],
    }


# --- 3. watch-list matching --------------------------------------------------

def run_watchlist(all_paths, detector, matcher, people, rng):
    by_person = {}
    for path in all_paths:
        by_person.setdefault(path.parent.name, []).append(path)
    names = sorted(name for name, items in by_person.items() if len(items) >= 2)
    names = [names[i] for i in sorted(rng.choice(len(names), size=min(people, len(names)), replace=False))]

    def feature(path):
        rgb = vault.load_image(path)
        faces = detector.detect(rgb)
        return matcher.embed(rgb, faces[0]) if faces else None

    enrolled = {name: feature(by_person[name][0]) for name in names}
    probes = {name: feature(by_person[name][1]) for name in names}
    usable = [name for name in names if enrolled[name] is not None and probes[name] is not None]

    genuine = [matcher.similarity(enrolled[name], probes[name]) for name in usable]
    # Each probe against the next person's enrolment: same count of impostor trials.
    impostor = [matcher.similarity(enrolled[usable[(i + 1) % len(usable)]], probes[name]) for i, name in enumerate(usable)]
    threshold = detect.MATCH_THRESHOLD
    return {
        "people": len(usable),
        "threshold": threshold,
        "true_accept_pct": 100.0 * float(np.mean(np.array(genuine) >= threshold)),
        "false_accept_pct": 100.0 * float(np.mean(np.array(impostor) >= threshold)),
        "genuine_similarity_mean": mean(genuine),
        "impostor_similarity_mean": mean(impostor),
    }


# --- 4. speed at camera resolutions -----------------------------------------

def run_speed(detector, repeats=15):
    from skimage import data

    base = np.ascontiguousarray(data.astronaut())
    _, public = hybrid.generate_keypair()
    key = AESGCM.generate_key(256)
    rows = []
    for width, height in ((640, 480), (1280, 720), (1920, 1080)):
        frame = cv2.resize(base, (width, height), interpolation=cv2.INTER_AREA)
        faces = detector.detect(frame)
        boxes = [face.box for face in faces]
        _, header = vault.protect(frame, boxes, public)
        roi_pixels = sum(w * h for _, _, w, h in header["boxes"])
        shares = legacy_rns.encode(frame)
        rows.append({
            "resolution": f"{width}x{height}",
            "faces": len(faces),
            "face_area_pct": 100.0 * roi_pixels / (width * height),
            "detect_ms": median_ms(lambda: detector.detect(frame), repeats),
            "facevault_protect_ms": median_ms(lambda: vault.protect(frame, boxes, public), repeats),
            "of_which_background_hash_ms": median_ms(lambda: vault._background_hash(frame, header["boxes"]), repeats),
            "whole_frame_aes_gcm_ms": median_ms(lambda: AESGCM(key).encrypt(b"\0" * 12, frame.tobytes(), None), repeats),
            "legacy_encode_ms": median_ms(lambda: legacy_rns.encode(frame), repeats),
            "legacy_decode_ms": median_ms(lambda: legacy_rns.decode(shares), repeats),
        })
    return rows


# --- 5. threshold key sharing ------------------------------------------------

def run_sharing(trials=200):
    rows = []
    for k, n in ((2, 3), (3, 5), (5, 8)):
        secret = hybrid.generate_keypair()[0]
        shares = sharing.split(secret, k, n)
        subsets = list(itertools.combinations(shares, k))
        correct = sum(sharing.combine(list(subset)) == secret for subset in subsets)
        leaked = 0
        for _ in range(trials):
            trial_secret = hybrid.generate_keypair()[0]
            trial_shares = sharing.split(trial_secret, k, n)
            leaked += sharing.reconstruct_unchecked(trial_shares[:k - 1]) == trial_secret
        sharing._parameters.cache_clear()
        start = time.perf_counter()
        sharing._parameters(len(secret), k, n)
        setup_ms = (time.perf_counter() - start) * 1000.0
        rows.append({
            "k": k, "n": n,
            "one_time_setup_ms": setup_ms,
            "k_subsets_tested": len(subsets),
            "k_subsets_correct": correct,
            "k_minus_1_trials": trials,
            "k_minus_1_recovered_key": leaked,
            "split_ms": median_ms(lambda: sharing.split(secret, k, n), 20),
            "combine_ms": median_ms(lambda: sharing.combine(shares[:k]), 20),
        })
    return rows


# --- figures -----------------------------------------------------------------

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def make_figures(results, detector):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import data

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Demo image: NASA portrait of Eileen Collins (public domain), bundled with scikit-image.
    rgb = np.ascontiguousarray(data.astronaut())
    private, public = hybrid.generate_keypair()
    shares = legacy_rns.encode(rgb)
    protected, header = vault.protect(rgb, [face.box for face in detector.detect(rgb)], public)
    revealed = vault.reveal(protected, header, private).image
    panels = [
        ("Original", rgb),
        ("2024 scheme: file as stored (mod 17)", shares[2]),
        ("Same file, brightened", attacks.stretch_share(shares[2])),
        ("2024 scheme: rebuilt with no key", attacks.recover_without_key(shares)[0]),
        ("FaceVault: protected", protected),
        ("FaceVault: unlocked by 2 of 3 officers", revealed),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    for axis, (title, picture) in zip(axes.ravel(), panels):
        axis.imshow(picture)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(FIGURES / "comparison.png", dpi=150)
    plt.close(figure)

    # The same face hidden six ways.
    face_box = vault.prepare_boxes([face.box for face in detector.detect(rgb)], rgb.shape)
    x, y, w, h = face_box[0]
    pad = 40
    window = (slice(max(y - pad, 0), y + h + pad), slice(max(x - pad, 0), x + w + pad))
    rows = {row["method"]: row for row in results["obfuscation"]}
    panels = [(label, hide(rgb, face_box), rows[label]) for label, hide, _ in obfuscation_methods(public)]
    figure, axes = plt.subplots(1, len(panels), figsize=(13.5, 3.1))
    for axis, (label, picture, row) in zip(axes, panels):
        axis.imshow(picture[window])
        axis.set_title(label, fontsize=9.5)
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_visible(False)
        if row is not None:
            caption = [f"recognised: {row['matched_to_other_photo_pct']:.1f}%"]
            if row["original_recoverable"] is not None:
                caption.append(f"recoverable: {'yes' if row['original_recoverable'] else 'no'}")
            axis.set_xlabel(chr(10).join(caption), fontsize=9, color=INK)
    figure.tight_layout()
    figure.savefig(FIGURES / "obfuscation.png", dpi=150)
    plt.close(figure)

    # Crowded scenes: how many faces are locked against how much of the image is given up.
    if results.get("crowd"):
        crowd = results["crowd"]
        labels = crowd["variants"]
        colours = dict(zip(labels, (BLUE, ORANGE, AQUA)))
        matched = crowd["matched_area"]

        figure, axis = plt.subplots(figsize=(8.2, 4.2))
        offsets = dict(zip(labels, ((12, -5), (-12, -20), (-12, 8))))
        for key, label in labels.items():
            points = sorted((row[key]["area_locked_pct"], row[key]["faces_32px_and_over"]["locked_pct"], row["lock_threshold"])
                            for row in crowd["by_lock_threshold"])
            axis.plot([x for x, _, _ in points], [y for _, y, _ in points], color=colours[key], linewidth=2,
                      marker="o", markersize=4.5, label=label)
            x, y = matched["area_locked_pct"][key], matched["faces_32px_and_over"][key]["locked_pct"]
            axis.plot([x], [y], marker="o", markersize=10, markerfacecolor="none", markeredgecolor=colours[key], markeredgewidth=2)
            axis.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=offsets[key], color=INK, fontsize=9.5,
                          ha="left" if offsets[key][0] > 0 else "right")
        axis.set_xlabel("Share of the image locked (%)")
        axis.set_ylabel("Faces 32 px and over locked (%)")
        axis.set_xlim(0, 30)
        axis.set_ylim(40, 100)
        axis.grid(True, color="#e6e5e1", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, loc="lower right")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        figure.tight_layout()
        figure.savefig(FIGURES / "crowd_tradeoff.png", dpi=150)
        plt.close(figure)

        rows = matched["rows"]
        width = 0.8 / len(labels)
        figure, axis = plt.subplots(figsize=(9.2, 3.9))
        positions = np.arange(len(rows))
        for index, (key, label) in enumerate(labels.items()):
            offset = (index - (len(labels) - 1) / 2) * width
            values = [row[key]["locked_pct"] for row in rows]
            axis.bar(positions + offset, values, width=width * 0.92, color=colours[key], label=label)
            for position, value in zip(positions + offset, values):
                axis.text(position, value + 1.5, f"{value:.0f}%", ha="center", color=INK, fontsize=8.5)
        first = next(iter(labels))
        axis.set_xticks(positions, [row["face_height"] + chr(10) + f"({row[first]['faces']} faces)" for row in rows])
        axis.tick_params(axis="x", length=0, labelcolor=INK)
        axis.set_ylim(0, 120)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_ylabel("Annotated faces locked (%)")
        axis.set_xlabel("Face height in the image")
        axis.yaxis.grid(True, color="#e6e5e1", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, loc="upper left", ncols=len(labels))
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        figure.tight_layout()
        figure.savefig(FIGURES / "crowd.png", dpi=150)
        plt.close(figure)

    # Was a face still found? One bar per kind of image.
    found = results["legacy"]["face_detected_pct"]
    bars = [
        ("Original image", found["original"], MUTED),
        ("2024 scheme: mod 17 file, brightened", found["share_mod_17_brightened"], ORANGE),
        ("2024 scheme: mod 5 file, brightened", found["share_mod_5_brightened"], ORANGE),
        ("2024 scheme: mod 3 file, brightened", found["share_mod_3_brightened"], ORANGE),
        ("2024 scheme: rebuilt with no key", found["recovered_without_key"], ORANGE),
        ("FaceVault: protected image", results["facevault"]["face_detected_in_protected_pct"], BLUE),
    ]
    figure, axis = plt.subplots(figsize=(8.2, 3.6))
    positions = range(len(bars))[::-1]
    axis.barh(list(positions), [value for _, value, _ in bars], color=[colour for _, _, colour in bars], height=0.62)
    for position, (_, value, _) in zip(positions, bars):
        axis.text(value + 1.2, position, f"{value:.1f}%", va="center", color=INK, fontsize=9.5)
    axis.set_yticks(list(positions), [label for label, _, _ in bars])
    axis.tick_params(axis="y", length=0, labelcolor=INK)
    axis.set_xlim(0, 112)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel(f"Images in which the face detector still finds a face (n = {results['legacy']['images']})")
    axis.xaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(FIGURES / "face_detection.png", dpi=150)
    plt.close(figure)

    # Histograms of the face region: photo vs residue file vs ciphertext.
    x, y, w, h = header["boxes"][0]
    series = [
        ("Original face region", rgb[y:y + h, x:x + w], MUTED),
        ("2024 scheme (mod 17)", shares[2][y:y + h, x:x + w], ORANGE),
        ("FaceVault ciphertext", protected[y:y + h, x:x + w], BLUE),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=False)
    for axis, (title, region, colour) in zip(axes, series):
        axis.bar(range(256), np.bincount(region.ravel(), minlength=256), width=1.0, color=colour)
        axis.set_title(f"{title}\nentropy {metrics.entropy(region):.2f} bits", fontsize=10)
        axis.set_xlim(0, 255)
        axis.set_xticks([0, 128, 255])
        axis.set_yticks([])
        axis.set_xlabel("Byte value")
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(FIGURES / "histograms.png", dpi=150)
    plt.close(figure)


# --- tables ------------------------------------------------------------------

def recogniser_table(rows):
    """The same comparison with every recogniser at one common operating point."""
    if "recognised_pct" not in rows[0]:
        return []
    settings = [key for key in rows[0]["recognised_pct"] if key != "sface_recommended"]
    names = {"sface_far_0.1pct": "SFace", "arcface_far_0.1pct": "ArcFace R50"}
    thresholds = rows[0]["thresholds"]
    return [
        "Recognised against another photo of the same person, with each recogniser's threshold set so that",
        "1 pair of different people in 1,000 is accepted ("
        + ", ".join(f"{names[key]}: {thresholds[key]:.3f}" for key in settings) + "):",
        "",
        "| Method | " + " | ".join(f"{names[key]}, recognised" for key in settings) + " | "
        + " | ".join(f"{names[key]}, wrong person" for key in settings) + " |",
        "|---|" + "---|" * (2 * len(settings)),
        *["| " + row["method"] + " | " + " | ".join(f"{row['recognised_pct'][key]:.1f}%" for key in settings) + " | "
          + " | ".join(f"{row['wrong_person_pct'][key]:.1f}%" for key in settings) + " |" for row in rows],
        "",
    ]


def crowd_tables(results):
    crowd = results.get("crowd")
    if not crowd:
        return []
    labels = crowd["variants"]
    names = list(labels)
    first = names[0]
    matched = crowd["matched_area"]
    by_size = crowd["by_size_at_lock_threshold"]
    return [
        "## Table 2d. Crowded scenes: faces the lock step reaches",
        "",
        f"{crowd['dataset']}: {crowd['images']} images, {crowd['faces']} annotated faces (faces marked invalid left out).",
        f"A face counts as locked when at least {crowd['locked_means_covered_at_least']:.0%} of its annotated box is encrypted.",
        "'Two passes' adds a detection pass on the image enlarged 2x. SCRFD-10G is a stronger, research-only detector,",
        "run with the image's longer side scaled to 640 px.",
        "",
        "Detectors score on different scales, so they are compared at equal cost: each at the confidence that locks",
        f"about the same share of the image as {labels[first]} does at confidence {by_size['lock_threshold']}.",
        "",
        "| Detector | Lock confidence | Image area locked | Faces 32 px and over locked | Left fully visible | Median detection time |",
        "|---|---|---|---|---|---|",
        *[f"| {labels[n]} | {matched['lock_threshold'][n]} | {matched['area_locked_pct'][n]:.1f}% | "
          f"{matched['faces_32px_and_over'][n]['locked_pct']:.1f}% | {matched['faces_32px_and_over'][n]['visible_pct']:.1f}% | "
          f"{crowd['detect_ms_median'][n]:.0f} ms |" for n in names],
        "",
        "Share of faces locked by face size, at those confidences:",
        "",
        "| Face height | Faces | " + " | ".join(labels[n] for n in names) + " |",
        "|---|---|" + "---|" * len(names),
        *[f"| {row['face_height']} | {row[first]['faces']} | " + " | ".join(f"{row[n]['locked_pct']:.1f}%" for n in names) + " |"
          for row in matched["rows"]],
        "",
        "Every confidence tried, faces 32 px and over: share locked (share of image area locked).",
        "",
        "| Lock confidence | " + " | ".join(labels[n] for n in names) + " |",
        "|---|" + "---|" * len(names),
        *[f"| {row['lock_threshold']} | " + " | ".join(
            f"{row[n]['faces_32px_and_over']['locked_pct']:.1f}% ({row[n]['area_locked_pct']:.1f}%)" for n in names) + " |"
          for row in crowd["by_lock_threshold"]],
        "",
    ]


def write_tables(results):
    legacy, fv, wl = results["legacy"], results["facevault"], results["watchlist"]
    found = legacy["face_detected_pct"]
    lines = [
        "# Measured results",
        "",
        f"Generated by `experiments/run_all.py` on {results['meta']['date']} "
        f"({results['meta']['images']} LFW images, seed {results['meta']['seed']}, {results['meta']['machine']}).",
        "",
        "## Table 1. The 2024 residue scheme (moduli 3, 5, 17)",
        "",
        "| Measurement | Value |",
        "|---|---|",
        f"| Images rebuilt with no key, from the stored files alone | {legacy['keyless_recovery_success_pct']:.1f}% |",
        f"| The same, with the first 500 pixels left unaltered as the 2024 app did | {legacy['keyless_recovery_with_500_unaltered_pixels_pct']:.1f}% |",
        f"| Images damaged by its own decryption | {legacy['images_damaged_by_round_trip_pct']:.1f}% |",
        f"| Mean MSE after decryption (a cipher must give 0) | {legacy['round_trip_mse_mean']:.2f} |",
        f"| Mean PSNR over the damaged images | {legacy['round_trip_psnr_db_mean_over_damaged_images']:.2f} dB |",
        f"| Pixel values destroyed, mean / worst image | {legacy['pixel_values_lost_pct_mean']:.2f}% / {legacy['pixel_values_lost_pct_max']:.2f}% |",
        f"| Stored size, three residue PNGs vs the original PNG | {legacy['storage_ratio_png_shares_vs_original']:.2f}x |",
        "",
        "## Table 2. Can a face detector still find the face?",
        "",
        "| Image given to the detector | Face found |",
        "|---|---|",
        f"| Original | {found['original']:.1f}% |",
    ]
    for m in legacy["moduli"]:
        lines.append(f"| 2024 scheme, mod {m} file as stored | {found[f'share_mod_{m}_as_stored']:.1f}% |")
        lines.append(f"| 2024 scheme, mod {m} file brightened | {found[f'share_mod_{m}_brightened']:.1f}% |")
    lines += [
        f"| 2024 scheme, rebuilt with no key | {found['recovered_without_key']:.1f}% |",
        f"| FaceVault protected image | {fv['face_detected_in_protected_pct']:.1f}% |",
        "",
        f"The observer's detector counts a face at confidence {OBSERVER_THRESHOLD}. FaceVault locks at confidence {fv['lock_threshold']}.",
        "",
        "## Table 2b. Faces the lock step misses, by lock confidence",
        "",
        "The third column separates a face seen through the noise from a face the lock step never detected.",
        "Counts move by a few images between runs because the ciphertext noise is random.",
        "",
        "| Lock confidence | Images where a face is still found | of which centred on a locked region | Regions locked per image | Area locked |",
        "|---|---|---|---|---|",
        *[f"| {row['lock_threshold']} | {row['images_with_face_still_found']} of {results['meta']['images']} ({row['images_with_face_still_found_pct']:.1f}%) | "
          f"{row['images_with_detection_inside_a_locked_region']} | {row['regions_locked_per_image_mean']:.2f} | {row['area_locked_pct_mean']:.1f}% |" for row in results["lock_threshold"]],
        "",
        "## Table 2c. Blur, pixelation and a black box against FaceVault",
        "",
        "The same face regions hidden by each method. The recogniser (SFace) is told exactly where the face is",
        f"and counts a match at similarity {detect.MATCH_THRESHOLD} or above. 'Same photo': the attacker holds the unprotected",
        f"copy of that photo ({results['obfuscation'][0]['images']} images). 'Another photo': the attacker holds a different photo of the",
        f"same person, as a watch-list would ({results['obfuscation'][0]['people']} people). 'Wrong person': the hidden face compared with",
        "someone else's photo, which shows what chance alone produces.",
        "",
        "| Method | Face detected in the hidden region | Recognised against the same photo | Recognised against another photo (mean similarity) | Matched to a wrong person (mean similarity) | Original recoverable |",
        "|---|---|---|---|---|---|",
        *[f"| {row['method']} | {row['face_detected_in_hidden_region_pct']:.1f}% | {row['matched_to_same_photo_pct']:.1f}% | "
          f"{row['matched_to_other_photo_pct']:.1f}% ({row['similarity_to_other_photo_mean']:.3f}) | "
          f"{row['matched_to_wrong_person_pct']:.1f}% ({row['similarity_to_wrong_person_mean']:.3f}) | "
          f"{'not needed' if row['original_recoverable'] is None else 'yes, bit for bit' if row['original_recoverable'] else 'no'} |"
          for row in results["obfuscation"]],
        "",
        *recogniser_table(results["obfuscation"]),
        *crowd_tables(results),
        "## Table 3. Statistics of the stored data",
        "",
        "| Data | Entropy (bits/byte) | Adjacent-pixel correlation |",
        "|---|---|---|",
        f"| Original image | {legacy['entropy_bits']['original']:.3f} | {legacy['adjacent_correlation']['original']:.3f} |",
    ]
    for m in legacy["moduli"]:
        lines.append(f"| 2024 scheme, mod {m} file | {legacy['entropy_bits'][f'share_mod_{m}']:.3f} | {legacy['adjacent_correlation'][f'share_mod_{m}']:.3f} |")
    sim = fv["sface_similarity_original_vs_protected"]
    lines += [
        f"| FaceVault face region (per region) | {fv['ciphertext_entropy_bits_per_region_mean']:.3f} | {fv['ciphertext_adjacent_correlation_mean']:.4f} |",
        f"| FaceVault face regions (all pooled) | {fv['ciphertext_entropy_bits_pooled']:.5f} | |",
        "",
        "## Table 4. FaceVault",
        "",
        "| Measurement | Value |",
        "|---|---|",
        f"| Images with a face / total | {fv['images_with_a_face']} / {fv['images']} |",
        f"| Unlocked image identical to the original, bit for bit | {fv['bit_exact_reveal_pct']:.1f}% |",
        f"| Images with any pixel changed outside the face regions | {fv['images_with_any_change_outside_faces']} |",
        f"| Face area encrypted, mean share of the image | {fv['face_area_pct_of_image_mean']:.1f}% |",
        f"| NPCR / UACI between two encryptions differing by one input bit | {fv['npcr_pct_mean']:.2f}% / {fv['uaci_pct_mean']:.2f}% |",
        f"| Rejected: one bit flipped in a face region | {fv['rejected_pct']['one_bit_flipped_in_face_region']:.1f}% |",
        f"| Rejected: context changed in the header | {fv['rejected_pct']['context_changed_in_header']:.1f}% |",
        f"| Rejected: wrong private key | {fv['rejected_pct']['wrong_private_key']:.1f}% |",
        f"| Face-recognition similarity, original vs protected (mean / max) | {sim['mean']:.3f} / {sim['max']:.3f} |",
        f"| Protected faces still matching their original (threshold {sim['match_threshold']}) | {sim['pct_at_or_above_match_threshold']:.1f}% |",
        f"| Median time per 250x250 image: detect / protect / unlock | {fv['time_ms_median']['detect']:.1f} / {fv['time_ms_median']['protect']:.2f} / {fv['time_ms_median']['reveal']:.2f} ms |",
        f"| Stored size, protected PNG vs original PNG | {fv['storage_ratio_png_protected_vs_original']:.2f}x |",
        f"| Header size, mean | {fv['header_bytes_mean']:.0f} bytes |",
        "",
        "## Table 5. Watch-list matching (SFace, one enrolled image per person)",
        "",
        "| Measurement | Value |",
        "|---|---|",
        f"| People tested | {wl['people']} |",
        f"| Same person accepted (threshold {wl['threshold']}) | {wl['true_accept_pct']:.1f}% |",
        f"| Different person wrongly accepted | {wl['false_accept_pct']:.1f}% |",
        f"| Mean similarity, same / different person | {wl['genuine_similarity_mean']:.3f} / {wl['impostor_similarity_mean']:.3f} |",
        "",
        "## Table 6. Speed at camera resolutions (median, one face in frame)",
        "",
        "| Resolution | Face area | Detect | FaceVault protect | of which background hash | Whole-frame AES-GCM | 2024 encode | 2024 decode |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results["speed"]:
        lines.append(
            f"| {row['resolution']} | {row['face_area_pct']:.1f}% | {row['detect_ms']:.1f} ms | {row['facevault_protect_ms']:.2f} ms | "
            f"{row['of_which_background_hash_ms']:.2f} ms | {row['whole_frame_aes_gcm_ms']:.2f} ms | "
            f"{row['legacy_encode_ms']:.1f} ms | {row['legacy_decode_ms']:.1f} ms |")
    lines += [
        "",
        "## Table 7. Threshold key sharing (256-bit key)",
        "",
        "| k of n | k-subsets that rebuilt the key | Trials where k-1 shares gave the key | One-time prime search | Split | Combine |",
        "|---|---|---|---|---|---|",
    ]
    for row in results["sharing"]:
        lines.append(
            f"| {row['k']} of {row['n']} | {row['k_subsets_correct']} / {row['k_subsets_tested']} | "
            f"{row['k_minus_1_recovered_key']} / {row['k_minus_1_trials']} | {row['one_time_setup_ms']:.0f} ms | {row['split_ms']:.2f} ms | {row['combine_ms']:.3f} ms |")
    (RESULTS / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--people", type=int, default=500, help="identities in the watch-list test")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--crowd-images", type=int, help="use only the first N WIDER FACE images (for a quick check)")
    parser.add_argument("--only", help="comma-separated steps to rerun, keeping the other saved results (e.g. obfuscation)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_paths = lfw_paths()
    paths = [all_paths[i] for i in sorted(rng.choice(len(all_paths), size=min(args.images, len(all_paths)), replace=False))]
    detector, matcher = detect.FaceDetector(), detect.FaceMatcher()
    observer = detect.FaceDetector(OBSERVER_THRESHOLD)
    RESULTS.mkdir(exist_ok=True)

    results = {"meta": {
        "date": time.strftime("%Y-%m-%d"),
        "images": len(paths),
        "seed": args.seed,
        "dataset": "LFW (funneled), random sample",
        "machine": f"{platform.processor() or platform.machine()}, Python {platform.python_version()}, OpenCV {cv2.__version__}, CPU only",
    }}
    steps = {
        "legacy": lambda: run_legacy(paths, observer),
        "facevault": lambda: run_facevault(paths, detector, observer, matcher, rng),
        "lock_threshold": lambda: run_lock_threshold(paths, observer),
        "obfuscation": lambda: run_obfuscation(paths, photo_pairs(all_paths, args.people, args.seed), detector, observer, matcher,
                                               research_models.ArcFaceMatcher() if research_models.ARCFACE.exists() else None),
        "crowd": lambda: run_crowd(args.crowd_images),
        "watchlist": lambda: run_watchlist(all_paths, detector, matcher, args.people, rng),
        "speed": lambda: run_speed(detector),
        "sharing": run_sharing,
    }
    if not args.only and not (WIDER_DIR / "wider_face_split").exists():
        print("WIDER FACE not downloaded: skipping the crowd step")
        del steps["crowd"]
    if args.only:
        saved = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
        if (saved["meta"]["images"], saved["meta"]["seed"]) != (len(paths), args.seed):
            sys.exit("--only needs the same --images and --seed as the saved results")
        results = saved
        steps = {name: steps[name] for name in args.only.split(",") if name != "none"}   # "--only none" redraws tables and figures
    for name, step in steps.items():
        start = time.perf_counter()
        results[name] = step()
        print(f"{name}: done in {time.perf_counter() - start:.0f} s", flush=True)

    (RESULTS / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_tables(results)
    make_figures(results, detector)
    print(f"wrote {RESULTS / 'results.json'}, tables.md and figures/")


if __name__ == "__main__":
    main()
