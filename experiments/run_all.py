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

from facevault import attacks, detect, hybrid, legacy_rns, metrics, obfuscate, sharing, vault  # noqa: E402

LFW_DIR = ROOT / "data" / "lfw_home" / "lfw_funneled"
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
    moduli_inferred = damaged = 0
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


def run_obfuscation(paths, pairs, detector, observer, matcher):
    """Hide the same face regions with each method; ask a detector and a recogniser what is left."""
    _, public = hybrid.generate_keypair()
    methods = obfuscation_methods(public)

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
    enrolled, probes = [], []
    for enrolled_path, probe_path in pairs:
        enrolled_rgb, probe_rgb = vault.load_image(enrolled_path), vault.load_image(probe_path)
        enrolled_faces, probe_faces = detector.detect(enrolled_rgb), detector.detect(probe_rgb)
        if enrolled_faces and probe_faces:
            enrolled.append(matcher.embed(enrolled_rgb, enrolled_faces[0]))
            probes.append((probe_rgb, probe_faces))
    people = len(probes)
    other_photo = [[] for _ in methods]
    wrong_person = [[] for _ in methods]
    for person, (probe_rgb, probe_faces) in enumerate(probes):
        _, versions = hidden_versions(probe_rgb, probe_faces)
        for index, hidden in enumerate(versions):
            feature = matcher.embed(hidden, probe_faces[0])
            other_photo[index].append(matcher.similarity(enrolled[person], feature))
            wrong_person[index].append(matcher.similarity(enrolled[(person + 1) % people], feature))

    def matched(scores):
        return 100.0 * float(np.mean(np.array(scores) >= detect.MATCH_THRESHOLD))

    return [{
        "method": label,
        "images": used,
        "people": people,
        "face_detected_in_hidden_region_pct": 100.0 * detected[index] / used,
        "matched_to_same_photo_pct": matched(same_photo[index]),
        "matched_to_other_photo_pct": matched(other_photo[index]),
        "matched_to_wrong_person_pct": matched(wrong_person[index]),
        "similarity_to_same_photo_mean": mean(same_photo[index]),
        "similarity_to_other_photo_mean": mean(other_photo[index]),
        "similarity_to_wrong_person_mean": mean(wrong_person[index]),
        "original_recoverable": recoverable,
    } for index, (label, _, recoverable) in enumerate(methods)]


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
BLUE, ORANGE = "#2a78d6", "#eb6834"


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
        "obfuscation": lambda: run_obfuscation(paths, photo_pairs(all_paths, args.people, args.seed), detector, observer, matcher),
        "watchlist": lambda: run_watchlist(all_paths, detector, matcher, args.people, rng),
        "speed": lambda: run_speed(detector),
        "sharing": run_sharing,
    }
    if args.only:
        saved = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
        if (saved["meta"]["images"], saved["meta"]["seed"]) != (len(paths), args.seed):
            sys.exit("--only needs the same --images and --seed as the saved results")
        results = saved
        steps = {name: steps[name] for name in args.only.split(",")}
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
