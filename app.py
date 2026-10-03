"""FaceVault demo.   Run:  streamlit run app.py

A single-machine demonstration. In a real deployment the camera, the alert
recipient and each officer would hold their keys on separate devices.
"""

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from facevault import attacks, legacy_rns, metrics, pipeline, sharing, vault
from facevault.audit import AuditLog
from facevault.detect import FaceDetector, FaceMatcher, Watchlist

ROOT = Path(__file__).resolve().parent
KEYS = ROOT / "demo_keys"
OUTBOX = KEYS / "outbox"

st.set_page_config(page_title="FaceVault", page_icon="🔒", layout="wide")


@st.cache_resource
def models():
    return FaceDetector(), FaceMatcher()


def sample_image():
    from skimage import data

    return np.ascontiguousarray(data.astronaut())


def read_upload(upload):
    return np.array(Image.open(upload).convert("RGB"))


def png_download(image, header=None):
    buffer = io.BytesIO()
    if header is None:
        Image.fromarray(image).save(buffer, format="PNG")
    else:
        vault.save_png(buffer, image, header)
    return buffer.getvalue()


def load_shares():
    return {path.stem: sharing.Share.from_json(path.read_text(encoding="utf-8")) for path in sorted((KEYS / pipeline.SHARES_DIR).glob("*.json"))}


st.title("FaceVault")
st.caption("Faces are locked the moment they are captured. Unlocking takes two of three officers, and every unlock is logged.")

if not (KEYS / pipeline.VAULT_PUBLIC).exists():
    st.info("No demo vault on this machine yet.")
    if st.button("Create demo vault (2 of 3 officers)", type="primary"):
        pipeline.setup_keys(KEYS, threshold=2, officers=3)
        st.rerun()
    st.stop()

detector, matcher = models()
tab_protect, tab_reveal, tab_alerts, tab_audit, tab_legacy = st.tabs(
    ["1. Protect", "2. Unlock", "3. Alerts", "4. Audit log", "Why the 2024 scheme fails"])

with tab_protect:
    left, right = st.columns(2)
    with left:
        upload = st.file_uploader("Camera image", type=["jpg", "jpeg", "png"], key="protect_upload")
        watch = st.file_uploader("Watch-list faces (optional; the file name is the label)", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="watch_upload")
        camera = st.text_input("Camera name", "gate-01")
        st.caption("No upload? A public-domain NASA portrait is used as the sample.")
    rgb = read_upload(upload) if upload else sample_image()
    source = upload.name if upload else "sample-astronaut.png"

    watchlist = None
    if watch:
        watchlist = Watchlist(detector, matcher)
        for item in watch:
            if not watchlist.add(Path(item.name).stem, read_upload(item)):
                st.warning(f"No face found in watch-list image {item.name}")

    if st.button("Protect image", type="primary"):
        outcome = pipeline.protect_image(rgb, pipeline.read_key(KEYS / pipeline.VAULT_PUBLIC), detector,
                                         {"camera": camera, "source": source}, watchlist)
        name = f"{Path(source).stem}.protected.png"
        st.session_state["protected"] = (outcome.protected, outcome.header, name)
        st.session_state["protect_note"] = f"{len(outcome.faces)} face(s) found, {len(outcome.header['boxes'])} region(s) locked."
        if outcome.matches:
            pipeline.send_alert(pipeline.make_alert(outcome, name), KEYS, OUTBOX)
            labels = ", ".join(match["label"] for match in outcome.matches)
            st.session_state["protect_note"] += f" Watch-list match: {labels}. An encrypted alert was sent."

    with right:
        if "protected" in st.session_state:
            protected, header, name = st.session_state["protected"]
            st.image(protected, caption="Protected image: this is all the camera stores or transmits")
            st.success(st.session_state["protect_note"])
            st.download_button("Download protected PNG", png_download(protected, header), file_name=name, mime="image/png")
        else:
            st.image(rgb, caption="Original (never stored)")

with tab_reveal:
    left, right = st.columns(2)
    with left:
        upload = st.file_uploader("Protected PNG (or use the one from step 1)", type=["png"], key="reveal_upload")
        shares = load_shares()
        chosen = st.multiselect("Officers presenting their key share", list(shares), default=[])
        reason = st.text_input("Reason for unlocking (recorded in the audit log)")
        go = st.button("Unlock faces", type="primary")
    with right:
        item = None
        if upload:
            try:
                picture = Image.open(upload)
                item = (np.array(picture.convert("RGB")), json.loads(picture.text[vault.PNG_KEY]), upload.name)
            except (KeyError, AttributeError, ValueError):
                st.error("That file is not a FaceVault image.")
        elif "protected" in st.session_state:
            item = st.session_state["protected"]

        if item is None:
            st.info("Protect an image in step 1 first, or upload a protected PNG.")
        elif go:
            protected, header, name = item
            if not reason.strip():
                st.error("A reason is required.")
            else:
                try:
                    result = pipeline.reveal_image(protected, header, [shares[c] for c in chosen], reason.strip(),
                                                   KEYS / pipeline.AUDIT_LOG, name)
                    st.image(result.image, caption="Unlocked: identical to the original, bit for bit")
                    if not result.background_intact:
                        st.warning("The area outside the faces was edited after the image was protected.")
                except (ValueError, vault.TamperedError) as error:
                    st.image(protected, caption="Still locked")
                    st.error(f"Denied: {error}. The attempt was logged.")
        else:
            st.image(item[0], caption="Locked")

with tab_alerts:
    files = sorted(OUTBOX.glob("*.alert.json"), reverse=True) if OUTBOX.exists() else []
    if not files:
        st.info("No alerts yet. Add a watch-list face in step 1 and protect an image that contains that person.")
    for path in files:
        with st.expander(path.name):
            left, right = st.columns(2)
            left.markdown("**What travels over the network**")
            left.code(path.read_text(encoding="utf-8"), language="json")
            right.markdown("**What the recipient reads** (signature checked, then decrypted)")
            right.json(pipeline.read_alert(path, KEYS))

with tab_audit:
    log = AuditLog(KEYS / pipeline.AUDIT_LOG)
    entries = log.entries()
    ok, bad = log.verify()
    if not entries:
        st.info("Nothing has been unlocked yet.")
    else:
        (st.success if ok else st.error)("Log chain intact: no entry was changed or removed." if ok else f"Log chain BROKEN at entry {bad}.")
        st.dataframe([{"#": entry["seq"], "time (UTC)": entry["time"], "action": entry["action"],
                       "officers": ", ".join(entry["actors"]), "image": entry["target"], "reason": entry["reason"]}
                      for entry in entries], hide_index=True, width="stretch")

with tab_legacy:
    st.markdown(
        "The 2024 version stored each pixel as its remainders modulo **3, 5 and 17**. "
        "There is no key, so the stored files give the picture back to anyone, and pure white (255) cannot be stored at all.")
    upload = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="legacy_upload")
    rgb = read_upload(upload) if upload else sample_image()
    residues = legacy_rns.encode(rgb)
    columns = st.columns(4)
    columns[0].image(rgb, caption="Original")
    columns[1].image(residues[2], caption="'Encrypted' file (mod 17) as stored")
    columns[2].image(attacks.stretch_share(residues[2]), caption="Same file, brightened")
    try:
        recovered, moduli = attacks.recover_without_key(residues)
        columns[3].image(recovered, caption=f"Rebuilt with no key (moduli read from the files: {moduli})")
    except ValueError:
        # Happens only on images too plain to contain every remainder, e.g. a single flat colour.
        columns[3].image(legacy_rns.decode(residues), caption="Rebuilt with the published moduli (3, 5, 17)")
    a, b, c = st.columns(3)
    a.metric("MSE after its own decryption", f"{metrics.mse(rgb, legacy_rns.decode(residues)):.2f}", help="A cipher must give 0.")
    b.metric("Pixel values destroyed", f"{attacks.lost_pixel_fraction(rgb) * 100:.2f}%")
    c.metric("Entropy of the mod 17 file", f"{metrics.entropy(residues[2]):.2f} bits", help="Good ciphertext is 8.00.")
