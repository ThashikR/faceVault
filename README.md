# FaceVault

Surveillance that keeps faces locked until authorised people unlock them.

A camera image goes in. Every face in it is encrypted in place, so the stored
image shows the scene with noise where the faces were. Unlocking needs key
shares from several officers (two of three by default), every unlock is
logged, and a watch-list match sends an encrypted, signed alert.

![The 2024 scheme and FaceVault on the same image](results/figures/comparison.png)

*Sample image: NASA portrait of Eileen Collins, public domain.*

## Where this came from

FaceVault is a rebuild of a 2024 final-year team project, *Advanced
Surveillance System with Encrypted Alerts and Privacy Enhancement*. That
version "encrypted" images by storing each pixel's remainders modulo 3, 5 and
17. Reviewing it showed that this is not encryption: there is no key, so the
stored files give the image back to anyone, and pixel value 255 is destroyed
on the way. The full review is in
[docs/review-of-2024-report.md](docs/review-of-2024-report.md).

The old scheme is kept in this repository (`facevault/legacy_rns.py`) together
with the attacks on it (`facevault/attacks.py`), so the comparison can be
reproduced.

## How it works

| Part | What it does | Built with |
|---|---|---|
| Face detection | Finds faces on the CPU | OpenCV YuNet |
| Locking | Encrypts each face rectangle and writes the ciphertext into the same pixels | AES-256-GCM |
| Public-key camera | The camera derives each image's key from the vault's public key, so it cannot unlock what it locked | X25519, HKDF-SHA256 |
| Shared custody | The vault's private key is split so any *k* of *n* officers can rebuild it and fewer cannot | Asmuth-Bloom sharing (Chinese Remainder Theorem) |
| Alerts | A watch-list match is sent encrypted to the recipient and signed by the camera | SFace, X25519, ChaCha20-Poly1305, Ed25519 |
| Audit log | Every unlock and every refused attempt is recorded in a hash-chained file | SHA-256 |

The protected image is one PNG file. It carries everything needed to unlock it
except the private key. Changing a single bit of a face region, of the header,
or of the camera/time context makes unlocking fail; editing the rest of the
picture is reported.

No new cipher is invented here. The Chinese Remainder Theorem, which the 2024
project used on pixels, is used for the job it suits: sharing a key.

## Results

Measured on 1,000 randomly sampled images from the public LFW face dataset, on
a laptop CPU. Every number comes from `python experiments/run_all.py`; the full
tables are in [results/tables.md](results/tables.md).

**The 2024 residue scheme**

| Measurement | Result |
|---|---|
| Images rebuilt from the stored files alone, with no key | 1,000 of 1,000 |
| Images damaged by the scheme's own decryption | 81.0% |
| Mean MSE after decryption (a cipher must give 0) | 244.37 |
| Stored size compared with the original PNG | 2.20x |
| Entropy of the stored files (ideal: 8 bits per byte) | 1.58, 2.30 and 4.02 bits |

**FaceVault**

| Measurement | Result |
|---|---|
| Unlocked image identical to the original, bit for bit | 1,000 of 1,000 |
| Pixels changed outside the face regions | 0 |
| Protected images in which a face detector still finds a face | 0 of 1,000 |
| Protected faces a face recogniser still matches to the original | 0 of 1,000 |
| Rejected: one bit flipped in a face region / changed context / wrong key | 100% / 100% / 100% |
| Entropy of the encrypted regions | 8.00000 bits per byte (pooled) |
| Correlation between neighbouring ciphertext pixels | 0.0001 (0.989 in the originals) |
| Time to lock / unlock a 250x250 image, after detection | 1.05 ms / 0.72 ms |
| Time to detect faces in a 1920x1080 frame | 97.7 ms |
| Watch-list matching, 500 people, one enrolled photo each | 96.4% of genuine pairs accepted; 0 of 500 impostor pairs accepted |
| Key sharing: every tested group of *k* officers rebuilt the key | 69 of 69 groups |
| Key sharing: *k* - 1 officers rebuilt the key | 0 of 600 trials |

![Share of images in which a face detector still finds a face](results/figures/face_detection.png)

![Byte histograms of a face region: original, 2024 scheme, FaceVault](results/figures/histograms.png)

Things the numbers also say, which matter as much as the good ones:

- **One residue file alone does not show a face.** A detector found no face in
  any single brightened file. The 2024 scheme fails because its files together
  need no key, not because each file is readable.
- **The detector, not the cipher, limits privacy.** With the lock step at the
  detector's usual confidence of 0.8, a face was still found in 9 of 1,000
  protected images. Every one of those detections lay outside the locked
  regions, on a face the lock step had scored just under its cut-off (in the
  cases inspected, a second person at the edge of the photo); none was a
  locked face. Locking at confidence 0.5
  brought this to 0 of 1,000 while locking 1.1 percentage points more of the
  image. The 0.5 setting was chosen on this same sample, so it needs checking
  on other data.
- **Locking only the faces is not faster than encrypting the whole frame.**
  At 1920x1080, FaceVault's lock step takes 13.6 ms against 6.9 ms for
  whole-frame AES-GCM; 11.1 ms of that is hashing the background for tamper
  detection. The reason to lock only faces is that the scene stays usable,
  not speed. Face detection (97.7 ms) dominates either way.
- **Protected files are larger.** Encrypted regions do not compress, so the
  protected PNG is 1.42x the original on these face-filling images.
- Counts involving the detector on protected images can move by a few images
  between runs, because the ciphertext noise is random.

## Run it

Needs Python 3.10 or newer. Tested on Windows 11 with Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/fetch_models.py
```

On macOS or Linux the second line is `source .venv/bin/activate`.

Demo app:

```bash
streamlit run app.py
```

Command line:

```bash
python -m facevault setup --keys keys --threshold 2 --officers 3
python -m facevault protect photo.jpg -o protected.png --watchlist people/
python -m facevault reveal protected.png -o unlocked.png --share keys/shares/officer-1.json --share keys/shares/officer-3.json --reason "case 42"
python -m facevault read-alert outbox/<file>.alert.json
python -m facevault audit
python -m facevault legacy photo.jpg
```

Tests:

```bash
pytest
```

Reproduce the measurements (downloads the LFW dataset, about 230 MB, on first use):

```bash
python -c "from sklearn.datasets import fetch_lfw_people; fetch_lfw_people(data_home='data', min_faces_per_person=100)"
python experiments/run_all.py
```

## Layout

```
facevault/
  vault.py        lock and unlock face regions; protected PNG format
  hybrid.py       public-key key agreement
  sharing.py      k-of-n key sharing
  alerts.py       encrypted, signed alerts
  audit.py        hash-chained unlock log
  detect.py       face detection and watch-list matching
  pipeline.py     the workflow that ties the parts together
  cli.py          command line
  legacy_rns.py   the 2024 scheme, kept as a baseline
  attacks.py      attacks on the 2024 scheme
  metrics.py      entropy, correlation, NPCR, UACI, MSE, PSNR
app.py            Streamlit demo
experiments/      the script behind every number in this README
tests/            pytest suite
docs/             technical review of the 2024 report
```

## Limits

- Still images only. Video would need the ciphertext to survive the video
  codec, which this design does not attempt.
- Protected images must be stored losslessly (PNG).
- A face the detector misses is not locked. Privacy is bounded by the
  detector, not by the cipher, which is why locking uses a low detection
  confidence (see Results).
- Hair, clothing and surroundings stay visible and can identify someone to a
  person who knows them.
- Asmuth-Bloom sharing hides the key statistically, not perfectly. Shamir's
  scheme is the usual production choice.
- The audit log detects edits. It cannot stop someone who can rewrite the whole
  file unless its latest hash is also kept somewhere else.
- The demo keeps every key on one machine. A real deployment puts the camera
  key, the alert recipient's key and each officer's share on separate devices.
- This is a student research project and has not had an independent security
  review.

## Responsible use

Face recognition in public places is restricted or banned in many
jurisdictions. Use this project only with images you have the right to
process: your own, consenting participants', or public research datasets.
The measurements here use LFW for aggregate statistics only; no LFW image is
included in this repository.

## Credits

- Original 2024 project: a five-student team (the author among them) and
  their guide at the Dept. of CSE (AI & ML), Vidyavardhaka College of
  Engineering, Mysuru.
- This rebuild: Thashik R Paul, developed with AI assistance (Claude).
- Face models: [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (YuNet, SFace).
- Dataset: Labeled Faces in the Wild, University of Massachusetts Amherst.

MIT licence. See [LICENSE](LICENSE).
