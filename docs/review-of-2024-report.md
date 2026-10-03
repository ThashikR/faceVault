# Review of the 2024 project report

Report reviewed: *Advanced Surveillance System with Encrypted Alerts and Privacy
Enhancement*, B.E. project report, Dept. of CSE (AI & ML), Vidyavardhaka College
of Engineering, 2023-24 (five authors, one guide).

The original 2024 code is in [original-2024/](../original-2024/).

This review is the reason FaceVault exists. Page numbers are the report's own.
Where a claim below is a measurement, the number is in
[results/tables.md](../results/tables.md) and can be reproduced with
`python experiments/run_all.py`.

## What the 2024 system did

1. Compared faces in a test image with a folder of known faces, using the
   `face_recognition` library (p. 25).
2. If a known face was found, wrote the whole image out as three files holding
   each pixel's remainder modulo 3, 5 and 17 (p. 30).
3. Rebuilt the image from the three files with the Chinese Remainder Theorem (p. 27).

## Technical problems

### 1. There is no key

A cipher needs a secret that the attacker lacks. Here the only parameters are
the moduli 3, 5 and 17. They are fixed in the code, printed in the report, and
can be read off the output files: a file of remainders modulo *m* contains the
values 0 to *m* - 1, so its largest value is *m* - 1.

Measured: every sampled image was rebuilt from its three stored files with no
other input (Table 1, first row). The output is exactly what the legitimate
"decryption" produces.

The report calls the residue number system "a form of asymmetric encryption"
(p. 16). It is a number representation. It has no public or private key.

### 2. Decryption damages the image

3 x 5 x 17 = 255, so the three remainders can only tell apart the values 0 to
254. Pixel value 255 has the same remainders as 0 and comes back as 0. That is
the cyan, magenta and black damage on the windows and ceiling in Fig 7.1.6
(p. 35).

The report presents MSE 13.56 and PSNR 36.81 dB as evidence of quality (p. 31).
For a cipher the only acceptable values are MSE 0 and infinite PSNR. Moduli
whose product is at least 256, for example 7, 8 and 9, remove the damage but
not problem 1.

### 3. The stored files are not random

Good ciphertext has close to 8 bits of entropy per byte and no correlation
between neighbouring pixels. The residue files have about 1.6, 2.3 and 4.1 bits
(Table 3), which is simply log2 of each modulus, and their neighbouring pixels
remain correlated (0.13, 0.19 and 0.42, against 0.99 in the original and
0.00 in real ciphertext).

They look black in Fig 7.1.4 only because their values are between 0 and 16.
Brightened, a single file shows the outlines of flat areas of the scene. It
does not show a usable face: a face detector found none in the brightened
files (Table 2). The decisive weakness is problem 1, not this one.

### 4. It does not compress

Section 6.2 says RNS is used "to drop the size of the image" (p. 25-26). Three
residue images replace one image. Stored as PNG they take about twice the
space of the original (Table 1, last row).

### 5. The system described in chapter 5 was not built

Chapter 5 describes real-time detection on video, encrypted alerts to
authorities, multi-factor authentication, role-based access control, a
collaborative analysis platform, penetration testing and continuous monitoring.
The implementation in chapters 6 and 7 has none of these, and "Future
Enhancements" (p. 37) lists real-time detection and e-mail alerts as work still
to do. The title promises "Encrypted Alerts".

### 6. Nothing about security was measured

Chapter 7 reports one MSE and one PSNR for one image. There is no entropy,
correlation, NPCR/UACI, key-sensitivity, tamper or timing measurement, and no
measurement of face-recognition accuracy. The 99.38% on p. 25 is the figure the
`face_recognition` library publishes for the LFW benchmark, not a result of
this project.

### 7. Other points

- The whole image is encrypted, so the protected output is useless for
  monitoring. The title's "privacy enhancement" would call for hiding the
  people and keeping the scene.
- "The first 500 pixels of the image remained unaltered" (p. 30) is not
  explained and leaves part of every image unprotected.

## What FaceVault changes

| 2024 system | FaceVault |
|---|---|
| Residues modulo fixed public numbers, no key | AES-256-GCM under a key derived with X25519 and HKDF-SHA256 |
| Whole image scrambled | Only face regions encrypted; the scene stays usable |
| Pixel value 255 destroyed | Unlocked image identical to the original, bit for bit |
| No integrity protection | Any changed bit in a face region, the header or the context is rejected |
| Anyone with the files can decrypt | The camera holds only a public key; it cannot unlock what it locked |
| CRT used on pixels | CRT used for threshold sharing of the private key (k of n officers) |
| Alerts described, not built | Alerts encrypted to the recipient and signed by the camera |
| Access control described, not built | Every unlock, and every refused attempt, goes into a hash-chained log |
| One MSE figure | Seven tables measured on 1,000 public face images |

## What FaceVault does not claim

- No new cipher. Confidentiality rests on standard primitives from the
  `cryptography` library, on purpose.
- Asmuth-Bloom sharing is statistically, not perfectly, secret. Shamir's
  scheme is the usual choice in production.
- The audit log detects edits; it cannot stop someone who can rewrite the whole
  file unless its latest hash is also kept elsewhere.
- Protected images must be stored losslessly (PNG). JPEG or video compression
  would destroy the ciphertext.
- Hair, clothing, body shape and surroundings stay visible and can identify a
  person to someone who knows them.
- A face the detector misses is not locked. Privacy is bounded by the
  detector, not by the cipher (Table 2b).
