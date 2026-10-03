# Advanced Surveillance System with Encrypted Alerts and Privacy Enhancement (2024)

The original final-year project, kept here as it was so it can be found next
to its rebuild, [FaceVault](../README.md).

- **When:** 2023-24, B.E. project, Dept. of CSE (AI & ML), Vidyavardhaka
  College of Engineering, Mysuru
- **App name on screen:** *Surveillance System for Criminal Detection*
- **Built with:** Python, Streamlit, the `face_recognition` library (dlib),
  NumPy, Pillow

## Published paper

A literature survey from this project was published as:

> Ranjan Kumar H S, Nisarga Nab, Paavani M R, Thashik R Paul, Varun S P,
> Dileep M, "Advanced Surveillance System with Encrypted Alerts and Privacy
> Enhancement", *International Journal of Scientific Research in Engineering
> and Management (IJSREM)*, vol. 8, no. 3, March 2024.
> DOI: [10.55041/IJSREM29001](https://doi.org/10.55041/IJSREM29001)

The paper reviews twelve works on image encryption and object detection. It
does not describe the implementation; the implementation is what this folder
holds.

## What the system does

```
                      Home page
        ┌─────────────────┼──────────────────┐
 Training images     Test images       Encrypted images
 (known faces)            │                  │
                    known face found?     Decrypt
                     no │      │ yes     (Chinese Remainder
                   Discard   Encrypt      Theorem)
```

1. **Training images.** A folder with one photo per known person. Each face is
   turned into an encoding and the encodings are saved to a file.
2. **Test images.** Every face in each test image is compared with the known
   encodings. The app lists who was found.
3. **Encryption.** An image containing a known face is written out as three
   images: every pixel value is replaced by its remainder modulo 3, modulo 5
   and modulo 17 (a residue number system). They are saved as
   `output_folder/<image name>/encrypted_0.png`, `encrypted_1.png` and
   `encrypted_2.png`. The first 500 pixels are left unaltered.
4. **Decryption.** The three images are combined pixel by pixel with the
   Chinese Remainder Theorem and saved as `decrypted_image.png` in the same
   folder.

The project report gave one measurement, for one image: mean squared error
13.56 and PSNR 36.81 dB between the original and the decrypted image.

## About this code

**This is a reconstruction.** The original source files were not kept.
`app.py` was rebuilt in 2026 from the pseudo-code, the results chapter and the
screenshots in the project report. Function names, screen text, folder and
file names, the moduli and the 500-pixel rule follow the report. Two things
are known to differ: NumPy is used where the report describes pixel-by-pixel
loops, and the decorative background image is left out.

What has been checked:

| Part | Status |
|---|---|
| Encryption and decryption functions | Tested (`tests/test_original_2024.py`) |
| Decryption screen | Tested with Streamlit's test runner |
| Face matching and the Encryption screen | **Not run.** They need dlib, which was not installed when this was rebuilt |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

`face_recognition` installs dlib, which is compiled during installation. On
Windows that needs CMake and the Visual Studio C++ build tools. Decryption
works without it.

## Known weaknesses

A later review found that this scheme does not protect the images:

- There is no key. Anyone holding the three encrypted images can rebuild the
  picture, and the moduli can be read off the files.
- 3 × 5 × 17 = 255, so pixel value 255 cannot be stored and comes back as 0.
  That is where the reported error of 13.56 comes from.
- Alerts, access control and real-time detection, which the title and design
  describe, were not built.

The details and measurements are in
[docs/review-of-2024-report.md](../docs/review-of-2024-report.md), and
[FaceVault](../README.md) is the version that fixes them. Do not use this
folder's code to protect real images.

## Not included

The college project report and its test photographs are not published here,
because they contain other people's photographs and personal details.
