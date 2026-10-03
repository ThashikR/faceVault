# Advanced Surveillance System with Encrypted Alerts and Privacy Enhancement (2024)

The original final-year project, kept here as it was so it can be found next
to its rebuild, [FaceVault](../README.md).

- **When:** 2023-24, B.E. project, Dept. of CSE (AI & ML), Vidyavardhaka
  College of Engineering, Mysuru
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

1. **Training images.** A folder of photos of known people. Each face is
   turned into an encoding and the encodings are saved to
   `stored_face_data.pkl`.
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

## The files

These are the original 2024 files, unchanged.

| File | Date | What it is |
|---|---|---|
| `my_app.py` | 20 Apr 2024 | The Streamlit app: training, face matching, encryption, decryption |
| `development/0_RNS.ipynb` | 26 Feb 2024 | First experiment with residues on a greyscale image |
| `development/1_rns00.py` | 30 Mar 2024 | Residues modulo 3, 5, 17 on the raw bytes of a BMP file |
| `development/2_crt00.py` | 30 Mar 2024 | The matching decryption |
| `development/3_test2_RNS.ipynb` | 31 Mar 2024 | The move from BMP bytes to image pixels, and the CRT functions the app uses |
| `development/4_check_and_delete.ipynb` | 18 Apr 2024 | Face matching joined to the encryption |
| `development/5_send.ipynb` | 13 Aug 2024 | The whole pipeline in two cells, without the screens |

The file names in `development/` have a number added in front to show the
order. Python files are byte-for-byte copies. In the notebooks the code cells
are untouched and the saved outputs have been removed, because the outputs
list the people in the test photographs.

`1_rns00.py` appears to be where the 500 rule comes from: it works on the raw
bytes of a BMP file and leaves the first 500 bytes alone, which keeps the file
header intact so the result still opens as an image. Later versions carried
the same rule over to pixels.

The version shown in the project report's screenshots had a sidebar and the
longer title *Surveillance System for Criminal Detection*. That later file was
not found; `my_app.py` is the latest version that was.

## Run it

```bash
pip install -r requirements.txt
streamlit run my_app.py
```

`face_recognition` needs dlib. On Windows, install a prebuilt dlib wheel that
matches your Python version first, or have CMake and the Visual Studio C++
build tools so that pip can compile it. `os.startfile`, used by the "Open
folder" buttons, exists only on Windows.

## What has been checked

| Part | Status |
|---|---|
| Encryption and decryption functions of `my_app.py` | Run unchanged by `tests/test_original_2024.py` |
| FaceVault's stand-in for this scheme (`facevault/legacy_rns.py`) | Shown by the same tests to give byte-identical output |
| Face matching and the app's screens | **Not run** for this repository, because dlib was not installed |

## Known weaknesses

A later review found that this scheme does not protect the images:

- There is no key. Anyone holding the three encrypted images can rebuild the
  picture, and the moduli can be read off the files.
- 3 × 5 × 17 = 255, so pixel value 255 cannot be stored and comes back as 0.
  That explains the reported error of 13.56.
- Alerts, access control and real-time detection, which the title and design
  describe, were not built.

The details and measurements are in
[docs/review-of-2024-report.md](../docs/review-of-2024-report.md), and
[FaceVault](../README.md) is the version that fixes them. Do not use this
folder's code to protect real images.

## Not included

The college project report and the training and test photographs are not
published here, because they contain other people's photographs and personal
details.
