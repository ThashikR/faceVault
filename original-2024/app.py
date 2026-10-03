"""Surveillance System for Criminal Detection: the 2024 final-year project app.

RECONSTRUCTION. The original source files were not kept. This file was rebuilt
in 2026 from the pseudo-code (section 6.4), the results chapter and the
screenshots of the project report, and it behaves as the report describes,
weaknesses included. See README.md in this folder.

Run:  streamlit run app.py
"""

import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

MODULI = (3, 5, 17)        # report, chapter 7
UNALTERED_PIXELS = 500     # "the first 500 pixels of the image remained unaltered"
FACE_DATA_FILE = "face_data.pkl"
OUTPUT_FOLDER = "output_folder"
IMAGE_TYPES = (".jpg", ".jpeg", ".png")


# --- helper functions --------------------------------------------------------

def multiplicative_inverse(a, m):
    """Return x such that (a * x) % m == 1."""
    return pow(a, -1, m)


def crt(pixels, moduli):
    """Chinese Remainder Theorem: rebuild pixel values from their remainders."""
    product = 1
    for m in moduli:
        product *= m
    total = np.zeros(pixels[0].shape, dtype=np.int64)
    for remainder, m in zip(pixels, moduli):
        partial = product // m
        total += remainder.astype(np.int64) * partial * multiplicative_inverse(partial, m)
    return total % product


def decrypt_image(encrypted_paths, moduli):
    """Rebuild the image from its three encrypted images."""
    pixels = [np.array(Image.open(path).convert("RGB")) for path in encrypted_paths]
    return crt(pixels, moduli).astype(np.uint8)


def decrypt_folder_for_filename(folder_path, filename):
    """Decrypt output_folder/<filename>/encrypted_*.png and save decrypted_image.png beside them."""
    folder = Path(folder_path) / filename
    encrypted_paths = [folder / f"encrypted_{index}.png" for index in range(len(MODULI))]
    decrypted_path = folder / "decrypted_image.png"
    Image.fromarray(decrypt_image(encrypted_paths, MODULI)).save(decrypted_path)
    return decrypted_path


def encrypt_image(image_path, output_folder):
    """Write one encrypted image per modulus into output_folder/<image name>/."""
    image = np.array(Image.open(image_path).convert("RGB"))
    pixels = image.reshape(-1, 3)
    folder = Path(output_folder) / Path(image_path).stem
    folder.mkdir(parents=True, exist_ok=True)
    encrypted_paths = []
    for index, m in enumerate(MODULI):
        encrypted = pixels % m
        encrypted[:UNALTERED_PIXELS] = pixels[:UNALTERED_PIXELS]
        path = folder / f"encrypted_{index}.png"
        Image.fromarray(encrypted.reshape(image.shape).astype(np.uint8)).save(path)
        encrypted_paths.append(path)
    return encrypted_paths


# --- face recognition --------------------------------------------------------
# face_recognition is imported inside these functions so that decryption still
# works on a machine where dlib is not installed.

def process_training_images(train_path):
    """Encode the face in every training image and save the encodings to a file."""
    import face_recognition

    face_data = {}
    for file in sorted(os.listdir(train_path)):
        if file.lower().endswith(IMAGE_TYPES):
            image = face_recognition.load_image_file(os.path.join(train_path, file))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_data[file] = encodings[0]
    with open(FACE_DATA_FILE, "wb") as handle:
        pickle.dump(face_data, handle)
    return face_data


def load_face_data():
    with open(FACE_DATA_FILE, "rb") as handle:
        return pickle.load(handle)


def search_faces(target_encoding, face_data):
    """Return the training file names whose face matches the target face."""
    import face_recognition

    names = list(face_data)
    if not names:
        return []
    matches = face_recognition.compare_faces([face_data[name] for name in names], target_encoding)
    return [name for name, matched in zip(names, matches) if matched]


def detect_and_search_faces(image_path, face_data):
    """Find the faces in one test image and look each of them up in the training data."""
    import face_recognition

    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)
    matching = []
    for encoding in encodings:
        matching.extend(search_faces(encoding, face_data))
    return image, matching


def remove_duplicates(dictionary):
    return {key: list(dict.fromkeys(values)) for key, values in dictionary.items()}


def open_folder(path):
    if hasattr(os, "startfile") and os.path.isdir(path):
        os.startfile(path)


# --- app ---------------------------------------------------------------------

def main():
    st.title("Surveillance System for Criminal Detection")
    action = st.sidebar.radio("Select Action", ["Encryption", "Decryption"])

    if action == "Encryption":
        st.header("Encryption")
        train_path = st.text_input("Enter the path to upload training images")
        if st.button("Open Train Images Folder"):
            open_folder(train_path)
        test_path = st.text_input("Enter the path to upload images for detection and encryption")
        if st.button("Open Test Images Folder"):
            open_folder(test_path)

        if st.button("Encrypt"):
            process_training_images(train_path)
            face_data = load_face_data()
            detected_persons = {}
            for file in sorted(os.listdir(test_path)):
                if not file.lower().endswith(IMAGE_TYPES):
                    continue
                image_path = os.path.join(test_path, file)
                image, matching = detect_and_search_faces(image_path, face_data)
                detected_persons[file] = matching
                detected_persons = remove_duplicates(detected_persons)
                st.image(image)
                if detected_persons[file]:
                    st.write(f"Faces detected in {file}:")
                    for person in detected_persons[file]:
                        st.write(f"- Person: {person}")
                    encrypt_image(image_path, OUTPUT_FOLDER)
                else:
                    st.write(f"No known face in {file}: discarded")
            st.success(f"Encryption successful and stored within the folder named {OUTPUT_FOLDER}")

    elif action == "Decryption":
        st.header("Decryption")
        folder_path = st.text_input("Enter the folder path containing encrypted images", OUTPUT_FOLDER)
        filename = st.text_input("Enter the folder name inside the encrypted images folder")
        if st.button("Decrypt"):
            decrypted_path = decrypt_folder_for_filename(folder_path, filename)
            st.image(str(decrypted_path))
            st.success(f"Decryption successful: {decrypted_path}")


if __name__ == "__main__":
    main()
