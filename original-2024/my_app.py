import os
import streamlit as st
from PIL import Image
import numpy as np
import face_recognition
import pickle
import shutil
import cv2

def multiplicative_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

def crt(pixels, moduli):
    total = 0
    prod = np.prod(moduli, dtype=np.uint64)
    for pixel, modulus in zip(pixels, moduli):
        p = prod // modulus
        inv = multiplicative_inverse(p, modulus)
        total += pixel * inv * p
    return total % prod

def decrypt_image(encrypted_paths, moduli):
    encrypted_images = [np.array(Image.open(path), dtype=np.uint64) for path in encrypted_paths]

    decrypted_img_array = np.zeros_like(encrypted_images[0])

    for i in range(decrypted_img_array.shape[0]):
        for j in range(decrypted_img_array.shape[1]):
            pixels = [img[i, j] for img in encrypted_images]
            decrypted_img_array[i, j] = crt(pixels, moduli)

    decrypted_img_array = decrypted_img_array.astype(np.uint8)

    return Image.fromarray(decrypted_img_array)

def decrypt_folder_for_filename(folder_path, filename):
    folder_path = os.path.join(folder_path, filename)
    encrypted_paths = [os.path.join(folder_path, f'encrypted_{i}.png') for i in range(3)]
    moduli = [3, 5, 17]
    decrypted_img = decrypt_image(encrypted_paths, moduli)
    decrypted_img.save(os.path.join(folder_path, 'decrypted_image.png'))

    st.success("Decryption successful!")
    st.write(f"Decrypted image can be found inside the folder named '{filename}'.")

def main():
    st.title("Surveillance System")

    # Encryption Section
    st.markdown("## Encryption")

    # Input for training images path
    train_path = st.text_input("Enter the path to upload training images", "")

    # Button to open training images folder
    if st.button("Open Train Images Folder"):
        if os.path.exists(train_path):
            os.startfile(train_path)
        else:
            st.error("Please provide a valid path.")

    # Input for test images path
    test_path = st.text_input("Enter the path to upload images for detection and encryption", "")

    # Button to open test images folder
    if st.button("Open Test Images Folder"):
        if os.path.exists(test_path):
            os.startfile(test_path)
        else:
            st.error("Please provide a valid path.")

    # Button for encryption
    if st.button("Encrypt"):
        def process_images(directory):
            face_data = {"encodings": [], "filenames": [], "locations": []}
            for filename in os.listdir(directory):
                if filename.endswith(('.jpg', '.png')):
                    image_path = os.path.join(directory, filename)
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image)
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    for encoding, location in zip(face_encodings, face_locations):
                        face_data["encodings"].append(encoding)
                        face_data["locations"].append(location)
                        face_data["filenames"].append(filename)

            with open("stored_face_data.pkl", "wb") as f:
                pickle.dump(face_data, f)

        def search_face(target_face_encoding):
            with open("stored_face_data.pkl", "rb") as f:
                stored_face_data = pickle.load(f)
            matching_faces = []
            for i, stored_encoding in enumerate(stored_face_data["encodings"]):
                if face_recognition.compare_faces([stored_encoding], target_face_encoding)[0]:
                    matching_faces.append(stored_face_data["filenames"][i])
            return matching_faces

        def detect_and_search_faces(image_path):
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            matching_photos = []
            for i, face_encoding in enumerate(face_encodings):
                matching_photos.extend(search_face(face_encoding))
            return image, matching_photos
        
        def remove_duplicates_from_dict_values(di):
            for key, value in di.items():
        # Convert the list of values to a set to remove duplicates
                unique_values = set(value)
        # Convert the set back to a list
                di[key] = list(unique_values)
            return di

        def encrypt_image(image_path, output_folder):
            try:
                img = Image.open(image_path)
                img_array = np.array(img)

                encrypted_images = [np.zeros_like(img_array) for _ in range(3)]

                for i in range(img_array.shape[0]):
                    for j in range(img_array.shape[1]):
                        if i * img_array.shape[1] + j < 500:
                            for k in range(3):
                                encrypted_images[k][i, j] = img_array[i, j]
                        else:
                            encrypted_images[0][i, j] = img_array[i, j] % 3
                            encrypted_images[1][i, j] = img_array[i, j] % 5
                            encrypted_images[2][i, j] = img_array[i, j] % 17

                # Create the output folder if it doesn't exist
                os.makedirs(output_folder, exist_ok=True)

            # Save the encrypted images
                for k in range(3):
                    encrypted_img = Image.fromarray(encrypted_images[k])
                    encrypted_img.save(os.path.join(output_folder, f"encrypted_{k}.png"))

                print("Encryption completed successfully!")
                
            except Exception as e:
                print(f"Error during encryption: {e}")
                st.error("Error during encryption")


        process_images(train_path)

        with open("stored_face_data.pkl", "rb") as f:
            stored_face_data = pickle.load(f)

        images_with_persons = {}
        output_file = open("output.txt", "w")

        for filename in os.listdir(test_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                written_persons = set()
                target_image_path = os.path.join(test_path, filename)
                image, matching_faces = detect_and_search_faces(target_image_path)
                if matching_faces:
                    images_with_persons[filename] = matching_faces
                output_file.write("Faces detected in " + filename + ":\n")
                st.image(image, caption='Original Image - ' + filename, use_column_width=True)
                st.write("Faces detected in " + filename + ":\n")
                for person in matching_faces:
                    if person not in written_persons:
                        output_file.write("- Person: " + person + "\n")
                        st.write("- Person: " + person + "\n")
                        written_persons.add(person)
                output_file.write("\n")
                #st.success("Encrypted image successfully")
        output_file.close()
                    
        images_with_persons = remove_duplicates_from_dict_values(images_with_persons)
        print("Images with detected persons:", images_with_persons)
        
        for key in images_with_persons.keys():
            filename = os.path.join(test_path, key)
            if os.path.isfile(filename):
                input_image_file = filename
                
                # Extract subfolder name (e.g., 'akshaykumar' from 'akshaykumar.jpg')
                subfolder_name = os.path.splitext(key)[0]
                # Create subdirectory inside output folder with the name of subfolder_name
                output_subdirectory = os.path.join("output_folder", subfolder_name)
                os.makedirs(output_subdirectory, exist_ok=True)
                
                #########encrypt 
                output_folder_path = output_subdirectory
                encrypt_image(input_image_file, output_folder_path)

        st.success('Encryption successful and stored within downloads and the folder is named as encrypted_images')

        print("Images with detected persons:", images_with_persons)

    # Decryption Section
    st.markdown("## Decryption")

    # Input for encrypted images folder path
    folder_path = st.text_input("Enter the  folder of the 'encrypted_images' folder should be in downloads and named as encrypted_iamges", "")

    # Button to open encrypted images folder
    if st.button("Open encrypted images folder"):
        if os.path.exists(folder_path):
            os.startfile(folder_path)
        else:
            st.error("Path does not exist.")

    # Input for folder name inside the encrypted_images folder
    filename = st.text_input("Enter the folder name inside the folder encrypted_images", "")

    # Button to trigger decryption
    if st.button("Decrypt"):
        decrypt_folder_for_filename(folder_path, filename)

if __name__ == "__main__":
    main()