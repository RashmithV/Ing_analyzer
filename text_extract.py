import os
import easyocr

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from a single image file.
    Returns the extracted text as a string.
    """
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path, detail=0)  # extract text
    extracted_text = "\n".join(results)
    return extracted_text


def extract_text_from_images(input_folder: str, output_folder: str):
    """
    Extract text from all images in input_folder and save to output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    reader = easyocr.Reader(['en'])

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
            print(f"Processing: {file_name}")
            results = reader.readtext(file_path, detail=0)
            extracted_text = "\n".join(results)

            # Save output text
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            print(f"Saved: {output_file}")
        else:
            print(f"Skipped (not an image): {file_name}")


if __name__ == "__main__":
    input_folder = "input"   # put your images here
    output_folder = "output" # extracted text files will be saved here
    extract_text_from_images(input_folder, output_folder)