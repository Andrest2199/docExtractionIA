# %% extract image from PDF

from PyPDF2 import PdfReader
from PIL import Image
from utils.file_utils import FileUtils
from pdf2image import convert_from_path
import re
import os
import shutil

"""
INSTALLATION:
conda install conda-forge::pypdf2
conda install anaconda::pillow
pip install pdf2image
conda install -c conda-forge poppler
conda install conda-forge::poppler


TODO's:
- Delete, escaneado from scanscanner
- In a 2nd version explore other libriares to extract images from PDFs

POTENTIAL IMPROVEMENTS
- Using regex extract employee name, folio ID and employee ID. Andres to ask Adriana de la Rosa.
"""


def pdf_has_text(file_path, string_threshold=200):
    # Extract text
    text_corpus = get_text_from_pdf(file_path)
    # Clean the text
    regex_cleaning_list = [r"\n", r"Escaneado con CamScanner"]
    for ii in range(len(regex_cleaning_list)):
        regex_pattern = regex_cleaning_list[ii]
        text_corpus = re.sub(regex_pattern, "", text_corpus)
    # If the length of the string is cero
    if len(text_corpus) > string_threshold:
        return True
    else:
        return False


def process_images(file_path, output_folder_path) -> None:
    """
    Function to process images from a PDF file
    """

    file_name_procesed = f"{os.path.basename(file_path)}_procesed"

    with Image.open(file_path) as img:
        image_format = img.format
        image_format = image_format.lower()

    file_name_procesed_image = file_name_procesed + "." + image_format

    # Build the full source and target paths
    target_path = os.path.join(output_folder_path, file_name_procesed_image)

    # Copy the image to the target folder with the new name
    shutil.copy2(file_path, target_path)
    procesed_images_list = FileUtils.create_list(output_folder_path)
    return procesed_images_list


# Define function to extract text
def get_text_from_pdf(file_path=str) -> str:
    """
    Function to extract text from a PDF file"""
    # Initialize a PDF reader object and read the PDF
    reader = PdfReader(file_path)

    # Initialize an empty string to hold all the text
    text_corpus = ""

    # Iterate through each page in the PDF and extract text
    for page in reader.pages:
        text_corpus += page.extract_text() + "\n"  # Adding a newline character

    return text_corpus


def pdf_to_image(input_file, output_folder_path) -> list:
    filename = os.path.basename(input_file).strip(".pdf")

    # Store Pdf with convert_from_path function
    try:
        images = convert_from_path(input_file)
        # Save pages as images in the pdf
        for i in range(len(images)):
            output_path = (
                output_folder_path + "/" + filename + " page" + str(i) + ".jpeg"
            )
            images[i].save(output_path, "JPEG")
            images_in_pdf = FileUtils.create_list(output_folder_path)
        return images_in_pdf

    except Exception as e:
        print("Error in convert_pdf_to_image", e)
