from utils.file_utils import FileUtils
from utils.general_utils import Utils
from image_pre_procesing import (
    pdf_has_text,
    get_text_from_pdf,
    pdf_to_image,
    process_images,
)
from improve_image_quality import improve_image_quality
from ocr_aws_textract import extract_text_from_image
from base64 import b64encode
import os

image_preprocessed_folder = os.path.join(os.getcwd(), "1_image_preprocessed")
image_improved_folder = os.path.join(os.getcwd(), "2_image_improved")


def document_handler(file_path=str, doctype=str) -> tuple|list:
    """
    This function is the main function that handles the document processing. It identifies the type of file and processes it accordingly.
    
        Args:
            file_path (str): The file path that is going to be processed.
            doctype (str): Type of document to process. It can be one of the following: IMSS, INFONAVIT, SAT. _optional_
        
        Returns:
            tuple: 
                with the text extracted, the improved image path and level of confidence.
            list:
                with the texts extracted from each page of a document and their improved images path and level of confidence.
    """
    # Borramos archivos dentro de las carpetas de prepocesado y mejora de documentos
    FileUtils.delete_from_folder(image_preprocessed_folder)
    FileUtils.delete_from_folder(image_improved_folder)

    file_name = os.path.basename(file_path)
    filetype = FileUtils.identify_file(file_name)
    text_corpus_pdf = ""

    if filetype == "pdf":
        """
        If the file is a PDF, check if it has text. If it does, extract the text and save it as a .txt file.
        """
        has_text = pdf_has_text(file_path)
        if has_text:
            text_corpus_pdf = get_text_from_pdf(file_path)

        images_list = pdf_to_image(file_path, image_preprocessed_folder)
    else:
        """
        If the file is an image, process it as an image.
        """
        images_list = process_images(file_path, image_preprocessed_folder)

    # Itera sobre las paginas del documento y mejoramos la calidad
    for image in images_list:
        image_path = os.path.join(image_preprocessed_folder, image)
        improve_image_quality(image_path, image_improved_folder)

    # Crea lista de imagenes mejoradas
    improved_images_list = FileUtils.create_list(image_improved_folder)
    # Crea variable para multiples paginas en un documento
    all_text_corpus = []
    # Itera sobre la imagenes mejoradas y aplica OCR
    for improved_image in improved_images_list:
        improved_image_path = os.path.join(image_improved_folder, improved_image)
        
        text_corpus_ocr, docConfidence, hasManuscript = extract_text_from_image(improved_image_path)
       
        text_corpus = text_corpus_ocr
        
        if len(text_corpus) < len(text_corpus_pdf):
            text_corpus = text_corpus_pdf
            docConfidence = 100
        
        if len(improved_images_list) == 1:
            return text_corpus, docConfidence, hasManuscript, improved_image_path
        else:
            all_text_corpus.append([text_corpus, docConfidence, hasManuscript, improved_image_path])
    return all_text_corpus, None, False, None