# %% ocr_aws_textract
from dotenv import load_dotenv
import boto3
import os

load_dotenv()
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
REGION_NAME = os.environ.get("REGION_NAME")

aws_config = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "region_name": REGION_NAME,
}

def extract_text_from_image(image_path: str) -> tuple:
    """
    Función para extraer el cuerpo de texto e identificar campos de un formulario.
        
        Args:
            image_path (str): Ruta de la imagen a procesar.

        Libraries:
            Install boto3
            pip install amazon-textract-response-parser
        
        Code samples: 
            https://github.com/aws-samples/amazon-textract-code-samples
    """
    print("Extracting text from image with AWS...")
    # Definimos variables de control
    itemsCount = 0
    docConfidence = 0
    wordConfidence = 0
    wordCount = 0
    hwCount = 0
    hasManuscript = False
    # Amazon Textract client
    textract = boto3.client("textract", region_name=aws_config["region_name"],
    aws_access_key_id=aws_config["aws_access_key_id"],
    aws_secret_access_key=aws_config["aws_secret_access_key"])

    # Leemos document content
    with open(image_path, "rb") as document:
        imageBytes = bytearray(document.read())

    # Llamamos Amazon Textract
    response = textract.detect_document_text(Document={"Bytes": imageBytes})
    # Creamos text corpus
    text_corpus = ""
    text_corpus_words = ""
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            itemsCount += 1
            docConfidence += item["Confidence"]
            text_corpus += item["Text"] + " "
        if item["BlockType"] == "WORD":
            wordCount += 1
            wordConfidence += item["Confidence"]
            text_corpus_words += item["Text"] + " "
            if item["TextType"] == "HANDWRITING":
                hwCount += 1

    docConfidence = (docConfidence / itemsCount) if itemsCount > 0 else 0
    wordConfidence = (wordConfidence / wordCount) if wordCount > 0 else 0
    print (f"HW COUNT: {hwCount}")
    print (f"W COUNT: {wordCount}")
    hwPercentage = (hwCount * 100) / wordCount if wordCount > 0 else 0
    print (f"HW PERCENTAGE: {hwPercentage}")
    hasManuscript = hwPercentage > 5 and hwPercentage < 20 if hwPercentage > 0 else False
    if wordConfidence > docConfidence:
        text_corpus = text_corpus_words
        docConfidence = wordConfidence
    print (f"CONFIDENCE: {docConfidence}")
    return text_corpus, round(docConfidence,2), hasManuscript

# %%
