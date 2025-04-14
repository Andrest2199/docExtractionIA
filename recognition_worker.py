from utils.file_utils import FileUtils
from utils.general_utils import Utils
from vision_recognition import vision_entity_extraction
from chat_completion import chat_completions_entity_extraction
from base64 import b64decode
from document_handler import document_handler
import os
import re
import sys
import thresholds


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
image_raw_folder = os.path.join(os.getcwd(), "0_image_raw")
image_preprocessed_folder = os.path.join(os.getcwd(), "1_image_preprocessed")
text_extracted_folder = os.path.join(os.getcwd(), "3_text_extracted")

image_inject_folder = os.path.join(os.getcwd(), "image_inject")
data_inject_folder = os.path.join(os.getcwd(), "data_inject")
thresholds.length_threshold_calculator(data_inject_folder)

# TODO: Add logic for two paged documents


def recognition_worker(filename=str, doctype=str, file_base64=str) -> dict:
    """
    Main function to process the document and extract the information from it.
        Args:
            file_path: Path to the document to process.
            doctype: Type of document to process. It can be one of the following: IMSS, INFONAVIT, SAT.
            file_base64: The content of the file in base64.
        Returns:
            dict: A dictionary with the extracted information from the document.
    """
    # Borramos el URI del contenido base64
    data_url_pattern = re.compile(
        r"data:(application|image)/(jpeg|jpg|pdf|png);base64,"
    )
    if data_url_pattern.match(file_base64):
        file_base64 = data_url_pattern.sub("", file_base64)
    # Borramos todas los archivos de la carpeta 0 y 3
    FileUtils.delete_from_folder(image_raw_folder)
    # FileUtils.delete_from_folder(text_extracted_folder)
    
    # Decodeamos el contenido del documento
    image = b64decode(file_base64)
    # Guardamos el contenido del documento en la carpeta 0 para procesar la calidad del documento
    file_path = FileUtils.save(image_raw_folder + "/" + filename, image)

    # Mandamos al manejador de documentos para mejorar la calidad y generar la extraccion de datos
    text_extracted, docConfidence, hasManuscript, improved_image_path = document_handler(file_path, doctype)
    # return (text_extracted, docConfidence, hasManuscript)
    # TODO: Comentar esta linea
    # Guardamos el texto extraido en la carpeta 3
    FileUtils.save(text_extracted_folder + "/" + re.sub(r"\.(pdf|jpg|jpeg)$", ".txt", filename, flags=re.IGNORECASE), str(text_extracted))
    
    # Si el documento contiene multi-pagina
    if isinstance(text_extracted, list):
        errorMsg = None
        itemCount = 0
        # Contruimos JSON dictionary
        extraction = {
            "filename": os.path.basename(file_path),
            "doc_type": doctype,
            "num_pages": len(text_extracted),
            "pages": {}
        }
        for items in text_extracted:
            # Validar la calidad del texto extraído
            text_quality, message, process_type = raw_text_validator(items[0], doctype, items[1], items[2])
            print (text_quality,message,process_type)
            
            # Si el texto es inválido, anexar error
            if not text_quality:
                errorMsg = message
            
            if errorMsg == None:
                # Procesar con la función correspondiente
                if process_type == "vision_entity_extraction":
                    # fields_extracted, usage = ("hola", 2000)
                    fields_extracted, usage, content = vision_entity_extraction(
                        items[3], image_inject_folder, doctype
                    )
                elif process_type == "chat_completions_entity_extraction":
                    # fields_extracted, usage = ("hola", 2000)
                    fields_extracted, usage, content = chat_completions_entity_extraction(
                        items[0], data_inject_folder, doctype
                    )
                else:
                    errorMsg = "Método de procesamiento no valido."
                
                if errorMsg == None:
                    # Concatenamos a pages toda la informacion de la pagina actual
                    extraction["pages"][itemCount] = {"process_type": process_type,"values": fields_extracted,"usage": usage,"content": str(content)}
                    #Validamos algunos campos antes de regresar la información
                    isFatalError, validated_values = Utils.validate_fields(extraction["pages"][itemCount])
                    
                    if isFatalError:
                        extraction["pages"][itemCount]["values"] = {"detail" : validated_values}
                    else:
                        extraction["pages"][itemCount]["values"] = validated_values
                else:
                    extraction["pages"][itemCount] = {"detail": errorMsg}
            else:
                extraction["pages"][itemCount] = {"detail": errorMsg}
            itemCount += 1
        return extraction
            
    # Validar la calidad del texto extraído
    text_quality, message, process_type = raw_text_validator(text_extracted, doctype, docConfidence, hasManuscript)
    print (text_quality,message,process_type)
    
    # Si el texto es inválido, lanzar error
    if not text_quality:
        raise ValueError(message)
    
    # Procesar con la función correspondiente
    if process_type == "vision_entity_extraction":
        # fields_extracted, usage = ("hola", 2000)
        fields_extracted, usage, content = vision_entity_extraction(
            improved_image_path, image_inject_folder, doctype
        )
    elif process_type == "chat_completions_entity_extraction":
        # fields_extracted, usage = (text_extracted, 2000)
        fields_extracted, usage, content = chat_completions_entity_extraction(
            text_extracted, data_inject_folder, doctype
        )
    else:
        raise ValueError("Método de procesamiento no valido.")

    # Contruimos JSON dictionary
    extraction = {
        "filename": os.path.basename(file_path),
        "doc_type": doctype,
        "process_type": process_type,
        "values": fields_extracted,
        "usage": usage,
        "content": str(content),
    }
    
    #Validamos algunos campos antes de regresar la información
    isFatalError, validated_values = Utils.validate_fields(extraction)
    if isFatalError:
        raise ValueError(validated_values)
    
    extraction["values"] = validated_values
    return extraction

def raw_text_validator(text_extracted, doctype, docConfidence, hasManuscript = False):
    """
    Valida el texto extraído y determina la estrategia de extracción de entidades,
    priorizando la extracción basada en visión (Gemini).

    Args:
        text_extracted (str): El texto extraído por el OCR.
        doctype (str): Tipo de documento ('IMSS', 'INFONAVIT', 'SAT').
        docConfidence (float): Nivel de confianza del OCR (0-100).
        hasManuscript (bool): True si se detectó texto manuscrito (5%-20% del total).

    Returns:
        tuple: (isValid: bool, message: str, strategy: str | None)
               strategy puede ser 'vision_entity_extraction' o 'chat_completions_entity_extraction'.
    """
    # Umbral para definir high_threshold basado en la media
    THRESHOLD = 0.8 

    # Umbrales mínimos y "altos" por tipo de documento
    doc_thresholds = {
        "IMSS": (
            thresholds.IMSS_MIN_LENGTH - 100,
            thresholds.IMSS_MEAN_LENGTH * THRESHOLD,
        ),
        "INFONAVIT": (
            thresholds.INFONAVIT_MIN_LENGTH - 100,
            thresholds.INFONAVIT_MEAN_LENGTH * THRESHOLD,
        ),
        "SAT": (
            thresholds.SAT_MIN_LENGTH - 100,
            thresholds.SAT_MEAN_LENGTH * THRESHOLD,
        ),
        # Añadir otros tipos de documentos si es necesario
    }

    if doctype not in doc_thresholds:
        return (False, f"Tipo de documento '{doctype}' no reconozido o thresholds no definido.", None)

    low_threshold, high_threshold = doc_thresholds[doctype]

    # Rechazo por texto demasiado corto
    if len(text_extracted) < low_threshold:
        return (
            False,
            f"Texto extraido ({len(text_extracted)} caracteres) esta por debajo del mínimo threshold ({low_threshold}) para {doctype}. Por favor, intente con un documento de mejor calidad.",
            None,
        )

    # Usar Visión si la confianza es baja o si hay manuscrito.
    # Razón: Visión puede manejar mejor la incertidumbre del OCR bajo y el contexto espacial del manuscrito.
    # Ajustamos el umbral de confianza si es necesario (ej. < 90)
    CONFIDENCE_THRESHOLD_FOR_VISION = 90
    if docConfidence < CONFIDENCE_THRESHOLD_FOR_VISION or hasManuscript:
        message = f"Using vision entity extraction due to "
        reasons = []
        if docConfidence < CONFIDENCE_THRESHOLD_FOR_VISION:
            reasons.append(f"OCR confidence ({docConfidence:.2f} < {CONFIDENCE_THRESHOLD_FOR_VISION})")
        if hasManuscript:
            reasons.append("detected handwriting")
        message += " and ".join(reasons) + "."
        return (
            True,
            message,
            "vision_entity_extraction",
        )

    # Usar Chat Completions si la confianza es ALTA, NO hay manuscrito Y el texto es largo.
    # Razón: Este es el caso ideal para extracción basada en texto puro.
    # (docConfidence >= CONFIDENCE_THRESHOLD_FOR_VISION and not hasManuscript son implícitos por llegar aquí)
    if len(text_extracted) >= high_threshold:
        return (
            True,
            f"Using chat completions entity extraction: Good confidence ({docConfidence:.2f}), sufficient text length ({len(text_extracted)} >= {high_threshold}), and no handwriting detected.",
            "chat_completions_entity_extraction",
        )

    # (Default): Usar Visión si la confianza es ALTA, NO hay manuscrito, pero el texto NO es tan largo.
    # Razón: Aún preferimos visión, y la longitud menor al umbral 'high' no justifica cambiar.
    # (docConfidence >= CONFIDENCE_THRESHOLD_FOR_VISION and not hasManuscript and len(text_extracted) < high_threshold son implícitos)
    else:
        return (
            True,
            f"Using vision entity extraction: Good confidence ({docConfidence:.2f}), moderate text length ({len(text_extracted)} < {high_threshold}), and no handwriting detected.",
            "vision_entity_extraction",
        )
    # Si por alguna razón no entra en ninguna categoría, probabilidad nula
    return (False, "Unexpected state reached in text validation.", None)