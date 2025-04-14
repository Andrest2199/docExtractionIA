# %% Google Gemini 2.0 Flash Lite Vision Recognition

from google import genai
from google.genai import types
from utils.file_utils import FileUtils
from utils.general_utils import Utils
from dotenv import load_dotenv
import sys
import os
import PIL.Image
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def vision_entity_extraction(image_path, image_inject_folder, type_doc) -> tuple:
    """
    -> image_path -> result
    
    Receives the image to process and returns a JSON with the relevant fields.
    This function use Gemini 2.0 Flash Lite from Google.
    
        Args:
            image_path: The image file path to process.
            image_inject_folder: The folder path of the context data for the system.
            type_doc: The folder path of the context data that is going to be pass to the system.
            
        Return:
            A structured JSON with the relevant fields extracted.
    """
    # Define variables
    input_results_list = []
    input_image_list = []
    image_count = 0
    results_count = 0
    all_image_inject_files = []
    # Retrieve files from 'Data Inject'
    if type_doc == "IMSS":
        data_inject_sub_folder = os.path.join(image_inject_folder, "IMSS")
        all_image_inject_files = FileUtils.get_paths(data_inject_sub_folder, 2)
    elif type_doc == "INFONAVIT":
        data_inject_sub_folder = os.path.join(image_inject_folder, "INFONAVIT")
        all_image_inject_files = FileUtils.get_paths(data_inject_sub_folder, 2)
    elif type_doc == "SAT":
        data_inject_sub_folder = os.path.join(image_inject_folder, "SAT")
        all_image_inject_files = FileUtils.get_paths(data_inject_sub_folder, 1)
    else:
        raise ValueError(
            "Tipo de documento no reconozido. Por favor, proporcione un tipo de documento válido: IMSS, INFONAVIT, SAT"
        )

    for file in all_image_inject_files:
        file_name = os.path.basename(file)
        if file_name.startswith("result"):
            input_results_list.append(FileUtils.read(file))
            results_count += 1
        elif file_name.startswith("image"):
            image = PIL.Image.open(file)
            input_image_list.append(image)
            image_count += 1
        else:
            print(f"Documento: {file} no reconozido.")
    
    # Set context data
    first_line_context = f"The next {image_count} images are examples of documents you will receive and next them the JSON's of the relevant information extracted in key pairs of each one, respectively."
    last_line_context = f"Your task is to parse the last image, recognize the entities to extract, and create a JSON with the relevant entities. Use the following format for the output JSON:\n\n"
    if type_doc == "IMSS":
        last_line_context += '{"serie_y_folio": "string", "tipo_incapacidad": "string", "ramo_de_seguro": "string", "probable_riesgo_trabajo": "string", "dias_autorizados": "string", "fecha_a_partir": "date (DD/MM/YYYY)", "fecha_expedido": "date (DD/MM/YYYY)", "numero_de_seguridad_social": "string", "curp": "string", "nombre_del_asegurado": "string", "clave_patronal": "string", "nombre_del_patron": "string"}'

    elif type_doc == "INFONAVIT":
        last_line_context += '{"titulo": "string", "motivo": "ALTA|SUSPENSION", "folio": "string", "fecha_notificacion": "date (DD/MM/YYYY)", "fecha_emision": "date (DD/MM/YYYY)", "fecha_tramite": "date (DD/MM/YYYY)", "fecha_recepcion": "date (DD/MM/YYYY)", "numero_de_credito": "string", "descuento": "oneOf": "[{"cantidad": "string"}, {"porcentaje": "string"}, {"factor": "string"}]", "rfc": "string", "numero_de_seguridad_social": "string", "rfc_patron": "string", "numero_de_registro_patronal": "string", "razon_social": "string", "sello_de_la_empresa": "true|false", "leyenda_aplicacion_descuento": "string"}'

    elif type_doc == "SAT":
        last_line_context += '{"codigo_postal": "number", "curp": "string", "nombres": "string", "primer_apellido": "string", "segundo_apellido": "string", "rfc": "string", "estatus_en_el_padron": "string"}'
    else:
        raise ValueError(
            "Tipo de documento no reconozido. Por favor, proporcione un tipo de documento válido: IMSS, INFONAVIT, SAT"
        )

    # Set content data
    content = []
    content.append(first_line_context)
    content.append(input_image_list)
    content.append(input_results_list)
    content.append(last_line_context)
    content.append(PIL.Image.open(image_path))
    
    
    # return content,200,1
# image_path="/Users/andrestobar/Desktop/GRUPO ONO/IA/document_extraction/2_image_improved/CSF HECTOR DE JESUS SANCHEZ MENDOZA page0.jpeg"
# image_inject_folder = os.path.join(os.getcwd(), "image_inject")
# type_doc="SAT"
# vision_entity_extraction(image_path, image_inject_folder, type_doc)

    client = genai.Client(api_key=GEMINI_API_KEY)
#     response = client.models.count_tokens(
#         model="gemini-2.0-flash-lite",
#         contents=content,
#     )
#     print("Tokens Des:",response)
#     print("Prompt tokens:",response.total_tokens)
#     total_tokens= response.total_tokens
#     return "hola",total_tokens
# image_path="/Users/andrestobar/Desktop/GRUPO ONO/IA/document_extraction/2_image_improved/CSF HECTOR DE JESUS SANCHEZ MENDOZA page0.jpeg"
# image_inject_folder = os.path.join(os.getcwd(), "image_inject")
# type_doc="SAT"
# vision_entity_extraction(image_path, image_inject_folder, type_doc)
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=content,
    )
    
    # Extract json content from response
    json_string = response.text
    json_string = json_string.replace("```json\n", "").replace("\n```", "")
    print(type(json_string))
    print(json_string)
    # Return json data
    json_data = Utils.to_dict(json_string)
    print("------------------")
    print(json_data)
    return json_data,response.usage_metadata,content


# # %%improved pdf to images
#     from image_pre_procesing import (
#         pdf_has_text,
#         get_text_from_pdf,
#         pdf_to_image,
#         process_images,
#     )
#     from utils.file_utils import FileUtils
#     from improve_image_quality import improve_image_quality
#     import os
#     input_results_list = []
#     input_image_list = []
#     image_count = 0
#     results_count = 0
#     all_image_inject_files = []

#     image_inject_folder = os.path.join(os.getcwd(), "image_inject")
#     data_inject_sub_folder = os.path.join(image_inject_folder, "SAT")
#     all_image_inject_files = FileUtils.get_paths(data_inject_sub_folder, 1)
#     # return all_image_inject_files, len(all_image_inject_files)
#     for file in all_image_inject_files:
#         file_name = os.path.basename(file)
#         if file_name.startswith("result"):
#             input_results_list.append(FileUtils.read(file))
#             results_count += 1
#         elif file_name.startswith("image"):
#             input_image_list.append(file)
#             image_count += 1
#         else:
#             print(f"Documento: {file} no reconozido.")

#     image_preprocessed_folder = os.path.join(os.getcwd(), "1_image_preprocessed")
#     image_improved_folder = os.path.join(os.getcwd(), "2_image_improved")

#     for image in input_image_list:
#         images_list = pdf_to_image(image, image_preprocessed_folder)
#         # image_path = os.path.join(image_preprocessed_folder, image)
#         # improve_image_quality(image_path, image_improved_folder)
#     for image in images_list:
#         image_path = os.path.join(image_preprocessed_folder, image)
#         improve_image_quality(image_path, image_improved_folder)

# %%
