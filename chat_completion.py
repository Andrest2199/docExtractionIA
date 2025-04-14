# %% Open AI Chat Completions

from openai import OpenAI
from dotenv import load_dotenv
from utils.file_utils import FileUtils
from utils.general_utils import Utils
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
# OpenAI API Key
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")


def chat_completions_entity_extraction(
    extracted_text, data_inject_folder, type_doc
) -> dict:
    """
    -> text_extracted -> result

    Receives the extracted text to process and returns a JSON with the relevant fields.
    This function use gtp-4o-mini from Open AI.

        Args:
            extracted_text: The extracted text from the new file to process.
            data_inject_folder: The folder path of the context data for the system.
            type_doc: The folder path of the context data that is going to be pass to the system.

        Return:
            A structured JSON with the relevant fields extracted.
    """
    # Set Data for system from folder 'Data inject'
    context_data_inyection = ""
    input_results_list = []
    input_txt_list = []
    txt_count = 0
    results_count = 0
    all_data_inject_files = []
    # Retrieve files from 'Data Inject'
    if type_doc == "IMSS":
        data_inject_sub_folder = os.path.join(data_inject_folder, "IMSS")
        all_data_inject_files = FileUtils.get_paths(data_inject_sub_folder, 2)
    elif type_doc == "INFONAVIT":
        data_inject_sub_folder = os.path.join(data_inject_folder, "INFONAVIT")
        all_data_inject_files = FileUtils.get_paths(data_inject_sub_folder, 2)
    elif type_doc == "SAT":
        data_inject_sub_folder = os.path.join(data_inject_folder, "SAT")
        all_data_inject_files = FileUtils.get_paths(data_inject_sub_folder, 1)
    else:
        raise ValueError(
            "Tipo de documento no reconozido. Por favor, proporcione un tipo de documento válido: IMSS, INFONAVIT, SAT"
        )

    for file in all_data_inject_files:
        file_name = os.path.basename(file)
        if file_name.startswith("result"):
            input_results_list.append(FileUtils.read(file))
            results_count += 1
        elif file_name.startswith("data"):
            input_txt_list.append(FileUtils.read(file))
            txt_count += 1
        else:
            print(f"File {file} not recognized")

    # Set context data
    context_data_inyection = f"Your role is to extract relevant information from raw text. In between XML tags you will find {txt_count} examples of raw text inputs and information extracted outputs with the relevant entities to be recognized. \n\n"

    txt_count = 1
    results_count = 1
    for input_txt, input_result in zip(input_txt_list, input_results_list):
        context_data_inyection += f"<raw_text_input_example_{txt_count}>\n\n{input_txt}\n\n</raw_text_input_example_{txt_count}>\n\n<information_extracted_output_example_{results_count}>\n\n{input_result}\n\n</information_extracted_output_example_{results_count}>\n\n"
        txt_count += 1
        results_count += 1

    context_data_inyection += "\nYou will recive a new raw text by the user. Your task is to analyse the raw text, recognize the entities to be extracted, and create a JSON with the relevant entities. Use the following format for the output JSON:\n\n"
    if type_doc == "IMSS":
        context_data_inyection += '{"serie_y_folio": "string", "tipo_incapacidad": "string", "ramo_de_seguro": "string", "probable_riesgo_trabajo": "string", "dias_autorizados": "string", "fecha_a_partir": "date (DD/MM/YYYY)", "fecha_expedido": "date (DD/MM/YYYY)", "numero_de_seguridad_social": "string", "curp": "string", "nombre_del_asegurado": "string", "clave_patronal": "string", "nombre_del_patron": "string"}'

    elif type_doc == "INFONAVIT":
        context_data_inyection += '{"titulo": "string", "motivo": "ALTA|SUSPENSION", "folio": "string", "fecha_notificacion": "date (DD/MM/YYYY)", "fecha_emision": "date (DD/MM/YYYY)", "fecha_tramite": "date (DD/MM/YYYY)", "fecha_recepcion": "date (DD/MM/YYYY)", "numero_de_credito": "string", "descuento": "oneOf": "[{"cantidad": "string"}, {"porcentaje": "string"}, {"factor": "string"}]", "rfc": "string", "numero_de_seguridad_social": "string", "rfc_patron": "string", "numero_de_registro_patronal": "string", "razon_social": "string", "sello_de_la_empresa": "true|false", "leyenda_aplicacion_descuento": "string"}'

    elif type_doc == "SAT":
        context_data_inyection += '{"codigo_postal": "number", "curp": "string", "nombres": "string", "primer_apellido": "string", "segundo_apellido": "string", "rfc": "string", "estatus_en_el_padron": "string"}'
    else:
        raise ValueError(
            "Tipo de documento no reconozido. Por favor, proporcione un tipo de documento válido: IMSS, INFONAVIT, SAT"
        )
    # Set system role
    system_content = {"role": "developer", "content": context_data_inyection}

    user_content = {"role": "user", "content": extracted_text}

    # Create instance of openAI client
    client = OpenAI(api_key=OPEN_AI_API_KEY)

    # Get response
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # gpt-4-0125-preview, #gpt-4-vision-preview , #gpt-4-turbo-preview
        messages=[
            system_content,
            user_content,
        ],
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
    )

    # Extract json content from response
    json_string = response.choices[0].message.content
    json_string = json_string.replace("```json\n", "").replace("\n```", "")
    print(type(json_string))
    print(json_string)

    # Return json data
    json_data = Utils.to_dict(json_string)
    print("-------------------")
    print(json_data)

    tokens_count_by_gpt = response.usage
    content = str(system_content)+str(user_content)
    return json_data, tokens_count_by_gpt, content
