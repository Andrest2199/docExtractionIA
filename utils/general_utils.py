#%%
import json
import re
import tiktoken
import base64
from unidecode import unidecode
from datetime import datetime

class Utils:
    @classmethod
    def to_dict(self, json_string=str) -> dict:
        """Converts a JSON string to a dictionary with two methods"""
        try:
            json_data = json.loads(str(json_string))
        except Exception as e:
            print(f"Failed to load JSON:{e}, trying to build dictionary...")
            try:
                json_data = self.build_dictionary(str(json_string))
            except Exception as e:
                json_data = json_string
                raise Exception(f"Error: {e}, fail attempting build dictionary...")
        return json_data
    
    @staticmethod
    def build_dictionary(json_string=str) -> dict:
        new_json = {}
        json_string = str(json_string)
        if json_string.startswith("{") and json_string.endswith("}"):
            json_string = json_string.replace("{", "").replace("}", "")
            json_string = json_string.replace("\n", "").replace("\t", "")
            json_string = json_string.split(",")
            for element in json_string:
                countColon = element.count(":")
                if countColon > 1:
                    element = element.replace(":", "", (countColon - 1))
                temp = element.strip().replace('"', "").split(":")

                if len(temp) > 1:
                    new_json[temp[0]] = (
                        None if temp[1].strip() == "NULL" else temp[1].strip()
                    )
                else:
                    new_json[temp[0]] = None
        else:
            raise ValueError("Input string is not a Valid JSON string")
        new_json = json.dumps(new_json, indent=4, sort_keys=True, ensure_ascii=False)
        return json.loads(new_json)

    @staticmethod
    def decode_text(texto):
        # Decodificamos caracteres UTF-8
        try:
            patron = re.compile(r"\\u([\d\w]{4})")
            final_text = patron.sub(
                lambda x: unidecode(chr(int(x.group(1), 16))), texto
            )
            return final_text
        except Exception as e:
            print(f"Error decoding text: {e}")
            return texto

    @staticmethod
    def read_file(file_path):
        with open(file_path) as file:
            return file.read()

    @staticmethod
    def num_tokens_from_string(string: str, model="gpt-4-vision-preview") -> int:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @classmethod
    def num_tokens_from_messages(
        self,
        messages,
        model="gpt-3.5-turbo-0125",
    ):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-0125",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-32k-0613",
            "gpt-4-vision-preview",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print(
                "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
            )
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print(
                "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
            )
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://platform.openai.com/docs/api-reference for more information of the model."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if not isinstance(value, dict):
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
                elif isinstance(value, dict):
                    for k, v in value.items():
                        num_tokens += len(encoding.encode(v))
                        if k == "name":
                            num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|im_start|>assistant<|im_sep|>
        return num_tokens

    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Error: The file path '{image_path}' was not found.")
        except PermissionError:
            print(
                f"Error: You do not have permissions to read the file in '{image_path}'."
            )
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

    def validate_fields(data) -> tuple:
        """
        Function to validate specific fields extracted before send them to client.
            Args:
                data: Extracted values via vision or chat completion process (dict).
            Returns:
                tuple: A tuple with the first arg being True|False and the second with the values|error.
        """
        try:
            if "values" in data.keys() and data["values"] != "" and data != "" and isinstance(data,dict):
                fields = data["values"]
            else:
                return True, "No se extrajo ningún campo, favor de validar documento."

            # Validacion de serie y folio
            if "serie_y_folio" in fields:
                fields["serie_y_folio"] = re.sub(" ","",fields["serie_y_folio"])
            
            # Validacion de ramo de seguro
            if "ramo_de_seguro" in fields:
                fields["ramo_de_seguro"] = fields["ramo_de_seguro"].str.lower()
                if fields["ramo_de_seguro"] == "enfermedad":
                    fields["ramo_de_seguro"] = "enfermedad general"
            
            # Validacion de NSS
            if "numero_de_seguridad_social" in fields:
                if len(fields["numero_de_seguridad_social"]) < 10 and len(fields["numero_de_seguridad_social"]) > 11:
                    fields["numero_de_seguridad_social"] = (
                            f"Error: El NSS '{fields['numero_de_seguridad_social']}' es incorrecto."
                        )
                    
            # Validacion de probable riesgo de trabajo
            if "probable_riesgo_trabajo" in fields:
                fields["probable_riesgo_trabajo"] = fields["probable_riesgo_trabajo"].str.lower()
            
            # Validacion de código postal
            if "codigo_postal" in fields:
                if fields["codigo_postal"] in ("","NA",None):
                    fields["codigo_postal"] = "Error: No se encontró el código postal"
                else:
                    if len(fields["codigo_postal"]) != 5:
                        fields["codigo_postal"] = (
                            f"Error: El código postal '{fields['codigo_postal']}' es incorrecto."
                        )

            # Validacion de CURP
            if "curp" in fields:
                if fields["curp"] in ("","NA",None):
                    fields["curp"] = "Error: No se encontró el CURP"
                else:
                    # Expresión regular para validar el formato de CURP
                    patron_CURP = re.compile(r"^[A-Z]{4}[0-9]{6}[HM][A-Z]{5}[0-9]{2}$")

                    # Validar el formato de CURP
                    if not patron_CURP.match(fields["curp"]):
                        fields["curp"] = f"Error: El CURP '{fields['curp']}' es incorrecto."

                    # Calcular el dígito verificador
                    if "Error" not in fields["curp"]:
                        suma = 0
                        diccionario_reemplazo = {
                            "A": "10",
                            "B": "11",
                            "C": "12",
                            "D": "13",
                            "E": "14",
                            "F": "15",
                            "G": "16",
                            "H": "17",
                            "I": "18",
                            "J": "19",
                            "K": "20",
                            "L": "21",
                            "M": "22",
                            "N": "23",
                            "Ñ": "24",
                            "O": "25",
                            "P": "26",
                            "Q": "27",
                            "R": "28",
                            "S": "29",
                            "T": "30",
                            "U": "31",
                            "V": "32",
                            "W": "33",
                            "X": "34",
                            "Y": "35",
                            "Z": "36",
                        }
                        for i in range(len(fields["curp"]) - 1):
                            if fields["curp"][i].isdigit():
                                suma += int(fields["curp"][i]) * (18 - i)
                            else:
                                suma += int(diccionario_reemplazo[fields["curp"][i]]) * (
                                    18 - i
                                )

                        digito_verificador = 10 - (suma % 10)
                        if digito_verificador == 10:
                            digito_verificador = 0

                        # Verificar el dígito verificador
                        if int(fields["curp"][-1]) != digito_verificador:
                            fields["curp"] = (
                                f"Error: El dígito verificador del CURP '{fields['curp']}' es incorrecto."
                            )

            # Validacion de RFC
            if "rfc" in fields:
                if fields["rfc"] in ("","NA",None):
                    fields["rfc"] = "Error: No se encontró el RFC"
                else:
                    # Expresión regular para validar el formato de RFC
                    patron_RFC = re.compile(r"^[A-ZÑ]{3,4}[0-9]{6}[A-V0-9]{2}[0-9A]$")

                    # Validar el formato de RFC
                    if not patron_RFC.match(fields["rfc"]):
                        fields["rfc"] = f"Error: El RFC '{fields['rfc']}' es incorrecto."

            # Validacion de Fechas
            for key in fields.keys():
                if "fecha" in key:
                    fecha = fields[key]
                    if fecha in ("","NA",None):
                        fields[key] = "Error: No existe fecha"
                    else:
                        patron_fecha = re.compile(r"\d{2}/\d{2}/\d{2}(\d{2})?$")
                        if not patron_fecha.match(fecha):
                            fields[key] = (
                                f"Error: El formato de la fecha '{fecha}' es incorrecto."
                            )
                        if "Error" not in fields[key]:
                            try:
                                # Dividir la fecha en día, mes y año
                                dia, mes, anio = map(int, fecha.split("/"))

                                # Verificar si el año es válido (en un rango razonable)
                                if len(str(anio)) == 2:
                                    anio_min = int(str(datetime.today().year)[2:4]) - 5
                                    anio_max = int(str(datetime.today().year)[2:4]) + 5
                                else:
                                    anio_min = int(str(datetime.today().year)) - 5
                                    anio_max = int(str(datetime.today().year)) + 5

                                if anio < anio_min or anio > anio_max:
                                    fields[key] = (
                                        f"Error: La fecha '{fecha}' no se encuentra dentro del rango más/menos 5 años."
                                    )

                                # Verificar si el mes está en el rango de 1 a 12
                                if "Error" not in fields[key]:
                                    if mes < 1 or mes > 12:
                                        fields[key] = (
                                            f"Error: El mes de la fecha '{fecha}' es invalido."
                                        )

                                    if "Error" not in fields[key]:
                                        # Verificar si el día está en el rango adecuado para cada mes
                                        dias_por_mes = [
                                            31,
                                            (
                                                28
                                                if anio % 4 != 0
                                                or (anio % 100 == 0 and anio % 400 != 0)
                                                else 29
                                            ),
                                            31,
                                            30,
                                            31,
                                            30,
                                            31,
                                            31,
                                            30,
                                            31,
                                            30,
                                            31,
                                        ]
                                        if dia < 1 or dia > dias_por_mes[mes - 1]:
                                            fields[key] = (
                                                f"Error: El día de la fecha '{fecha}' es invalido."
                                            )
                            except ValueError:
                                fields[key] = f"Error: La fecha '{fecha}' es invalida."
            
            #Validacion de None en todos los campos        
            for key in fields.keys():
                if fields[key] in ("","NA",None):
                    fields[key] = f"Error: No se encontró el {key}"
                    
            return False, fields
        
        except Exception as e:
            return True, str(e)
        

    # test={'filename': '05_Suspension Salvador de Jesus Alcala ok.pdf', 'doc_type': 'INFONAVIT', 'process_type': 'vision_entity_extraction', 'values': {'titulo': 'AVISO DE SUSPENSIÓN DE DESCUENTOS', 'motivo': 'SUSPENSION', 'folio': 'S1114023859630', 'fecha_notificacion': '29/09/2023', 'fecha_emision': None, 'fecha_tramite': None, 'fecha_recepcion': '05/10/2023', 'numero_de_credito': '6422018391', 'descuento': None, 'rfc': 'AACS960206LQ3', 'numero_de_seguridad_social': '54149652379', 'rfc_patron': 'LSM660818M98', 'numero_de_registro_patronal': 'Z2969597108', 'razon_social': 'LEVI STRAUSS DE MEXICO SA DE CV', 'sello_de_la_empresa': 'true', 'leyenda_aplicacion_descuento': 'a partir de la fecha en que se reciba este aviso, deberá suspender los descuentos que por concepto de amortización se vienen efectuando al trabajador'}, 'content': '["The next 8 images are examples of documents you will receive and next them the JSON\'s of the relevant information extracted in key pairs of each one, respectively.", [<PIL.JpegImagePlugin.JpegImageFile image mode=L size=1555x2081 at 0x11366EE90>, <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1546x2034 at 0x11366DD10>, <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1515x1917 at 0x1136A1220>, <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1582x2073 at 0x1136A1350>, <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1555x2081 at 0x1136A1480>, <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1492x2169 at 0x1136A15B0>, <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1505x1975 at 0x1136A09D0>, <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1474x1987 at 0x1136A0B00>], [\'{\\n  "titulo": "Acuse de notificación de inicio de trámite de crédito a través del Portal Empresarial del Infonavit",\\n  "motivo": "ALTA",\\n  "folio": "S1114023859630",\\n  "fecha_notificacion": "02/12/2023",\\n  "fecha_emision": null,\\n  "fecha_tramite": "01/11/23",\\n  "fecha_recepcion": "02/12/2023",\\n  "numero_de_credito": "*******601",\\n  "descuento": { "cantidad": "$2,511.29" },\\n  "rfc": null,\\n  "numero_de_seguridad_social": "68948003535",\\n  "rfc_patron": "LSM660818M98",\\n  "numero_de_registro_patronal": "B1014849108",\\n  "razon_social": "LEVI STRAUSS DE MEX SA CV",\\n  "sello_de_la_empresa": "true",\\n  "leyenda_aplicacion_descuento": null\\n}\', \'{\\n  "titulo": "Aviso para Retención de Descuentos",\\n  "motivo": "ALTA",\\n  "folio": "R0209025858838",\\n  "fecha_notificacion": "14/02/2025",\\n  "fecha_emision": "15/02/2025",\\n  "fecha_tramite": null,\\n  "fecha_recepcion": "18/02/2025",\\n  "numero_de_credito": "1425027102",\\n  "descuento": { "cantidad": "$4.349,19" },\\n  "rfc": "MAMJ971030R40",\\n  "numero_de_seguridad_social": "10169735270",\\n  "rfc_patron": "CGL1405078L1",\\n  "numero_de_registro_patronal": "Y5476751101",\\n  "razon_social": "CWL GLOBAL LOGISTICS MEXICO S DE RL DE C",\\n  "sello_de_la_empresa": "false",\\n  "leyenda_aplicacion_descuento": "A partir de la fecha del presente aviso"\\n}\', \'{\\n  "titulo": "AVISO PARA RETENCION DE DESCUENTOS",\\n  "motivo": "ALTA",\\n  "folio": "R0409023821693",\\n  "fecha_notificacion": "31/10/2023",\\n  "fecha_emision": null,\\n  "fecha_tramite": null,\\n  "fecha_recepcion": null,\\n  "numero_de_credito": "1505089794",\\n  "descuento": { "cantidad": "$2,936.80" },\\n  "rfc": "LOMC710925KQ6",\\n  "numero_de_seguridad_social": "20917104067",\\n  "rfc_patron": "LSM660818M98",\\n  "numero_de_registro_patronal": "B1014849108",\\n  "razon_social": "LEVI STRAUSS DE MEX SA CV",\\n  "sello_de_la_empresa": "false",\\n  "leyenda_aplicacion_descuento": "a partir del día siguiente a aquel en que se le haya notificado el presente aviso"\\n}\', \'{\\n  "titulo": "AVISO PARA RETENCION DE DESCUENTOS",\\n  "motivo": "ALTA",\\n  "folio": "R0414023383488",\\n  "fecha_notificacion": "10/10/2023",\\n  "fecha_emision": null,\\n  "fecha_tramite": null,\\n  "fecha_recepcion": null,\\n  "numero_de_credito": "2513074770",\\n  "descuento": { "factor": "22.2224" },\\n  "rfc": "RETS831223DB3",\\n  "numero_de_seguridad_social": "23048342077",\\n  "rfc_patron": "LSM660818M98",\\n  "numero_de_registro_patronal": "Z2969597108",\\n  "razon_social": "LEVI STRAUSS DE MEXICO SA DE CV",\\n  "sello_de_la_empresa": "false",\\n  "leyenda_aplicacion_descuento": "a partir del día siguiente a aquel en que se le haya notificado el presente aviso"\\n}\', \'{\\n  "titulo": "AVISO PARA RETENCION DE DESCUENTOS",\\n  "motivo": "ALTA",\\n  "folio": "212209338460",\\n  "fecha_notificacion": "15/09/2009",\\n  "fecha_emision": null,\\n  "fecha_tramite": null,\\n  "fecha_recepcion": "17/09/2009",\\n  "numero_de_credito": "2209106833",\\n  "descuento": { "factor": "25.931" },\\n  "rfc": "HEBA8105167RA",\\n  "numero_de_seguridad_social": "14988139698",\\n  "rfc_patron": "RVC861201Q99",\\n  "numero_de_registro_patronal": "E2382971109",\\n  "razon_social": "REFRESCOS VICTORIA DEL CENTRO S.A. DE C",\\n  "sello_de_la_empresa": "true",\\n  "leyenda_aplicacion_descuento": "a partir del día siguiente a aquel en que se le haya notificado el presente aviso"\\n}\', \'{\\n  "titulo": "AVISO DE SUSPENSION DE DESCUENTOS",\\n  "motivo": "SUSPENSION",\\n  "folio": "S1114023859630",\\n  "fecha_notificacion": "29/09/2023",\\n  "fecha_emision": null,\\n  "fecha_tramite": null,\\n  "fecha_recepcion": "05/10/2023",\\n  "numero_de_credito": "6422018391",\\n  "descuento": null,\\n  "rfc": "AACS960206LQ3",\\n  "numero_de_seguridad_social": "54149652379",\\n  "rfc_patron": "LSM660818M98",\\n  "numero_de_registro_patronal": "Z2969597108",\\n  "razon_social": "LEVI STRAUSS DE MEXICO SA DE CV",\\n  "sello_de_la_empresa": "true",\\n  "leyenda_aplicacion_descuento": "a partir de la fecha en que se reciba este aviso, deberá suspender los descuentos que por concepto de amortización se vienen efectuando al trabajador"\\n}\', \'{\\n  "titulo": "AVISO DE SUSPENSIÓN DE DESCUENTOS",\\n  "motivo": "SUSPENSION",\\n  "folio": "S1114023859630",\\n  "fecha_notificacion": "29/03/2023",\\n  "fecha_emision": null,\\n  "fecha_tramite": null,\\n  "fecha_recepcion": null,\\n  "numero_de_credito": "6422018391",\\n  "descuento": null,\\n  "rfc": "AACS960206LQ3",\\n  "numero_de_seguridad_social": "54149652379",\\n  "rfc_patron": "LSM660818M98",\\n  "numero_de_registro_patronal": "Z2969597108",\\n  "razon_social": "LEVI STRAUSS DE MEXICO SA DE CV",\\n  "sello_de_la_empresa": "false",\\n  "leyenda_aplicacion_descuento": "a partir de la fecha en que se reciba este aviso, deberá suspender los descuentos que por concepto de amotización se vienen efectuando al trabajador"\\n}\', \'{\\n  "titulo": "AVISO DE SUSPENSIÓN POR PROXIMA LIQUIDACION DE CREDITO",\\n  "motivo": "SUSPENSION",\\n  "folio": "S1001023582000",\\n  "fecha_notificacion": "05/12/2023",\\n  "fecha_emision": null,\\n  "fecha_tramite": null,\\n  "fecha_recepcion": "15/01/2024",\\n  "numero_de_credito": "0116041373",\\n  "descuento": null,\\n  "rfc": "SIHJ710316FV2",\\n  "numero_de_seguridad_social": "14947134616",\\n  "rfc_patron": "LSG1103226V6",\\n  "numero_de_registro_patronal": "A0172158105",\\n  "razon_social": "LEVI STRAUSS GLOBAL TRADING COMPANY II",\\n  "sello_de_la_empresa": "true",\\n  "leyenda_aplicacion_descuento": "a partir de la fecha en que se reciba este aviso, deberá suspender los descuentos, derivado a que el crédito está en la etapa final de cobro, motivo por el cual ya no deberá de retenerle la amortización ya que con el 5% de la aportación patronal durante un plazo establecido de 12 meses se terminará de pagar"\\n}\'], \'Your task is to parse the last image, recognize the entities to extract, and create a JSON with the relevant entities. Use the following format for the output JSON:\\n\\n{"titulo": "string", "motivo": "ALTA|SUSPENSION", "folio": "string", "fecha_notificacion": "date (DD/MM/YYYY)", "fecha_emision": "date (DD/MM/YYYY)", "fecha_tramite": "date (DD/MM/YYYY)", "fecha_recepcion": "date (DD/MM/YYYY)", "numero_de_credito": "string", "descuento": "oneOf": "[{"cantidad": "string"}, {"porcentaje": "string"}, {"factor": "string"}]", "rfc": "string", "numero_de_seguridad_social": "string", "rfc_patron": "string", "numero_de_registro_patronal": "string", "razon_social": "string", "sello_de_la_empresa": "true|false", "leyenda_aplicacion_descuento": "string"}\', <PIL.JpegImagePlugin.JpegImageFile image mode=L size=1474x1987 at 0x1136A0C30>]'}
    # validated_values = validate_fields(test)
    # test["values"] = validated_values
    # print(test)
# %%
