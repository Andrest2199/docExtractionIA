from pydantic_core import PydanticCustomError
from pydantic import BaseModel, field_validator


class OCRRequest(BaseModel):
    filename: str
    doc_type: str
    file_base64: str
    
    @field_validator("filename", "doc_type", "file_base64")
    @classmethod
    def check_not_empty(cls, value, field):
        if not value.strip():
            raise PydanticCustomError(
                "Value Error", f"El campo {field.field_name} no puede estar vacío"
            )
        if field.field_name == "doc_type" and value not in {"IMSS", "INFONAVIT", "SAT"}:
            raise PydanticCustomError(
                "Type error",
                f"Tipo de documento no reconocido. Por favor, proporciona un tipo valido: IMSS, INFONAVIT, SAT",
            )
        return value
