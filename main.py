from recognition_worker import recognition_worker
from fastapi import FastAPI, HTTPException
from models import OCRRequest

app = FastAPI()

@app.post("/ocr_recognize")
async def ocr_recognize(request: OCRRequest):
    """
    Endpoint to extract relevant field from a document.
        Returns:
            A JSON with the relevant information of the file, the extract fields and tokens usage.
    """
    try:
        data = recognition_worker(
            request.filename, request.doc_type, request.file_base64
        )
        return {"status": "success", "data": data}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
