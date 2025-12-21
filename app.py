from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from src.pipeline import recognize_plate
from src.io_utils import save_image
import tempfile
import os

app = FastAPI(
    title="License Plate Recognition API",
    description="API for recognizing license plates from images",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "License Plate Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/recognize": "POST - Upload image to recognize license plate",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "license_plate_recognition"}

@app.post("/recognize")
async def recognize_plate_endpoint(file: UploadFile = File(...)):
    """
    Recognize license plate from uploaded image
    
    - **file**: Image file (JPEG, PNG, etc.)
    
    Returns:
    - **plate_text**: Recognized license plate text
    - **status**: Success or error message
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please ensure it's a valid image file."
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, image)
        
        try:
            # Recognize plate
            plate_text = recognize_plate(tmp_path)
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "plate_text": plate_text,
                    "filename": file.filename
                }
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Failed to recognize license plate"
            }
        )

@app.post("/recognize/base64")
async def recognize_plate_base64(image_data: dict):
    """
    Recognize license plate from base64 encoded image
    
    Request body:
    - **image**: Base64 encoded image string
    """
    try:
        import base64
        
        if 'image' not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Decode base64
        image_bytes = base64.b64decode(image_data['image'])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, image)
        
        try:
            plate_text = recognize_plate(tmp_path)
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "plate_text": plate_text
                }
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)