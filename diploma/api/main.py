# api.py
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from starlette.responses import JSONResponse
from typing import Optional, List
from datetime import datetime
import uvicorn
from backend.service import AppService, CLASS_NAMES

app = FastAPI(title="🚀 GigaCardioAgent API v2.0", version="2.0")

service = None

@app.on_event("startup")
async def startup_event():
    global service
    service = AppService()
    print("✅ GigaCardioAgent API запущен!")

@app.get("/")
async def root():
    return {
        "message": "🚀 GigaCardioAgent API v2.0 ready!",
        "endpoints": {
            "health": "GET /health",
            "analyze_ecg": "POST /analyze_ecg (file + clinical)",
            "analyze_clinical": "POST /analyze_clinical (только текст)",
            "chat": "POST /chat (RAG чат)"
        },
        "device": service.config.device if service else "unknown",
        "model_loaded": service.model is not None if service else False,
        "classes": CLASS_NAMES
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": service.config.device if service else "cpu",
        "model_loaded": service.model is not None if service else False,
        "classes": CLASS_NAMES
    }

@app.post("/analyze_ecg")
async def analyze_ecg(
    ecg_file: UploadFile = File(...),
    clinical_info: str = Form(...)
):
    if not service:
        raise HTTPException(503, "Сервис не инициализирован")
    
    try:
        file_content = await ecg_file.read()
        
        result = await service.process_ecg(
            ecg_file_content=file_content,
            filename=ecg_file.filename,
            clinical_notes=clinical_info
        )
        
        if not result.get("success"):
            raise HTTPException(500, result.get("error", "Ошибка анализа"))
        
        return {
            "success": True,
            "diagnosis": result["diagnosis"],
            "confidence": result["confidence"],
            "structured_recommendation": result["structured_recommendation"],
            "full_cot_recommendation": result["full_cot_recommendation"],
            "rag_references": result["rag_references"],
            "clinical_info": clinical_info,
            "timestamp": result["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка анализа ЭКГ: {str(e)}")

@app.post("/analyze_clinical")
async def analyze_clinical(clinical_info: str = Form(...)):
    if not service:
        raise HTTPException(503, "Сервис не инициализирован")
    
    try:
        result = await service.analyze_clinical_only(clinical_info)
        
        return {
            "success": True,
            "mode": "clinical_only",
            "diagnosis": result.get("diagnosis", "Требуется ЭКГ для точного диагноза"),
            "confidence": result.get("confidence", 0.0),
            "clinical_info": clinical_info,
            "structured_recommendation": result.get("structured_recommendation", ""),
            "rag_references": result.get("rag_references", []),
            "formatted_sources": result.get("formatted_sources", []),
            "recommended_actions": result.get("recommended_actions", []),
            "requires_ecg": result.get("requires_ecg", True),
            "timestamp": result.get("timestamp", datetime.now().isoformat())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка клинического анализа: {str(e)}")

@app.post("/chat")
async def chat(
    message: str = Form(...),
    clinical_info: str = Form(""),
    diagnosis: str = Form(""),
    confidence: float = Form(0.0),
    rag_context: List[str] = Form([])
):
    if not service:
        raise HTTPException(503, "Сервис не инициализирован")
    
    try:
        response = await service.gigachat_client.chat_with_rag(
            diagnosis=diagnosis or "Неизвестно",
            clinical=clinical_info or message,
            confidence=confidence,
            rag_context=rag_context[:3] if rag_context else [],
            response_format="full_cot"
        )
        
        return {
            "success": True,
            "message": message,
            "response": response,
            "context": {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "clinical_info": clinical_info
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Ошибка чата: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )

