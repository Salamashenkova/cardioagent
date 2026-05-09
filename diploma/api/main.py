# diploma/api/main.py
import sys
from pathlib import Path

print("=== main.py: НАЧАЛО ЗАГРУЗКИ ===")
print(f"1. Текущая директория: {Path.cwd()}")
print(f"2. Путь к файлу: {Path(__file__)}")
print(f"3. Родительская директория: {Path(__file__).parent.parent}")

sys.path.append(str(Path(__file__).parent.parent))
print(f"4. sys.path после добавления: {sys.path}")

print("5. Импортируем fastapi и другие модули...")
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from starlette.responses import JSONResponse
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
import json
print("6. ✅ Базовые модули импортированы")

print("7. Импортируем backend.service...")
try:
    from backend.service import AppService, CLASS_NAMES
    print("8. ✅ AppService и CLASS_NAMES импортированы успешно!")
    print(f"9. CLASS_NAMES = {CLASS_NAMES}")
except Exception as e:
    print(f"8. ❌ Ошибка импорта backend.service: {e}")
    import traceback
    traceback.print_exc()
    raise

print("10. Создаём FastAPI приложение...")
app = FastAPI(
    title="🚀 GigaCardioAgent API v2.0", 
    version="2.0", 
    root_path="/"
)
print("11. ✅ FastAPI приложение создано")

service = None
print("12. Переменная service инициализирована как None")

print("13. Регистрируем startup_event...")
@app.on_event("startup")
async def startup_event():
    global service
    print("=== STARTUP_EVENT: НАЧАЛО ===")
    print("14. Создаём экземпляр AppService...")
    try:
        service = AppService()
        print("15. ✅ AppService успешно создан!")
        print(f"16. service.config.device = {service.config.device}")
        print(f"17. service.model is None? {service.model is None}")
        print(f"18. service.rag is None? {service.rag is None}")
        print(f"19. service.gigachat_client is None? {service.gigachat_client is None}")
    except Exception as e:
        print(f"15. ❌ Ошибка при создании AppService: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("=== ВСЕ ЗАРЕГИСТРИРОВАННЫЕ МАРШРУТЫ ===")
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        if methods:
            print(f"  {methods} {route.path}")
        else:
            print(f"  {route.path}")
    print("====================================")
    print("✅ GigaCardioAgent API запущен!")
    print("=== STARTUP_EVENT: КОНЕЦ ===")

print("20. Регистрируем корневой эндпоинт...")
@app.get("/")
async def root():
    print("*** ВЫЗВАН КОРНЕВОЙ ЭНДПОИНТ ***")
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
print("21. ✅ Корневой эндпоинт зарегистрирован")

print("22. Регистрируем health эндпоинт...")
@app.get("/health")
async def health():
    print("*** ВЫЗВАН HEALTH ЭНДПОИНТ ***")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": service.config.device if service else "cpu",
        "model_loaded": service.model is not None if service else False,
        "classes": CLASS_NAMES
    }
print("23. ✅ Health эндпоинт зарегистрирован")

print("24. Регистрируем analyze_ecg эндпоинт...")
@app.post("/analyze_ecg")
async def analyze_ecg(
    ecg_file: UploadFile = File(...),
    clinical_info: str = Form(...)
):
    print(f"*** ВЫЗВАН analyze_ecg: clinical_info={clinical_info[:50] if clinical_info else 'empty'}... ***")
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
            "rag_confidence": result.get("rag_confidence", 0.0),
            "top3_predictions": result.get("top3_predictions", []),
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
print("25. ✅ analyze_ecg эндпоинт зарегистрирован")

print("26. Регистрируем analyze_clinical эндпоинт...")
@app.post("/analyze_clinical")
async def analyze_clinical(clinical_info: str = Form(...)):
    print(f"*** ВЫЗВАН analyze_clinical: clinical_info={clinical_info[:50] if clinical_info else 'empty'}... ***")
    if not service:
        raise HTTPException(503, "Сервис не инициализирован")
    
    try:
        result = await service.analyze_clinical_only(clinical_info)
        
        return {
            "success": True,
            "mode": "clinical_only",
            "diagnosis": result.get("diagnosis", "Требуется ЭКГ для точного диагноза"),
            "confidence": result.get("confidence", 0.0),
            "rag_confidence": result.get("rag_confidence", 0.0),
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
print("27. ✅ analyze_clinical эндпоинт зарегистрирован")

print("28. Регистрируем chat эндпоинт...")
@app.post("/chat")
async def chat(
    message: str = Form(...),
    diagnosis: str = Form(""),
    confidence: float = Form(0.0),
    clinical_info: str = Form(""),
    rag_context: str = Form("[]")  # Изменено на строку, т.к. Form не поддерживает List напрямую
):
    print(f"*** ВЫЗВАН chat: message={message[:50] if message else 'empty'}... ***")
    print(f"    diagnosis={diagnosis}, confidence={confidence}")
    
    if not service:
        raise HTTPException(503, "Сервис не инициализирован")
    
    try:
        # Парсим rag_context из строки
        try:
            if rag_context and rag_context != "null" and rag_context != "[]":
                previous_rag = json.loads(rag_context)
            else:
                previous_rag = None
        except json.JSONDecodeError:
            print(f"   ⚠️ Ошибка парсинга rag_context: {rag_context[:100]}")
            previous_rag = None
        
        # Используем новый метод чата из service
        result = await service.chat_with_assistant(
            message=message,
            diagnosis=diagnosis if diagnosis else "Неизвестно",
            confidence=confidence,
            clinical_info=clinical_info,
            previous_rag_context=previous_rag
        )
        
        if result.get("success"):
            # Форматируем источники для ответа
            rag_references = []
            for doc in result.get("rag_references", []):
                if isinstance(doc, dict):
                    formatted_ref = f"**{doc.get('title', 'Источник')}** (релевантность: {doc.get('relevance', 0):.1%})\n{doc.get('content', '')[:500]}..."
                    rag_references.append(formatted_ref)
                else:
                    rag_references.append(str(doc))
            
            return {
                "success": True,
                "response": result.get("response", ""),
                "rag_references": rag_references,
                "rag_confidence": result.get("rag_confidence", 0.0),
                "context": {
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "clinical_info": clinical_info[:200] if clinical_info else ""
                }
            }
        else:
            return {
                "success": False,
                "response": result.get("response", "Извините, произошла ошибка"),
                "rag_references": [],
                "rag_confidence": 0.0,
                "error": result.get("error", "Unknown error")
            }
        
    except Exception as e:
        print(f"   ❌ Ошибка чата: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка чата: {str(e)}")
print("29. ✅ chat эндпоинт зарегистрирован")

print("30. Регистрируем обработчик исключений...")
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )
print("31. ✅ Обработчик исключений зарегистрирован")

print("=== main.py: КОНЕЦ ЗАГРУЗКИ ===")



