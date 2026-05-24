print("=== backend/service.py: НАЧАЛО ЗАГРУЗКИ ===")
print("1. Импортируем базовые модули...")

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
print(f"2. sys.path после добавления: {sys.path}")

print("3. Импортируем urllib3...")
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("4. Импортируем scipy.io...")
import scipy.io as sio
import io
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import datetime
print("5. Импортируем torch...")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import re
from collections import Counter, deque
import math
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as qmodels
from gigachat import GigaChat
from gigachat.models import Chat
import warnings
warnings.filterwarnings("ignore")

print("6. Загружаем dotenv...")
load_dotenv()
print("7. ✅ Базовые модули импортированы")

print("8. Определяем класс Config...")
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/ProECGNetbeststtc.pth"))
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "kr_production_cloud")
    gigachat_credentials: str = os.getenv("GIGACHAT_CREDENTIALS")
    verify_ssl: bool = os.getenv("VERIFY_SSL", "false").lower() == "true"
    chat_memory_len: int = int(os.getenv("CHAT_MEMORY_LEN", "10"))

    def __post_init__(self):
        if not self.qdrant_url or not self.qdrant_api_key:
            print("⚠️ ВНИМАНИЕ: Qdrant не настроен! Добавьте переменные QDRANT_URL и QDRANT_API_KEY в окружение.")
        if not self.gigachat_credentials:
            print("⚠️ ВНИМАНИЕ: GigaChat не настроен! Добавьте переменную GIGACHAT_CREDENTIALS в окружение.")

    def get_model_path(self):
        print("   get_model_path: ищем модель...")
        if self.model_path.exists():
            print(f"   ✅ Модель найдена по пути: {self.model_path}")
            return self.model_path

        api_relative = Path("../") / self.model_path
        if api_relative.exists():
            print(f"   ✅ Модель найдена по пути: {api_relative}")
            return api_relative

        root_relative = Path(".") / self.model_path
        if root_relative.exists():
            print(f"   ✅ Модель найдена по пути: {root_relative}")
            return root_relative

        models_dir = Path("models")
        if models_dir.exists():
            pth_files = list(models_dir.glob("*.pth"))
            if pth_files:
                print(f"   ✅ Найден файл модели: {pth_files[0]}")
                return pth_files[0]

        print(f"   ⚠️ Модель не найдена по путям: {self.model_path}, {api_relative}, {root_relative}")
        return self.model_path

print("9. Создаём экземпляр Config...")
config = Config()
print(f"10. Config создан: device={config.device}, model_path={config.model_path}")

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
print(f"11. CLASS_NAMES = {CLASS_NAMES}")
print(f"12. Qdrant URL: {bool(config.qdrant_url)}, GigaChat credentials: {bool(config.gigachat_credentials)}")
print(f"🚀 HybridRAG Pro v2.0 | Device: {config.device} | Qdrant: {bool(config.qdrant_url)} | GigaChat: {bool(config.gigachat_credentials)}")

print("13. Определяем класс BM25...")
class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.doc_freqs, self.idf, self.doc_lens, self.avgdl = self._precompute(docs)

    def _precompute(self, docs):
        doc_freqs = [Counter(re.findall(r'\w+', doc.lower())) for doc in docs]
        all_terms = set(term for freqs in doc_freqs for term in freqs)
        df = {term: sum(1 for freqs in doc_freqs if term in freqs) for term in all_terms}
        N = len(docs)
        idf = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in all_terms}
        doc_lens = [sum(freqs.values()) for freqs in doc_freqs]
        avgdl = np.mean(doc_lens) if doc_lens else 1.0
        return doc_freqs, idf, doc_lens, avgdl

    def scores(self, query: str, doc_indices: List[int] = None) -> List[float]:
        if doc_indices is None:
            doc_indices = list(range(len(self.doc_freqs)))
        query_terms = re.findall(r'\w+', query.lower())
        raw_scores = []
        for idx in doc_indices:
            score = 0.0
            doc_len = self.doc_lens[idx]
            for term in set(query_terms):
                tf = self.doc_freqs[idx].get(term, 0)
                if tf > 0 and term in self.idf:
                    bm25_term = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    score += bm25_term * self.idf.get(term, 0)
            raw_scores.append(score)
        if not raw_scores:
            return []
        min_score, max_score = min(raw_scores), max(raw_scores)
        if max_score > min_score:
            return [(s - min_score) / (max_score - min_score) for s in raw_scores]
        return [0.5] * len(raw_scores)

print("14. ✅ Класс BM25 определён")

print("15. Определяем класс QdrantRAG...")
class QdrantRAG:
    def __init__(self, config):
        print("   QdrantRAG.__init__: начало")
        self.config = config
        if config.qdrant_url and config.qdrant_api_key:
            try:
                print(f"   Подключаемся к Qdrant: {config.qdrant_url}")
                self.client = QdrantClient(
                    url=config.qdrant_url,
                    api_key=config.qdrant_api_key,
                    timeout=60.0
                )
                print("   ✅ Qdrant подключен")
            except Exception as e:
                self.client = None
                print(f"   ⚠️ Ошибка подключения к Qdrant: {e}")
        else:
            self.client = None
            print("   ⚠️ Qdrant недоступен")
        self._embedding_model = None
        print("   QdrantRAG.__init__: завершён")

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            print("   🚀 Ленивая загрузка: SentenceTransformer...")
            self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("   ✅ SentenceTransformer загружен")
        return self._embedding_model

    def _deduplicate_documents(self, hits):
        unique_by_content = {}
        for hit in hits:
            title = hit.payload.get('title', 'Документ без названия')
            text = hit.payload.get('text', '')
            score = getattr(hit, "score", 0.0)
            key = text[:200].strip().lower()
            if key not in unique_by_content or score > unique_by_content[key]["score"]:
                unique_by_content[key] = {
                    "title": title,
                    "content": text,
                    "score": score,
                    "source": hit.payload.get("source", "База знаний")
                }
        return list(unique_by_content.values())

    async def search_similar(self, diagnosis: str, clinical: str, limit: int = 10) -> Tuple[List[str], float]:
        if not self.client:
            return ["Qdrant недоступен - используем общие рекомендации"], 0.0

        query_text = f"{diagnosis} {clinical}".strip() if diagnosis else clinical
        print(f"   🔍 Поиск в базе знаний: {query_text[:100]}...")

        try:
            query_vec = self.embedding_model.encode(query_text).tolist()
            hits = self.client.query_points(
                collection_name=self.config.qdrant_collection,
                query=query_vec,
                limit=limit * 2,
                score_threshold=0.65
            )

            deduped = self._deduplicate_documents(hits.points)
            if not deduped:
                return ["Не найдено релевантных документов в базе знаний"], 0.0

            sorted_results = sorted(deduped, key=lambda x: x["score"], reverse=True)[:limit]
            results = [
                f"**{item['title']}** (релевантность: {item['score']:.2f})\n{item['content'][:500]}..."
                for item in sorted_results
            ]
            avg_rag_confidence = float(np.mean([item["score"] for item in sorted_results])) if sorted_results else 0.0
            print(f"   📊 Найдено источников: {len(results)}, средняя уверенность RAG: {avg_rag_confidence:.3f}")
            return results, avg_rag_confidence

        except Exception as e:
            print(f"   Qdrant search error: {e}")
            return ["Ошибка поиска в базе знаний. Используем общие рекомендации."], 0.0

    async def search_for_chat(self, query: str, limit: int = 5) -> Tuple[List[Dict], float]:
        if not self.client:
            return [], 0.0

        print(f"   🔍 Чат-поиск в базе знаний: {query[:100]}...")

        try:
            query_vec = self.embedding_model.encode(query).tolist()
            hits = self.client.query_points(
                collection_name=self.config.qdrant_collection,
                query=query_vec,
                limit=limit * 2,
                score_threshold=0.65
            )

            deduped = self._deduplicate_documents(hits.points)
            if not deduped:
                return [], 0.0

            sorted_results = sorted(deduped, key=lambda x: x["score"], reverse=True)[:limit]
            avg_relevance = float(np.mean([item["score"] for item in sorted_results])) if sorted_results else 0.0
            print(f"   📊 Для чата найдено источников: {len(sorted_results)}")
            return sorted_results, avg_relevance

        except Exception as e:
            print(f"   Чат-поиск error: {e}")
            return [], 0.0

print("16. ✅ Класс QdrantRAG определён")

print("17. Определяем класс GigaChatClient...")
class GigaChatClient:
    def __init__(self, config):
        print("   GigaChatClient.__init__: начало")
        self.config = config
        self._gigachat = None
        print("   GigaChatClient.__init__: завершён")

    @property
    def gigachat(self):
        if self._gigachat is None and self.config.gigachat_credentials:
            print("   🚀 Ленивая загрузка: GigaChat...")
            self._gigachat = GigaChat(
                credentials=self.config.gigachat_credentials,
                scope="GIGACHAT_API_PERS",
                verify_ssl_certs=self.config.verify_ssl,
                model="GigaChat-Pro"
            )
            print("   ✅ GigaChat инициализирован")
        return self._gigachat

    async def chat_with_rag(self, diagnosis: str, clinical: str, confidence: float,
                            rag_context: List[str], response_format: str = "structured") -> str:
        if not self.gigachat:
            return "🚨 GigaChat недоступен"

        context = "\n═══\n".join(rag_context[:3])
        if response_format == "structured":
            prompt = f"""Пациент: {clinical}
ЭКГ: {diagnosis} (доверие: {confidence:.0%})

📚 Контекст из базы знаний:
{context}

✅ КЛИНИЧЕСКАЯ РЕКОМЕНДАЦИЯ:"""
        else:
            prompt = f"""Пациент: {clinical}
ЭКГ: {diagnosis} (доверие: {confidence:.0%})

📚 Контекст из базы знаний:
{context}

Дай четкие клинические рекомендации."""

        for attempt in range(3):
            try:
                chunks = []
                resp = self.gigachat.stream(Chat(messages=[{"role": "user", "content": prompt}]))
                for chunk in resp:
                    delta = chunk.choices[0].delta.content or ""
                    chunks.append(delta)
                return "".join(chunks).strip()
            except Exception as e:
                print(f"   GigaChat retry {attempt+1}: {e}")
                await asyncio.sleep(2 ** attempt)
        return "🚨 GIGA OFFLINE: АСА 160мг + ЭКГ повтор + кардиолог ОЧНО"

    async def chat_answer(self, message: str, diagnosis: str, confidence: float,
                          clinical_info: str, rag_documents: List[Dict]) -> Tuple[str, List[Dict], float]:
        if not self.gigachat:
            return "🚨 GigaChat недоступен", [], 0.0

        sources_text = ""
        if rag_documents:
            sources_text = "\n\n📚 АКТУАЛЬНЫЕ ИСТОЧНИКИ ИЗ БАЗЫ ЗНАНИЙ:\n"
            for i, doc in enumerate(rag_documents[:5], 1):
                sources_text += f"\n{i}. **{doc.get('title', 'Источник')}** (релевантность: {doc.get('relevance', 0):.1%})\n"
                sources_text += f"   {doc.get('content', '')[:400]}...\n"

        prompt = f"""Ты - AI кардиологический ассистент. Отвечай на вопросы пользователя профессионально и по существу.

ДИАГНОЗ: {diagnosis}
ДОСТОВЕРНОСТЬ ДИАГНОЗА: {confidence:.1%}
КЛИНИЧЕСКАЯ ИНФОРМАЦИЯ: {clinical_info[:500]}
{sources_text}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {message}

ОТВЕТЬ НА РУССКОМ ЯЗЫКЕ, подробно и профессионально.
Если в источниках есть релевантная информация, обязательно используй её и ссылайся на источники.
Если точного ответа нет, дай общие рекомендации и предложи обратиться к врачу.
Будь полезным, но не давай опасных советов - всегда рекомендуй консультацию с врачом при необходимости."""

        for attempt in range(3):
            try:
                chunks = []
                resp = self.gigachat.stream(Chat(messages=[{"role": "user", "content": prompt}]))
                for chunk in resp:
                    delta = chunk.choices[0].delta.content or ""
                    chunks.append(delta)
                response = "".join(chunks).strip()
                avg_relevance = float(np.mean([d.get('relevance', 0) for d in rag_documents])) if rag_documents else 0.0
                return response, rag_documents, avg_relevance
            except Exception as e:
                print(f"   Chat retry {attempt+1}: {e}")
                await asyncio.sleep(2 ** attempt)

        return "🚨 Сервис временно недоступен. Пожалуйста, попробуйте позже.", [], 0.0

print("18. ✅ Класс GigaChatClient определён")

print("19. Определяем класс ProECGNet_SOTA...")
class ProECGNet_SOTA(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(64, 128, 9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, 7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)

print("20. ✅ Класс ProECGNet_SOTA определён")

print("21. Определяем класс AppService...")
class AppService:
    def __init__(self):
        print("=== AppService.__init__: НАЧАЛО ===")
        self.config = config
        self.device = config.device
        self._rag = None
        self._gigachat_client = None
        self._model = None
        self.chat_memory = deque(maxlen=self.config.chat_memory_len)
        self.chat_rag_memory = deque(maxlen=self.config.chat_memory_len)
        print(f"  ✅ AppService инициализирован на {self.device}")

    @property
    def rag(self):
        if self._rag is None:
            print("  🚀 Ленивая загрузка: инициализация RAG...")
            self._rag = QdrantRAG(self.config)
        return self._rag

    @property
    def gigachat_client(self):
        if self._gigachat_client is None:
            print("  🚀 Ленивая загрузка: инициализация GigaChat клиента...")
            self._gigachat_client = GigaChatClient(self.config)
        return self._gigachat_client

    @property
    def model(self):
        if self._model is None:
            print("  🚀 Ленивая загрузка: инициализация нейросети...")
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> ProECGNet_SOTA:
        print("=== _load_model: НАЧАЛО ===")
        try:
            model_path = self.config.get_model_path()
            print(f"  1. Загрузка модели из: {model_path}")
            model = ProECGNet_SOTA(num_classes=len(CLASS_NAMES))
            checkpoint = torch.load(model_path, map_location=self.config.device)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.to(self.config.device)
            model.eval()
            print("  ✅ Модель загружена успешно")
            return model
        except Exception as e:
            print(f"  ❌ Ошибка загрузки модели: {e}")
            traceback = __import__("traceback")
            traceback.print_exc()
            print("  ⚠️ Используется fallback модель")
            model = ProECGNet_SOTA(num_classes=len(CLASS_NAMES))
            model.to(self.config.device)
            model.eval()
            return model

    def _load_ecg_from_file(self, file_content: bytes, filename: str) -> np.ndarray:
        try:
            if filename.endswith('.mat'):
                mat = sio.loadmat(io.BytesIO(file_content))
                if 'val' in mat:
                    ecg = mat['val']
                else:
                    keys = [k for k in mat.keys() if not k.startswith('__')]
                    ecg = mat[keys[0]] if keys else None
                    if ecg is None:
                        raise ValueError("В .mat не найден массив ЭКГ")
            else:
                ecg = np.frombuffer(file_content, dtype=np.float32)

            if ecg.shape == (5000, 12):
                ecg = ecg.T
            elif ecg.shape == (12, 5000):
                pass
            elif ecg.shape == (1, 12, 5000):
                ecg = ecg[0]
            elif ecg.shape == (1, 5000, 12):
                ecg = ecg[0].T
            else:
                raise ValueError(f"Неверная форма: {ecg.shape}")

            return ecg.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки ЭКГ: {str(e)}")

    def _preprocess_ecg(self, ecg_data: np.ndarray) -> torch.Tensor:
        if ecg_data.shape != (12, 5000):
            raise ValueError(f"Ожидается (12, 5000), получено {ecg_data.shape}")

        mean = ecg_data.mean(axis=1, keepdims=True)
        std = ecg_data.std(axis=1, keepdims=True) + 1e-8
        ecg_normalized = (ecg_data - mean) / std

        kernel = np.ones(5) / 5
        ecg_filtered = np.array([np.convolve(lead, kernel, mode='same') for lead in ecg_normalized])

        ecg_tensor = torch.tensor(ecg_filtered, dtype=torch.float32).unsqueeze(0)
        return ecg_tensor.to(self.config.device)

    def _classify_ecg(self, ecg_tensor: torch.Tensor) -> Tuple[str, float, List[Tuple[str, float]]]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(ecg_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            top3_probs, top3_indices = torch.topk(probabilities, k=3, dim=1)
            top3_predictions = []
            for i in range(3):
                class_name = CLASS_NAMES[top3_indices[0][i].item()]
                prob = top3_probs[0][i].item()
                top3_predictions.append((class_name, prob))

        diagnosis = CLASS_NAMES[predicted_idx.item()]
        confidence = confidence.item()
        return diagnosis, confidence, top3_predictions

    def _add_memory_message(self, role: str, text: str):
        self.chat_memory.append({
            "role": role,
            "text": text,
            "ts": datetime.datetime.now().isoformat()
        })

    def _add_memory_sources(self, sources: List[Dict[str, Any]]):
        for s in sources:
            self.chat_rag_memory.append({
                "title": s.get("title", "Источник"),
                "content": (s.get("content") or s.get("full_text") or "")[:800],
                "relevance": float(s.get("relevance", s.get("score", 0.0)))
            })

    def _memory_context(self, max_chars: int = 1800) -> str:
        parts = []
        for m in list(self.chat_memory)[-8:]:
            parts.append(f"{m['role'].upper()}: {m['text']}")
        if self.chat_rag_memory:
            parts.append("\nRecent RAG sources:")
            for s in list(self.chat_rag_memory)[-3:]:
                parts.append(f"- {s['title']} ({s['relevance']:.2f}): {s['content'][:250]}")
        text = "\n".join(parts)
        return text[-max_chars:] if len(text) > max_chars else text

    async def process_ecg(self, ecg_file_content: bytes, filename: str, clinical_notes: str) -> Dict[str, Any]:
        try:
            ecg_data = self._load_ecg_from_file(ecg_file_content, filename)
            processed_ecg = self._preprocess_ecg(ecg_data)
            diagnosis, confidence, top3_predictions = self._classify_ecg(processed_ecg)
            top3_text = ", ".join([f"{cls} ({prob:.1%})" for cls, prob in top3_predictions])

            rag_results, rag_confidence = await self.rag.search_similar(diagnosis, clinical_notes)

            tasks = [
                self.gigachat_client.chat_with_rag(
                    diagnosis=diagnosis,
                    clinical=f"{clinical_notes}\n\nВозможные альтернативные диагнозы: {top3_text}",
                    confidence=confidence,
                    rag_context=rag_results,
                    response_format="structured"
                ),
                self.gigachat_client.chat_with_rag(
                    diagnosis=diagnosis,
                    clinical=f"{clinical_notes}\n\nВозможные альтернативные диагнозы: {top3_text}",
                    confidence=confidence,
                    rag_context=rag_results,
                    response_format="full_cot"
                )
            ]
            structured, full_cot = await asyncio.gather(*tasks)

            return {
                "success": True,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "rag_confidence": rag_confidence,
                "top3_predictions": top3_predictions,
                "rag_references": rag_results,
                "structured_recommendation": structured,
                "full_cot_recommendation": full_cot,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "diagnosis": "ERROR",
                "confidence": 0.0
            }

    async def analyze_clinical_only(self, clinical_notes: str) -> Dict[str, Any]:
        try:
            rag_results, rag_confidence = await self.rag.search_similar("", clinical_notes)
            recommendations = await self.gigachat_client.chat_with_rag(
                diagnosis="Клинический анализ симптомов (без ЭКГ)",
                clinical=clinical_notes,
                confidence=0.0,
                rag_context=rag_results,
                response_format="full_cot"
            )

            return {
                "success": True,
                "diagnosis": "Требуется ЭКГ для точного диагноза",
                "confidence": 0.0,
                "rag_confidence": rag_confidence,
                "structured_recommendation": recommendations,
                "rag_references": rag_results,
                "requires_ecg": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": True,
                "diagnosis": "Требуется ЭКГ для точного диагноза",
                "confidence": 0.0,
                "structured_recommendation": f"Ошибка анализа: {str(e)}",
                "rag_references": [],
                "requires_ecg": True,
                "timestamp": datetime.datetime.now().isoformat()
            }

    async def chat_with_assistant(
        self,
        message: str,
        diagnosis: str,
        confidence: float,
        clinical_info: str,
        previous_rag_context: Dict = None
    ) -> Dict[str, Any]:
        try:
            self._add_memory_message("user", message)

            search_query = f"Диагноз: {diagnosis}\nКлиника: {clinical_info[:300]}\nВопрос: {message}"
            rag_documents, rag_confidence = await self.rag.search_for_chat(search_query, limit=5)

            all_sources = []
            if previous_rag_context:
                prev_refs = previous_rag_context.get("references") or previous_rag_context.get("rag_references") or []
                for ref in prev_refs[:3]:
                    if isinstance(ref, dict):
                        all_sources.append(ref)
                    elif isinstance(ref, str):
                        all_sources.append({"title": "Предыдущий анализ", "content": ref[:800], "relevance": 0.5})

            all_sources.extend(rag_documents)
            self._add_memory_sources(all_sources)

            memory_context = self._memory_context(max_chars=2000)
            prompt_clinical = f"{clinical_info}\n\nRecent conversation and sources:\n{memory_context}\n\nВопрос: {message}"

            answer, used_sources, final_confidence = await self.gigachat_client.chat_answer(
                message=message,
                diagnosis=diagnosis,
                confidence=confidence,
                clinical_info=prompt_clinical,
                rag_documents=all_sources
            )

            self._add_memory_message("assistant", answer)

            return {
                "success": True,
                "response": answer,
                "rag_references": used_sources,
                "rag_confidence": final_confidence,
                "memory": {
                    "messages": list(self.chat_memory),
                    "sources": list(self.chat_rag_memory)
                }
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "response": "Извините, произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте позже.",
                "rag_references": [],
                "rag_confidence": 0.0,
                "error": str(e),
                "memory": {
                    "messages": list(self.chat_memory),
                    "sources": list(self.chat_rag_memory)
                }
            }

print("22. ✅ Класс AppService определён")
print("=== backend/service.py: КОНЕЦ ЗАГРУЗКИ ===")
