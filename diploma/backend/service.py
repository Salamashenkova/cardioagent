# diploma/backend/service.py - ПОЛНАЯ ВЕРСИЯ С ЛЕНИВОЙ ЗАГРУЗКОЙ

import sys
from pathlib import Path

# Добавляем путь к корневой папке diploma
sys.path.append(str(Path(__file__).parent.parent))

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import scipy.io as sio
import io
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import re
from collections import Counter
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

load_dotenv()

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/ProECGNetbeststtc.pth"))
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = "kr_production_cloud"
    gigachat_credentials: str = os.getenv("GIGACHAT_CREDENTIALS")
    verify_ssl: bool = os.getenv("VERIFY_SSL", "false").lower() == "true"

config = Config()
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
print(f"🚀 HybridRAG Pro v2.0 | Device: {config.device} | Qdrant: {bool(config.qdrant_url)} | GigaChat: {bool(config.gigachat_credentials)}")

class BM25:
    """BM25 с MIN-MAX нормализацией [0,1]"""
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
        avgdl = np.mean(doc_lens)
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
                if tf > 0:
                    bm25_term = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    score += bm25_term * self.idf.get(term, 0)
            raw_scores.append(score)
        min_score, max_score = min(raw_scores), max(raw_scores)
        if max_score > min_score:
            return [(s - min_score) / (max_score - min_score) for s in raw_scores]
        return [0.5] * len(raw_scores)

class QdrantRAG:
    """RAG поиск в базе знаний"""
    def __init__(self, config):
        self.config = config
        if config.qdrant_url and config.qdrant_api_key:
            try:
                self.client = QdrantClient(
                    url=config.qdrant_url, 
                    api_key=config.qdrant_api_key,
                    timeout=60.0
                )
                print(f"✅ Qdrant подключен: {config.qdrant_url}")
            except Exception as e:
                self.client = None
                print(f"⚠️ Ошибка подключения к Qdrant: {e}")
        else:
            self.client = None
            print("⚠️ Qdrant недоступен")
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Ленивая загрузка модели эмбеддингов"""
        if self._embedding_model is None:
            print("🚀 Ленивая загрузка: инициализация SentenceTransformer...")
            self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self._embedding_model

    async def search_similar(self, diagnosis: str, clinical: str, limit: int = 10) -> List[str]:
        """Поиск похожих документов в базе знаний"""
        if not self.client:
            return ["Qdrant недоступен - используем общие рекомендации"]
        
        # Формируем поисковый запрос
        if diagnosis and diagnosis != "Клинический анализ симптомов (без ЭКГ)":
            query_text = f"{diagnosis} {clinical}"
        else:
            query_text = clinical
        
        print(f"🔍 Поиск в базе знаний: {query_text[:100]}...")
        
        try:
            query_vec = self.embedding_model.encode(query_text).tolist()
            
            hits = self.client.query_points(
                collection_name=self.config.qdrant_collection,
                query=query_vec,
                limit=limit,
                score_threshold=0.65
            )
            
            results = []
            for hit in hits.points:
                title = hit.payload.get('title', 'Документ без названия')
                text = hit.payload.get('text', '')
                score = hit.score
                
                formatted = f"**{title}** (релевантность: {score:.2f})\n{text[:500]}..."
                results.append(formatted)
            
            if not results:
                return ["Не найдено релевантных документов в базе знаний"]
            
            return results
            
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return ["Ошибка поиска в базе знаний. Используем общие рекомендации."]

class GigaChatClient:
    def __init__(self, config):
        self.config = config
        self._gigachat = None

    @property
    def gigachat(self):
        """Ленивая загрузка GigaChat клиента"""
        if self._gigachat is None and self.config.gigachat_credentials:
            print("🚀 Ленивая загрузка: инициализация GigaChat...")
            self._gigachat = GigaChat(
                credentials=self.config.gigachat_credentials,
                scope="GIGACHAT_API_PERS",
                verify_ssl_certs=self.config.verify_ssl,
                model="GigaChat-Pro"
            )
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
                print(f"GigaChat retry {attempt+1}: {e}")
                await asyncio.sleep(2 ** attempt)
        return "🚨 GIGA OFFLINE: АСА 160мг + ЭКГ повтор + кардиолог ОЧНО"

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
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.head(x)

class AppService:
    def __init__(self):
        self.config = config
        self.device = config.device
        # Ленивая загрузка - ничего не инициализируем здесь
        self._rag = None
        self._gigachat_client = None
        self._model = None
        print(f"✅ AppService инициализирован (легковесная версия) на {self.device}")

    @property
    def rag(self):
        """Ленивая загрузка RAG компонента"""
        if self._rag is None:
            print("🚀 Ленивая загрузка: инициализация RAG...")
            self._rag = QdrantRAG(self.config)
        return self._rag

    @property
    def gigachat_client(self):
        """Ленивая загрузка GigaChat клиента"""
        if self._gigachat_client is None:
            print("🚀 Ленивая загрузка: инициализация GigaChat клиента...")
            self._gigachat_client = GigaChatClient(self.config)
        return self._gigachat_client

    @property
    def model(self):
        """Ленивая загрузка нейросетевой модели"""
        if self._model is None:
            print("🚀 Ленивая загрузка: инициализация нейросети...")
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> ProECGNet_SOTA:
        """Загрузка модели"""
        try:
            print(f"🔄 Загрузка модели: {self.config.model_path}")
            model = ProECGNet_SOTA(num_classes=len(CLASS_NAMES))
            
            checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
                
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            model.to(self.config.device)
            model.eval()
            
            print(f"✅ Модель загружена успешно!")
            return model
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            model = ProECGNet_SOTA(num_classes=len(CLASS_NAMES))
            model.to(self.config.device)
            model.eval()
            print("⚠️ Используется fallback модель")
            return model

    def _load_ecg_from_file(self, file_content: bytes, filename: str) -> np.ndarray:
        """Загрузка ЭКГ из bytes"""
        try:
            if filename.endswith('.mat'):
                mat = sio.loadmat(io.BytesIO(file_content))
                if 'val' in mat:
                    ecg = mat['val']
                else:
                    ecg = mat[list(mat.keys())[-1]]
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
            
            print(f"📊 ЭКГ загружена: форма {ecg.shape}")
            return ecg.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки ЭКГ: {str(e)}")

    def _preprocess_ecg(self, ecg_data: np.ndarray) -> torch.Tensor:
        """Предобработка ЭКГ"""
        if ecg_data.shape != (12, 5000):
            raise ValueError(f"Ожидается (12, 5000), получено {ecg_data.shape}")
        
        mean = ecg_data.mean(axis=1, keepdims=True)
        std = ecg_data.std(axis=1, keepdims=True) + 1e-8
        ecg_normalized = (ecg_data - mean) / std
        
        kernel = np.ones(5) / 5
        ecg_filtered = np.array([np.convolve(lead, kernel, mode='same') for lead in ecg_normalized])
        
        ecg_tensor = torch.tensor(ecg_filtered, dtype=torch.float32)
        ecg_tensor = ecg_tensor.unsqueeze(0)
        
        return ecg_tensor.to(self.config.device)

    def _classify_ecg(self, ecg_tensor: torch.Tensor) -> Tuple[str, float]:
        """Классификация ЭКГ"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(ecg_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        diagnosis = CLASS_NAMES[predicted_idx.item()]
        confidence = confidence.item()
        print(f"🎯 Диагноз: {diagnosis} (confidence: {confidence:.3f})")
        return diagnosis, confidence

    async def process_ecg(self, ecg_file_content: bytes, filename: str, clinical_notes: str) -> Dict[str, Any]:
        """Анализ ЭКГ с файлом"""
        try:
            ecg_data = self._load_ecg_from_file(ecg_file_content, filename)
            processed_ecg = self._preprocess_ecg(ecg_data)
            print(f"🔧 После предобработки: форма {processed_ecg.shape}")
            
            diagnosis, confidence = self._classify_ecg(processed_ecg)
            rag_results = await self.rag.search_similar(diagnosis, clinical_notes)
            
            tasks = [
                self.gigachat_client.chat_with_rag(diagnosis, clinical_notes, confidence, rag_results, "structured"),
                self.gigachat_client.chat_with_rag(diagnosis, clinical_notes, confidence, rag_results, "full_cot")
            ]
            structured, full_cot = await asyncio.gather(*tasks)
            
            return {
                "success": True,
                "diagnosis": diagnosis,
                "confidence": confidence,
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
        """Анализ только клинических симптомов без ЭКГ"""
        try:
            print(f"🩺 Клинический анализ симптомов: {clinical_notes[:100]}...")
            
            rag_results = await self.rag.search_similar("", clinical_notes)
            
            recommendations = await self.gigachat_client.chat_with_rag(
                diagnosis="Клинический анализ симптомов (без ЭКГ)",
                clinical=clinical_notes,
                confidence=0.0,
                rag_context=rag_results,
                response_format="full_cot"
            )
            
            formatted_sources = []
            for i, source in enumerate(rag_results[:5], 1):
                if "**" in source:
                    lines = source.split('\n')
                    title = lines[0].replace('**', '') if lines else "Источник"
                    content = '\n'.join(lines[1:]) if len(lines) > 1 else source
                else:
                    title = f"Источник {i}"
                    content = source[:300] + "..." if len(source) > 300 else source
                
                formatted_sources.append({
                    "title": title,
                    "content": content,
                    "full_text": source
                })
            
            sources_section = ""
            if formatted_sources:
                sources_section = "\n\n### 📚 Источники из базы знаний\n\n"
                for i, source in enumerate(formatted_sources, 1):
                    sources_section += f"**{i}. {source['title']}**\n"
                    sources_section += f"{source['content'][:200]}...\n\n"
            
            structured_rec = f"""### 📋 Оценка симптомов

**Анализ на основе предоставленных симптомов:**

{recommendations}

{sources_section}

### ⚠️ Важное примечание
**Для постановки точного кардиологического диагноза необходима запись ЭКГ.**

**Рекомендованные действия:**
1. Пройти запись ЭКГ в покое (12 отведений)
2. Консультация кардиолога с результатами ЭКГ
3. При необходимости - дополнительные обследования:
   - Эхокардиография (ЭхоКГ)
   - Холтеровское мониторирование ЭКГ (24-48 часов)
   - Общий и биохимический анализ крови
"""
            
            return {
                "success": True,
                "diagnosis": "Требуется ЭКГ для точного диагноза",
                "confidence": 0.0,
                "structured_recommendation": structured_rec,
                "rag_references": rag_results,
                "formatted_sources": formatted_sources,
                "recommended_actions": ["ЭКГ", "Консультация кардиолога", "Эхокардиография", "Анализ крови"],
                "requires_ecg": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Ошибка клинического анализа: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": True,
                "diagnosis": "Требуется ЭКГ для точного диагноза",
                "confidence": 0.0,
                "structured_recommendation": """### 📋 Клинический анализ (без ЭКГ)

На основании описанных симптомов невозможно поставить точный диагноз.

**Рекомендованные действия:**
1. Запись ЭКГ в покое (12 отведений)
2. Консультация кардиолога
3. Общий и биохимический анализ крови

### ⚠️ Важно
Для постановки точного диагноза необходима запись ЭКГ и очная консультация кардиолога.""",
                "rag_references": ["Для точной диагностики необходима запись ЭКГ и консультация кардиолога"],
                "formatted_sources": [],
                "recommended_actions": ["ЭКГ", "Консультация кардиолога", "Общий анализ крови"],
                "requires_ecg": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
