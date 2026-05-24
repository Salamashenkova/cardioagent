import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import numpy as np
import os
import sys
import re
from pathlib import Path

# Добавляем путь для импортов (если нужно)
sys.path.append(str(Path(__file__).parent.parent))

# Используем переменную окружения для URL бэкенда
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="GigaCardioAgent - AI Кардиолог",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .diagnosis-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .confidence-high {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .confidence-mid {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    }
    .confidence-low {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .rag-reference {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
    }
    .source-card {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .chat-source-previous {
        background: #f8f9fa;
        padding: 0.8rem;
        border-left: 3px solid #4facfe;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .chat-source-new {
        background: #e8f4f8;
        padding: 0.8rem;
        border-left: 3px solid #00b4d8;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Инициализация session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_diagnosis' not in st.session_state:
    st.session_state.last_diagnosis = None
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = []  # Кратковременная память для чата

def format_conversation_memory(memory, max_pairs=3):
    """
    Форматирует историю диалога для передачи в промпт.
    max_pairs = 3 означает 3 сообщения пользователя и 3 ответа ассистента (всего 6 сообщений)
    """
    if not memory:
        return ""
    
    # Берём последние max_pairs * 2 сообщений (каждая пара = user + assistant)
    max_messages = max_pairs * 2
    recent = memory[-max_messages:] if len(memory) > max_messages else memory
    
    formatted = "\n\n## Предыдущий диалог:\n"
    for msg in recent:
        role = "Пользователь" if msg['role'] == 'user' else "Ассистент"
        formatted += f"{role}: {msg['content']}\n"
    
    return formatted

@st.cache_data(ttl=60)
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Health check error: {e}")
    return None

def analyze_ecg(file, clinical_info):
    with st.spinner("🫀 Анализируем ЭКГ..."):
        files = {"ecg_file": file}
        data = {"clinical_info": clinical_info}
        try:
            response = requests.post(f"{API_URL}/analyze_ecg", files=files, data=data, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Ошибка API: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Ошибка подключения: {e}")
            return None

def analyze_clinical_only(clinical_info):
    with st.spinner("🩺 Анализируем симптомы..."):
        data = {"clinical_info": clinical_info}
        try:
            response = requests.post(f"{API_URL}/analyze_clinical", data=data, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Ошибка API: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Ошибка подключения: {e}")
            return None

def chat_with_bot(message, diagnosis, confidence, clinical_info, rag_context):
    with st.spinner("🤔 Думаю..."):
        rag_context_str = json.dumps(rag_context, ensure_ascii=False) if rag_context else "{}"
        
        data = {
            "message": message,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "clinical_info": clinical_info,
            "rag_context": rag_context_str
        }
        try:
            response = requests.post(f"{API_URL}/chat", data=data, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Ошибка API: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Ошибка подключения: {e}")
            return None

# ==================== БОКОВАЯ ПАНЕЛЬ ====================
st.sidebar.title("🔌 Статус системы")
api_status = check_api_health()

if api_status:
    st.sidebar.success(f"✅ API Online")
    st.sidebar.markdown(f"""
    **💻 Device:** {api_status.get('device', 'N/A')}  
    **📊 Model:** {'✅ Loaded' if api_status.get('model_loaded') else '❌ Not loaded'}
    """)
    
    if 'classes' in api_status:
        st.sidebar.markdown("### 🏷️ Доступные диагнозы")
        classes = api_status['classes']
        for i, cls in enumerate(classes[:8]):
            st.sidebar.markdown(f"- {cls}")
        if len(classes) > 8:
            st.sidebar.markdown(f"... и {len(classes) - 8} других")
else:
    st.sidebar.error(f"❌ API не доступен! URL: {API_URL}")

# ==================== ЗАГОЛОВОК ====================
st.markdown("""
<div class="main-header">
    <h1>❤️ GigaCardioAgent</h1>
    <p>Искусственный интеллект для анализа ЭКГ и кардиологических консультаций</p>
</div>
""", unsafe_allow_html=True)

# ==================== ТАБЫ ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Анализ ЭКГ", 
    "🩺 Клинический анализ", 
    "💬 AI Ассистент",
    "📚 О системе"
])

# ==================== TAB 1: Анализ ЭКГ ====================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Загрузка ЭКГ")
        ecg_file = st.file_uploader(
            "Выберите файл ЭКГ",
            type=['mat', 'npy', 'csv'],
            help="Поддерживаются форматы: .mat (PTB-XL), .npy, .csv"
        )
        
        if ecg_file:
            st.info(f"✅ Файл загружен: {ecg_file.name} ({ecg_file.size/1024:.1f} KB)")
    
    with col2:
        st.subheader("📋 Клиническая информация")
        clinical_info = st.text_area(
            "Анамнез, симптомы, жалобы",
            value="Пациент 65 лет, мужчина, жалобы на боль в груди при нагрузке, одышку, АГ в анамнезе",
            height=150,
            help="Подробно опишите клиническую картину для более точного анализа",
            key="clinical_ecg"
        )
    
    if st.button("🔍 Провести анализ ЭКГ", use_container_width=True, key="analyze_ecg_btn"):
        if ecg_file and clinical_info:
            result = analyze_ecg(ecg_file, clinical_info)
            
            if result and result.get('success'):
                # Сохраняем контекст для чата
                st.session_state.last_diagnosis = {
                    'diagnosis': result['diagnosis'],
                    'confidence': result['confidence'],
                    'clinical_info': clinical_info,
                    'rag_refs': result.get('rag_references', []),
                    'rag_confidence': result.get('rag_confidence', 0.0),
                    'type': 'ecg'
                }
                # Очищаем память чата при новом анализе
                st.session_state.conversation_memory = []
                st.session_state.chat_history = []
                
                st.success("✅ Анализ завершен!")
                
                confidence = result['confidence']
                if confidence > 0.8:
                    conf_class = "confidence-high"
                elif confidence > 0.6:
                    conf_class = "confidence-mid"
                else:
                    conf_class = "confidence-low"
                
                st.markdown(f"""
                <div class="diagnosis-card {conf_class}">
                    <h2>📊 Диагноз: {result['diagnosis']}</h2>
                    <p>Достоверность: {confidence:.1%}</p>
                    <p>🕐 {result['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if result.get('rag_confidence'):
                    st.metric(
                        label="📚 Уверенность поиска в базе знаний",
                        value=f"{result['rag_confidence']:.1%}",
                        help="Средняя релевантность найденных источников"
                    )
                
                if result.get('top3_predictions'):
                    st.subheader("📊 Все возможные диагнозы")
                    cols = st.columns(3)
                    for i, (diag, prob) in enumerate(result['top3_predictions']):
                        with cols[i]:
                            st.metric(
                                label=f"{diag}",
                                value=f"{prob:.1%}",
                                delta=None
                            )
                    
                    st.subheader("📈 Распределение вероятностей")
                    prob_data = []
                    for diag, prob in result['top3_predictions']:
                        prob_data.append({"Диагноз": diag, "Вероятность": prob})
                    
                    all_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
                    existing = [p[0] for p in result['top3_predictions']]
                    for cls in all_classes:
                        if cls not in existing:
                            prob_data.append({"Диагноз": cls, "Вероятность": 0.0})
                    
                    df_probs = pd.DataFrame(prob_data)
                    st.bar_chart(df_probs.set_index("Диагноз"))
                
                with st.expander("📋 Структурированные рекомендации", expanded=True):
                    rec = result.get('structured_recommendation', {})
                    if isinstance(rec, dict):
                        for key, value in rec.items():
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    elif isinstance(rec, str):
                        st.write(rec)
                    else:
                        st.json(rec)
                
                with st.expander("💬 Полный анализ (Chain of Thought)", expanded=False):
                    st.markdown(result.get('full_cot_recommendation', 'Нет данных'))
                
                if result.get('rag_references'):
                    st.subheader("📚 Источники из базы знаний")
                    for i, ref in enumerate(result['rag_references'][:3]):
                        st.markdown(f"""
                        <div class="rag-reference">
                            <b>📖 Источник {i+1}</b><br/>
                            {str(ref)[:500]}...
                        </div>
                        """, unsafe_allow_html=True)
        else:
            if not ecg_file:
                st.warning("⚠️ Загрузите файл ЭКГ")
            if not clinical_info:
                st.warning("⚠️ Заполните клиническую информацию")

# ==================== TAB 2: Клинический анализ ====================
with tab2:
    st.subheader("🩺 Анализ без ЭКГ")
    st.markdown("""
    Опишите клиническую картину, и AI предоставит рекомендации на основе 
    **базы клинических знаний**.
    """)
    
    clinical_only = st.text_area(
        "Клиническая картина",
        height=200,
        placeholder="Пример: Пациент 45 лет, мужчина, жалобы на одышку при нагрузке, повышение АД до 160/90, головные боли.",
        key="clinical_only_input"
    )
    
    if st.button("🔍 Анализировать симптомы", key="clinical_btn", use_container_width=True):
        if clinical_only:
            result = analyze_clinical_only(clinical_only)
            
            if result and result.get('success'):
                st.success("✅ Анализ симптомов завершен!")
                
                # Сохраняем контекст для чата
                st.session_state.last_diagnosis = {
                    'diagnosis': result.get('diagnosis', "Требуется ЭКГ"),
                    'confidence': result.get('confidence', 0.0),
                    'clinical_info': clinical_only,
                    'rag_refs': result.get('rag_references', []),
                    'rag_confidence': result.get('rag_confidence', 0.0),
                    'type': 'clinical'
                }
                # Очищаем память чата при новом анализе
                st.session_state.conversation_memory = []
                st.session_state.chat_history = []
                
                st.markdown("""
                <div class="warning-card">
                    ⚠️ <b>Важно:</b> Для постановки точного кардиологического диагноза необходима запись ЭКГ.
                    Данный анализ основан только на симптомах и носит рекомендательный характер.
                </div>
                """, unsafe_allow_html=True)
                
                if result.get('rag_confidence'):
                    st.metric(
                        label="📚 Уверенность поиска в базе знаний",
                        value=f"{result['rag_confidence']:.1%}",
                        help="Средняя релевантность найденных источников"
                    )
                
                st.subheader("💡 Клинические рекомендации")
                st.markdown(result.get('structured_recommendation', ''))
                
                if result.get('formatted_sources'):
                    with st.expander("📚 Источники из базы знаний", expanded=True):
                        for i, source in enumerate(result['formatted_sources'], 1):
                            relevance = source.get('relevance', result.get('rag_confidence', 0.5))
                            st.markdown(f"""
                            <div class="source-card">
                                <b>{i}. {source.get('title', 'Источник')}</b>
                                <span style="color: #666;">(релевантность: {relevance:.1%})</span><br/>
                                {source.get('content', source.get('text', ''))[:300]}...
                            </div>
                            """, unsafe_allow_html=True)
                
                if result.get('recommended_actions'):
                    st.subheader("📋 Рекомендованные действия")
                    for action in result['recommended_actions']:
                        st.markdown(f"- {action}")
                
                sources_count = len(result.get('formatted_sources', []))
                if sources_count > 0:
                    st.caption(f"✅ Анализ основан на {sources_count} источниках из базы знаний")
        else:
            st.warning("⚠️ Введите клиническую информацию")

# ==================== TAB 3: AI Ассистент ====================
with tab3:
    st.subheader("💬 AI Кардиологический Ассистент")
    st.markdown("Задайте вопросы о диагнозе, лечении или интерпретации результатов. Ассистент помнит последние 3-5 сообщений диалога.")
    
    if st.session_state.last_diagnosis:
        with st.expander("📋 Контекст последнего анализа", expanded=False):
            analysis_type = "ЭКГ" if st.session_state.last_diagnosis.get('type') == 'ecg' else "клинический"
            st.caption(f"Анализ: {analysis_type}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Диагноз", st.session_state.last_diagnosis['diagnosis'])
                st.metric("Достоверность", f"{st.session_state.last_diagnosis['confidence']:.1%}")
            with col2:
                total_sources = len(st.session_state.last_diagnosis.get('rag_refs', []))
                st.metric("Источников в RAG", total_sources)
            with col3:
                rag_conf = st.session_state.last_diagnosis.get('rag_confidence', 0)
                if rag_conf > 0:
                    st.metric("📚 Общая релевантность RAG", f"{rag_conf:.1%}")
            st.markdown("**Клиническая информация:**")
            st.info(st.session_state.last_diagnosis['clinical_info'][:300] + ("..." if len(st.session_state.last_diagnosis['clinical_info']) > 300 else ""))
        
        # Отображаем ПРЕДЫДУЩИЕ источники (из последнего анализа)
        previous_sources = st.session_state.last_diagnosis.get('rag_refs', [])
        if previous_sources:
            st.markdown("---")
            st.subheader("📚 Источники из последнего анализа")
            st.caption("Эти источники были найдены при анализе ЭКГ/симптомов")
            
            for i, source in enumerate(previous_sources[:3], 1):
                if isinstance(source, dict):
                    title = source.get('title', f'Источник {i}')
                    relevance = source.get('relevance', 0.5)
                    content = source.get('content', '')[:200]
                    st.markdown(f"""
                    <div class="chat-source-previous">
                        <b>📖 {i}. {title}</b>
                        <span style="color: #666;">(релевантность: {relevance:.1%})</span><br/>
                        <span style="color: #555; font-size: 0.85em;">{content}...</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-source-previous">
                        <b>📖 {i}. Источник</b><br/>
                        <span style="color: #555; font-size: 0.85em;">{str(source)[:200]}...</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
    else:
        st.info("ℹ️ Сначала выполните анализ ЭКГ или клинический анализ, чтобы получить контекст для чата")
    
    # История чата
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("💬 Задайте вопрос AI ассистенту о диагнозе или лечении")
        else:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"**👤 Вы:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 AI Ассистент:** {msg['content']}")
                    
                    # Отображаем НОВЫЕ источники
                    if msg.get('rag_references') and len(msg.get('rag_references', [])) > 0:
                        rag_conf = msg.get('rag_confidence', 0)
                        with st.expander(f"🔍 Найдено по вашему вопросу (релевантность: {rag_conf:.1%})", expanded=False):
                            for i, ref in enumerate(msg['rag_references'][:3], 1):
                                st.markdown(f"""
                                <div class="chat-source-new">
                                    <b>🔍 {i}. Источник</b><br/>
                                    <span style="color: #555; font-size: 0.85em;">{str(ref)[:200]}...</span>
                                </div>
                                """, unsafe_allow_html=True)
                st.markdown("---")
    
    # Ввод вопроса
    col1, col2 = st.columns([4, 1])
    with col1:
        user_message = st.text_input(
            "Ваш вопрос:", 
            key="chat_input", 
            placeholder="Например: Какие лекарства рекомендуются при таком диагнозе?",
            disabled=not st.session_state.last_diagnosis
        )
    with col2:
        send_button = st.button("✉️ Отправить", use_container_width=True, disabled=not st.session_state.last_diagnosis)
    
    col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 1])
    with col_clear2:
        if st.button("🗑️ Очистить историю", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.rerun()
    
    if send_button and user_message and st.session_state.last_diagnosis:
        diagnosis = st.session_state.last_diagnosis.get('diagnosis', 'N/A')
        confidence = st.session_state.last_diagnosis.get('confidence', 0.0)
        clinical_info = st.session_state.last_diagnosis.get('clinical_info', '')
        rag_refs = st.session_state.last_diagnosis.get('rag_refs', [])
        rag_confidence = st.session_state.last_diagnosis.get('rag_confidence', 0.0)
        
        # Форматируем историю диалога для передачи (максимум 3 пары = 6 сообщений)
        conversation_history = st.session_state.conversation_memory[-6:] if len(st.session_state.conversation_memory) > 6 else st.session_state.conversation_memory
        
        # Подготавливаем контекст для чата
        rag_context = {
            'references': rag_refs,
            'confidence': rag_confidence,
            'diagnosis': diagnosis,
            'clinical_info': clinical_info,
            'conversation_history': conversation_history
        }
        
        # Добавляем сообщение пользователя в память
        st.session_state.conversation_memory.append({
            'role': 'user',
            'content': user_message
        })
        
        # Добавляем в историю чата для отображения
        st.session_state.chat_history.append({'role': 'user', 'content': user_message})
        
        # Получаем ответ от API
        response = chat_with_bot(user_message, diagnosis, confidence, clinical_info, rag_context)
        
        if response and response.get('success'):
            bot_response = response.get('response', '')
            rag_references = response.get('rag_references', [])
            rag_conf = response.get('rag_confidence', 0.0)
            
            # Добавляем ответ ассистента в память
            st.session_state.conversation_memory.append({
                'role': 'assistant',
                'content': bot_response
            })
            
            # Добавляем в историю чата для отображения
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': bot_response,
                'rag_references': rag_references,
                'rag_confidence': rag_conf
            })
            st.rerun()
        elif response and not response.get('success'):
            st.error(f"❌ Ошибка: {response.get('error', 'Неизвестная ошибка')}")
            # Удаляем сообщение пользователя при ошибке
            st.session_state.chat_history.pop()
            st.session_state.conversation_memory.pop()
        else:
            st.error("❌ Ошибка при получении ответа от ассистента")
            st.session_state.chat_history.pop()
            st.session_state.conversation_memory.pop()

# ==================== TAB 4: О системе ====================
with tab4:
    st.markdown("""
    ### 🚀 GigaCardioAgent v2.0
    
    **Интеллектуальная система для кардиологической диагностики**
    
    #### 🔬 Технологии:
    - **Нейросетевая модель:** ProECGNet для анализа ЭКГ
    - **RAG (Retrieval-Augmented Generation):** Векторная база клинических рекомендаций в Qdrant
    - **LLM:** GigaChat для генерации интерпретаций
    - **FastAPI + Streamlit:** Высокопроизводительный бэкенд и удобный интерфейс
    
    #### 📊 Поддерживаемые диагнозы:
    """)
    
    if api_status and 'classes' in api_status:
        classes = api_status['classes']
        cols = st.columns(3)
        for i, cls in enumerate(classes):
            with cols[i % 3]:
                st.markdown(f"- **{cls}**")
    
    st.markdown("""
    #### 📁 Форматы файлов:
    - `.mat` - PTB-XL датасет (12 отведений, 5000 точек)
    - `.npy` - NumPy массивы
    - `.csv` - Табличные данные
    
    #### 🔄 API Endpoints:
    - `POST /analyze_ecg` - Анализ с загрузкой ЭКГ
    - `POST /analyze_clinical` - Анализ только симптомов
    - `POST /chat` - RAG чат с AI
    
    #### 💡 Особенности:
    - Ассистент помнит последние 3-5 сообщений диалога
    - Клинический анализ использует RAG для поиска похожих случаев
    - Все рекомендации основаны на клинических источниках
    """)
    
    if api_status:
        st.markdown("---")
        st.subheader("📊 Статус системы")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Устройство", api_status.get('device', 'N/A'))
        with col2:
            status = "✅ Загружена" if api_status.get('model_loaded') else "❌ Не загружена"
            st.metric("Модель", status)
        with col3:
            st.metric("Диагнозов", len(api_status.get('classes', [])))

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>© 2025 GigaCardioAgent - Ваш AI кардиолог | Работает на GigaChat и нейронных сетях</div>",
    unsafe_allow_html=True
)

if st.sidebar.button("🔄 Обновить статус"):
    st.cache_data.clear()
    st.rerun()

Ответить
