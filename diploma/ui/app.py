import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import numpy as np
import os
import sys
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
    </style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_diagnosis' not in st.session_state:
    st.session_state.last_diagnosis = None

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
                result = response.json()
                # Добавляем в результат структурированные документы, если они есть
                if result.get('success') and result.get('rag_references'):
                    # Пытаемся преобразовать rag_references в структурированный формат
                    if isinstance(result.get('rag_references'), list):
                        # Если это список словарей
                        if result.get('rag_references') and isinstance(result['rag_references'][0], dict):
                            result['rag_documents'] = result['rag_references']
                        else:
                            # Если это список строк, создаем структуру
                            result['rag_documents'] = [
                                {
                                    'title': f'Источник {i+1}',
                                    'content': ref,
                                    'relevance': result.get('rag_confidence', 0.5)
                                }
                                for i, ref in enumerate(result.get('rag_references', []))
                            ]
                    else:
                        result['rag_documents'] = []
                return result
            else:
                st.error(f"Ошибка API: {response.status_code} - {response.text}")
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
                result = response.json()
                # Добавляем в результат структурированные документы, если они есть
                if result.get('success') and result.get('formatted_sources'):
                    result['rag_documents'] = result.get('formatted_sources', [])
                    # Добавляем релевантность для каждого документа, если её нет
                    for doc in result['rag_documents']:
                        if 'relevance' not in doc:
                            doc['relevance'] = result.get('rag_confidence', 0.5)
                elif result.get('success') and result.get('rag_references'):
                    if isinstance(result.get('rag_references'), list):
                        if result['rag_references'] and isinstance(result['rag_references'][0], dict):
                            result['rag_documents'] = result['rag_references']
                        else:
                            result['rag_documents'] = [
                                {
                                    'title': f'Источник {i+1}',
                                    'content': ref,
                                    'relevance': result.get('rag_confidence', 0.5)
                                }
                                for i, ref in enumerate(result.get('rag_references', []))
                            ]
                    else:
                        result['rag_documents'] = []
                return result
            else:
                st.error(f"Ошибка API: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Ошибка подключения: {e}")
            return None

def chat_with_bot(message, diagnosis, confidence, clinical_info, rag_context):
    with st.spinner("🤔 Думаю..."):
        # Преобразуем rag_context в строку для отправки
        rag_context_str = json.dumps(rag_context, ensure_ascii=False) if isinstance(rag_context, dict) else str(rag_context)
        
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
    st.sidebar.error(f"❌ API не доступен! URL: {API_URL}\n\nУбедитесь, что бэкенд развернут и переменная API_URL настроена правильно.")

st.markdown("""
<div class="main-header">
    <h1>❤️ GigaCardioAgent</h1>
    <p>Искусственный интеллект для анализа ЭКГ и кардиологических консультаций</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Анализ ЭКГ", 
    "🩺 Клинический анализ", 
    "💬 AI Ассистент",
    "📚 О системе"
])

# TAB 1: Анализ ЭКГ с файлом
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
                st.session_state.last_diagnosis = {
                    'diagnosis': result['diagnosis'],
                    'confidence': result['confidence'],
                    'clinical_info': clinical_info,
                    'rag_refs': result.get('rag_references', []),
                    'rag_documents': result.get('rag_documents', []),
                    'rag_confidence': result.get('rag_confidence', 0.0)
                }
                
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
                
                # Блок отображения уверенности RAG
                if result.get('rag_confidence'):
                    st.metric(
                        label="📚 Уверенность поиска в базе знаний",
                        value=f"{result['rag_confidence']:.1%}",
                        help="Средняя релевантность найденных источников"
                    )
                
                # Отображение топ-3 диагнозов
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
                    
                    # Визуализация распределения вероятностей
                    st.subheader("📈 Распределение вероятностей")
                    prob_data = []
                    for diag, prob in result['top3_predictions']:
                        prob_data.append({"Диагноз": diag, "Вероятность": prob})
                    
                    # Добавляем остальные классы с вероятностью 0
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
                            {ref[:500]}...
                        </div>
                        """, unsafe_allow_html=True)
        else:
            if not ecg_file:
                st.warning("⚠️ Загрузите файл ЭКГ")
            if not clinical_info:
                st.warning("⚠️ Заполните клиническую информацию")

# TAB 2: Клинический анализ (без ЭКГ)
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
                
                st.markdown("""
                <div class="warning-card">
                    ⚠️ <b>Важно:</b> Для постановки точного кардиологического диагноза необходима запись ЭКГ.
                    Данный анализ основан только на симптомах и носит рекомендательный характер.
                </div>
                """, unsafe_allow_html=True)
                
                # Блок отображения уверенности RAG для клинического анализа
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
                                <span style="color: #666; font-size: 0.9em;">(релевантность: {relevance:.1%})</span><br/>
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
                
                if result.get('requires_ecg', True):
                    st.session_state.last_diagnosis = {
                        'diagnosis': "Требуется ЭКГ",
                        'confidence': 0.0,
                        'clinical_info': clinical_only,
                        'rag_refs': result.get('rag_references', []),
                        'rag_documents': result.get('rag_documents', []),
                        'rag_confidence': result.get('rag_confidence', 0.0)
                    }
        else:
            st.warning("⚠️ Введите клиническую информацию")

# TAB 3: AI Ассистент
with tab3:
    st.subheader("💬 AI Кардиологический Ассистент")
    st.markdown("Задайте вопросы о диагнозе, лечении или интерпретации результатов")
    
    if st.session_state.last_diagnosis:
        with st.expander("📋 Контекст последнего анализа", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Диагноз", st.session_state.last_diagnosis['diagnosis'])
                st.metric("Достоверность", f"{st.session_state.last_diagnosis['confidence']:.1%}")
            with col2:
                st.metric("Источников в RAG", len(st.session_state.last_diagnosis.get('rag_refs', [])))
            with col3:
                # Отображение общей уверенности RAG из последнего диагноза
                rag_conf = st.session_state.last_diagnosis.get('rag_confidence', 0)
                if rag_conf > 0:
                    st.metric(
                        "📚 Общая релевантность RAG",
                        f"{rag_conf:.1%}",
                        help="Средняя релевантность всех найденных источников"
                    )
                else:
                    st.metric("📚 RAG доступен", "✅ Да" if st.session_state.last_diagnosis.get('rag_refs') else "⚠️ Нет")
            st.markdown("**Клиническая информация:**")
            st.info(st.session_state.last_diagnosis['clinical_info'][:200] + "...")
        
        # Блок отображения наиболее релевантных документов из RAG
        if st.session_state.last_diagnosis.get('rag_documents') or st.session_state.last_diagnosis.get('rag_refs'):
            st.markdown("---")
            st.subheader("📚 Наиболее релевантные источники из базы знаний")
            st.caption("Документы, найденные при последнем анализе с указанием степени релевантности")
            
            # Получаем документы с релевантностью
            rag_docs = st.session_state.last_diagnosis.get('rag_documents', [])
            
            # Если есть структурированные документы с релевантностью
            if rag_docs and isinstance(rag_docs, list) and len(rag_docs) > 0:
                # Создаем DataFrame для визуализации
                docs_data = []
                for i, doc in enumerate(rag_docs[:5], 1):
                    title = doc.get('title', f'Источник {i}')
                    relevance = doc.get('relevance', doc.get('score', 0))
                    content_preview = doc.get('content', doc.get('text', ''))[:200]
                    
                    docs_data.append({
                        "№": i,
                        "Релевантность": relevance,
                        "Источник": title
                    })
                    
                    # Определяем цвет в зависимости от релевантности
                    if relevance > 0.8:
                        border_color = "#4facfe"
                        emoji = "🟢"
                    elif relevance > 0.6:
                        border_color = "#f6d365"
                        emoji = "🟡"
                    else:
                        border_color = "#fa709a"
                        emoji = "🔴"
                    
                    st.markdown(f"""
                    <div style="
                        background: #f8f9fa;
                        padding: 1rem;
                        border-left: 4px solid {border_color};
                        border-radius: 8px;
                        margin: 0.8rem 0;
                    ">
                        <b>{emoji} Источник #{i}: {title}</b><br/>
                        <span style="color: #666; font-size: 0.9em;">
                            📊 Релевантность: <b>{relevance:.1%}</b>
                        </span><br/>
                        <span style="color: #888; font-size: 0.85em;">
                            {content_preview}...
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Визуализация релевантности документов
                if docs_data:
                    st.markdown("#### 📊 График релевантности документов")
                    df_docs = pd.DataFrame(docs_data)
                    st.bar_chart(df_docs.set_index("№")["Релевантность"])
                    
                    # Общая статистика
                    avg_relevance = np.mean([d["Релевантность"] for d in docs_data])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "📈 Средняя релевантность топ-документов",
                            f"{avg_relevance:.1%}",
                            help="Средняя релевантность наиболее релевантных источников"
                        )
                    with col2:
                        best_match = max(docs_data, key=lambda x: x["Релевантность"])
                        st.metric(
                            "🏆 Наиболее релевантный источник",
                            best_match["Источник"][:30],
                            f"{best_match['Релевантность']:.1%}"
                        )
            
            # Если есть только текстовые референсы (без структурированной релевантности)
            elif st.session_state.last_diagnosis.get('rag_refs'):
                st.info("📖 Найдены следующие источники (информация о релевантности отсутствует):")
                for i, ref in enumerate(st.session_state.last_diagnosis['rag_refs'][:3], 1):
                    with st.expander(f"📖 Источник {i}"):
                        st.markdown(ref[:500] + ("..." if len(ref) > 500 else ""))
            
            st.markdown("---")
        
    else:
        st.info("ℹ️ Сначала выполните анализ ЭКГ или клинический анализ, чтобы получить контекст для чата")
    
    st.markdown("---")
    
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
                st.markdown("---")
    
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
    
    if st.button("🗑️ Очистить историю", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if send_button and user_message and st.session_state.last_diagnosis:
        diagnosis = st.session_state.last_diagnosis.get('diagnosis', 'N/A')
        confidence = st.session_state.last_diagnosis.get('confidence', 0.0)
        clinical_info = st.session_state.last_diagnosis.get('clinical_info', '')
        rag_refs = st.session_state.last_diagnosis.get('rag_refs', [])
        rag_docs = st.session_state.last_diagnosis.get('rag_documents', [])
        rag_confidence = st.session_state.last_diagnosis.get('rag_confidence', 0.0)
        
        # Передаем в чат полный RAG контекст
        rag_context = {
            'references': rag_refs,
            'documents': rag_docs,
            'confidence': rag_confidence
        }
        
        st.session_state.chat_history.append({'role': 'user', 'content': user_message})
        
        response = chat_with_bot(user_message, diagnosis, confidence, clinical_info, rag_context)
        
        if response and response.get('success'):
            bot_response = response['response']
            st.session_state.chat_history.append({'role': 'assistant', 'content': bot_response})
            st.rerun()
        else:
            st.error("❌ Ошибка при получении ответа от ассистента")

# TAB 4: О системе
with tab4:
    st.markdown("""
    ### 🚀 GigaCardioAgent v2.0
    
    **Интеллектуальная система для кардиологической диагностики**
    
    #### 🔬 Технологии:
    - **Нейросетевая модель:** Сверточная нейронная сеть для анализа ЭКГ
    - **RAG (Retrieval-Augmented Generation):** Векторная база клинических рекомендаций
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
    - Клинический анализ использует RAG для поиска похожих случаев
    - AI ассистент учитывает контекст последнего анализа
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
    "<div style='text-align: center; color: gray; padding: 1rem;'>© 2024 GigaCardioAgent - Ваш AI кардиолог | Работает на GigaChat и нейронных сетях</div>",
    unsafe_allow_html=True
)

if st.sidebar.button("🔄 Обновить статус"):
    st.cache_data.clear()
    st.rerun()
