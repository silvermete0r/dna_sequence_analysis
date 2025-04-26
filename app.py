import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import base64
import os
import datetime

st.set_page_config(page_title="Анализатор ДНК", layout="wide")

@st.cache_data
def calculate_gc_content(sequence):
    if not sequence:
        return 0
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

@st.cache_data
def extract_sequence_features(df):
    features = []
    for idx, row in df.iterrows():
        seq = row['sequence']
        length = len(seq)
        
        features.append({
            'seq_id': idx,
            'class': row['class'],
            'length': length,
            'gc_content': calculate_gc_content(seq),
            'a_freq': seq.count('A') / length,
            't_freq': seq.count('T') / length,
            'g_freq': seq.count('G') / length,
            'c_freq': seq.count('C') / length,
            'cpg_count': len(re.findall('CG', seq))
        })
    
    return pd.DataFrame(features)

@st.cache_data
def analyze_kmers(sequences, k, max_samples=50):
    """Анализ k-меров с кэшированием"""
    sample_size = min(max_samples, len(sequences))
    samples = np.random.choice(sequences, sample_size, replace=False)
    
    kmer_counts = Counter()
    for seq in samples:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] += 1
            
    return pd.DataFrame(kmer_counts.most_common(20), columns=['k-мер', 'частота'])

def get_csv_download_link(dataframe, filename="data.csv"):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Скачать CSV</a>'

# Основной интерфейс
st.title("🧬 Анализатор структуры ДНК")
st.markdown("""
### Основные возможности:
- Анализ GC-состава
- Частотный анализ нуклеотидов
- Распознавание паттернов CpG-островков
- Визуализация структурных особенностей
""")

# Загрузка файлов
st.header("Управление файлами ДНК")
upload_tab, select_tab = st.tabs(["Загрузить новый файл", "Выбрать существующий"])

with upload_tab:
    uploaded_file = st.file_uploader("Загрузить файл последовательностей ДНК (txt)", type=['txt'])
    
    if uploaded_file:
        try:
            df_new = pd.read_csv(uploaded_file, sep='\t')
            if 'sequence' in df_new.columns and 'class' in df_new.columns:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dna_sequences_{timestamp}.txt"
                os.makedirs("data", exist_ok=True)
                df_new.to_csv(os.path.join("data", filename), sep='\t', index=False)
                st.success(f"Файл успешно сохранен как {filename}")
            else:
                st.error("Файл должен содержать колонки 'sequence' и 'class'")
        except Exception as e:
            st.error(f"Ошибка обработки файла: {str(e)}")

with select_tab:
    data_folder = "data"
    available_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
    selected_file = st.selectbox("Выберите файл для анализа:", available_files)

if selected_file:
    with st.spinner("Обработка последовательностей ДНК..."):
        df = pd.read_table(os.path.join(data_folder, selected_file))
        features_df = extract_sequence_features(df)
        st.success(f"Обработано {len(df)} последовательностей")

    # Основная статистика
    st.header("Основные характеристики")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Распределение по классам")
        class_counts = features_df['class'].value_counts().reset_index()
        class_counts.columns = ['Класс', 'Количество']
        fig_class = px.bar(class_counts, x='Класс', y='Количество')
        st.plotly_chart(fig_class)
    
    with col2:
        st.subheader("GC-состав")
        fig_gc = px.box(features_df, x='class', y='gc_content', 
                       labels={'class': 'Класс', 'gc_content': 'GC-состав (%)'})
        st.plotly_chart(fig_gc)

    # Анализ нуклеотидов
    st.header("Анализ нуклеотидного состава")
    nucleotide_data = features_df.groupby('class')[['a_freq', 't_freq', 'g_freq', 'c_freq']].mean()
    nucleotide_data.columns = ['A', 'T', 'G', 'C']
    
    fig_nuc = px.imshow(nucleotide_data, 
                        labels=dict(x='Нуклеотид', y='Класс', color='Частота'),
                        color_continuous_scale='Viridis')
    st.plotly_chart(fig_nuc)

    # PCA анализ
    st.header("Анализ главных компонент")
    features_for_pca = ['length', 'gc_content', 'a_freq', 't_freq', 'g_freq', 'c_freq', 'cpg_count']
    
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaler.fit_transform(features_df[features_for_pca]))
    
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Класс': features_df['class']
    })
    
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Класс',
                        title=f'PCA (объясненная дисперсия: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})')
    st.plotly_chart(fig_pca)

    # K-мер анализ
    st.header("Анализ k-меров")
    k = st.slider("Выберите размер k-мера", min_value=2, max_value=8, value=3, step=1)
    
    selected_class = st.selectbox(
        "Выберите класс для анализа k-меров:",
        options=sorted(df['class'].unique()),
        key="kmer_class_select"
    )
    
    class_seqs = df[df['class'] == selected_class]['sequence'].values
    kmer_results = analyze_kmers(class_seqs, k)
    
    # Визуализация k-меров
    fig_kmer = px.bar(
        kmer_results,
        x='k-мер',
        y='частота',
        title=f'Топ-20 {k}-меров для класса {selected_class}'
    )
    st.plotly_chart(fig_kmer)
    st.dataframe(kmer_results)

    # Экспорт данных
    st.header("Экспорт результатов")
    st.markdown(get_csv_download_link(features_df, "результаты_анализа.csv"), unsafe_allow_html=True)

else:
    st.info("Загрузите файл с последовательностями ДНК для начала анализа.")
    st.subheader("Пример формата файла")
    sample_format = pd.DataFrame({
        'sequence': ['ATGCCCCAACTAAATACTACCGTATGGCC', 'ATGAACGAAAATCTGTTCGCTTCATTCAT'],
        'class': [0, 1]
    })
    st.dataframe(sample_format)

# Справка
st.sidebar.header("Справка")
st.sidebar.markdown("""
### Нуклеотиды ДНК:
- **A (Аденин)**: Пуриновое основание, образует пару с T
- **T (Тимин)**: Пиримидиновое основание, образует пару с A
- **G (Гуанин)**: Пуриновое основание, образует пару с C
- **C (Цитозин)**: Пиримидиновое основание, образует пару с G

### Классификация участков ДНК:
- **Класс 0**: Экзоны - кодирующие участки генов, непосредственно определяющие последовательность белка
- **Класс 1**: Интроны - некодирующие участки генов, удаляемые при созревании РНК
- **Класс 2**: Промоторы - регуляторные участки, контролирующие начало транскрипции
- **Класс 3**: Энхансеры - усилители транскрипции, регулирующие активность генов
- **Класс 4**: CpG-островки - участки с высоким содержанием CG, часто связанные с регуляцией генов
- **Класс 5**: Повторяющиеся элементы - мобильные генетические элементы и другие повторы
- **Класс 6**: Межгенные участки - некодирующие области между генами

### О программе:
Инструмент для комплексного анализа структурных особенностей и паттернов в последовательностях ДНК
""")