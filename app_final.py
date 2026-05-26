# ============================================
# ІМПОРТ БІБЛІОТЕК
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

np.random.seed(42)

# ============================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ============================================

st.set_page_config(
    page_title="Product Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Product Clustering Dashboard")
st.markdown("Аналіз та сегментація товарів за допомогою різних методів кластеризації")

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("⚙️ Налаштування")

uploaded_file = st.sidebar.file_uploader(
    "Завантажте датасет",
    type=["csv", "xlsx"]
)

# ============================================
# ФУНКЦІЇ (З КЕШУВАННЯМ)
# ============================================

@st.cache_data
def load_data(uploaded_file, selected_sheets=None):
    """
    Кешована функція для завантаження та об'єднання обраних аркушів.
    Параметр selected_sheets приймає список назв аркушів, які обрав користувач.
    """
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        # Якщо користувач нічого не обрав (на старті), беремо перший аркуш
        if not selected_sheets:
            return pd.read_excel(uploaded_file, sheet_name=0)
            
        df_list = []
        for sheet in selected_sheets:
            sheet_df = pd.read_excel(uploaded_file, sheet_name=sheet)
            # Додаємо технічну領колонку, щоб знати джерело рядків
            sheet_df['Excel_Sheet_Source'] = sheet
            df_list.append(sheet_df)
            
        # Об'єднуємо всі обрані аркуші в один великий DataFrame
        df = pd.concat(df_list, ignore_index=True)
        return df
    else:
        # Якщо завантажили звичайний CSV
        return pd.read_csv(uploaded_file)
    
@st.cache_data
def preprocess_data(df):
    cleaned_df = df.copy()
    cleaned_df["Invoice"] = cleaned_df["Invoice"].astype("str")

    mask = (cleaned_df["Invoice"].str.match("^\\d{6}$") == True)

    cleaned_df = cleaned_df[mask]
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype("str")

    mask = (
        (cleaned_df["StockCode"].str.match("^\\d{5}$") == True)
        | (cleaned_df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == True)
        | (cleaned_df["StockCode"].str.match("^PADS$") == True)
    )

    cleaned_df = cleaned_df[mask]

    cleaned_df.dropna(subset=["Customer ID"], inplace=True)

    cleaned_df = cleaned_df[cleaned_df["Price"] > 0.0]
    cleaned_df = cleaned_df[cleaned_df["Quantity"] > 0.0]
    cleaned_df["TotalSales"] = cleaned_df["Quantity"] * cleaned_df["Price"]
    aggregated_df = cleaned_df.groupby(by=["StockCode"], as_index=False) \
    .agg(
        Description=("Description", "first"),     # Беремо перший доступний опис для цього коду
        Quantity=("Quantity", "sum"),             # Загальний обсяг продажів товару
        MonetaryValue=("TotalSales", "sum"),      # Загальний виторг від товару
        Frequency=("Invoice", "nunique"),         # У скількох унікальних транзакціях брав участь
        LastInvoiceDate=("InvoiceDate", "max")    # Дата останнього продажу цього товару
    )
    max_invoice_date = aggregated_df["LastInvoiceDate"].max()
    aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days
    M_Q1 = aggregated_df["MonetaryValue"].quantile(0.25)
    M_Q3 = aggregated_df["MonetaryValue"].quantile(0.75)
    M_IQR = M_Q3 - M_Q1

    monetary_outliers_df = aggregated_df[(aggregated_df["MonetaryValue"] > (M_Q3 + 1.5 * M_IQR)) | (aggregated_df["MonetaryValue"] < (M_Q1 - 1.5 * M_IQR))].copy()
    F_Q1 = aggregated_df['Frequency'].quantile(0.25)
    F_Q3 = aggregated_df['Frequency'].quantile(0.75)
    F_IQR = F_Q3 - F_Q1

    frequency_outliers_df = aggregated_df[(aggregated_df['Frequency'] > (F_Q3 + 1.5 * F_IQR)) | (aggregated_df['Frequency'] < (F_Q1 - 1.5 * F_IQR))].copy()
    non_outliers_df = aggregated_df[(~aggregated_df.index.isin(monetary_outliers_df.index)) & (~aggregated_df.index.isin(frequency_outliers_df.index))]


    # Масштабування
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])

    return X_scaled, non_outliers_df

@st.cache_data
def calculate_elbow(X_scaled, max_k=10):
    # Зміна: Семплювання видалено. Розрахунок йде по повному масиву X_scaled
    inertias = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=5)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    return inertias


def run_clustering(method, X_scaled, params):
    # Зміна: Обмеження на Agglomerative кластеризацію повністю видалено

    if method == 'K-Means':
        model = KMeans(
            n_clusters=params['n_clusters'],
            random_state=42,
            n_init=10
        )
    elif method == 'DBSCAN':
        model = DBSCAN(
            eps=params['eps'],
            min_samples=params['min_samples']
        )
    elif method == 'Agglomerative':
        model = AgglomerativeClustering(
            n_clusters=params['n_clusters'],
            linkage=params['linkage']
        )

    with st.spinner(f"Обчислення кластеризації методом {method}... Зачекайте."):
        labels = model.fit_predict(X_scaled)

    return model, labels


def calculate_silhouette(X_scaled, labels):
    unique_labels = set(labels)

    if len(unique_labels) <= 1:
        return None

    if -1 in unique_labels:
        mask = labels != -1
        if len(set(labels[mask])) <= 1:
            return None
        X_eval = X_scaled[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X_scaled
        labels_eval = labels

    # Зміна: Семплювання видалено. Silhouette Score рахується для всіх точок X_eval
    with st.spinner("Розрахунок Silhouette Score..."):
        score = silhouette_score(X_eval, labels_eval)

    return score


# ============================================
# ОСНОВНА ЧАСТИНА
# ============================================

if uploaded_file is not None:
    # 1. Якщо це Excel — спочатку зчитуємо лише імена аркушів (це миттєво)
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        with st.spinner("Аналіз структури Excel файлу..."):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            sheets_count = len(sheet_names)
        
        st.sidebar.success(f"📊 Знайдено Excel-аркушів: {sheets_count}")
        
        # Якщо в Excel файлі більше ніж 1 аркуш — показуємо мультивибір
        if sheets_count > 1:
            selected_sheets = st.sidebar.multiselect(
                "Оберіть аркуші для аналізу (можна декілька)", 
                options=sheet_names,
                default=[sheet_names[0]]  # За замовчуванням активний перший аркуш
            )
        else:
            selected_sheets = [sheet_names[0]]
            
        # Захист: якщо користувач видалив усі аркуші з поля вибору
        if not selected_sheets:
            st.error("❌ Будь ласка, оберіть хоча б один аркуш у меню ліворуч для продовження.")
            st.stop()
            
        # Викликаємо нашу кешовану функцію, передаючи туди список обраних сторінок
        with st.spinner("Завантаження та об'єднання обраних аркушів..."):
            df = load_data(uploaded_file, selected_sheets)
            
    else:
        # Якщо це звичайний CSV файл, просто викликаємо функцію
        with st.spinner("Завантаження CSV файлу..."):
            df = load_data(uploaded_file)

    # Сповіщення про успіх з відображенням фінальної кількості рядків
    st.success(f"✅ Дані успішно завантажено! Рядків після об'єднання: {df.shape[0]}")

    # ============================================
    # ІНФОРМАЦІЯ ПРО ПОЧАТКОВИЙ ДАТАСЕТ
    # ============================================

    st.header("📁 Інформація про завантажений датасет")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Кількість транзакцій (рядків)", df.shape[0])

    with col2:
        st.metric("Кількість початкових колонок", df.shape[1])

    with col3:
        st.metric(
            "Числові колонки",
            len(df.select_dtypes(include=np.number).columns)
        )

    st.subheader("Перші рядки сирого датасету")
    st.dataframe(df.head())

    # ============================================
    # PREPROCESSING ТА АГРЕГАЦІЯ ТОВАРІВ
    # ============================================
    X_scaled, X_original = preprocess_data(df)

    # Покажемо користувачу інформацію про агреговані дані
    st.header("📦 Дані після агрегації по товарах (StockCode)")
    st.info(f"Після очищення від викидів та групування залишилось унікальних товарів: **{X_original.shape[0]}**")
    st.dataframe(X_original.head())

    # ============================================
    # ВИБІР МЕТОДУ
    # ============================================

    st.sidebar.header("📌 Метод кластеризації")

    method = st.sidebar.selectbox(
        "Оберіть метод",
        ['K-Means', 'DBSCAN', 'Agglomerative']
    )

    # ============================================
    # ПАРАМЕТРИ МОДЕЛІ
    # ============================================

    st.header("⚙️ Параметри кластеризації")

    params = {}

    if method == 'K-Means':
        params['n_clusters'] = st.slider('Кількість кластерів', 2, 10, 3)
    elif method == 'DBSCAN':
        params['eps'] = st.slider('eps', 0.1, 5.0, 0.5, 0.1)
        params['min_samples'] = st.slider('min_samples', 2, 20, 5)
    elif method == 'Agglomerative':
        params['n_clusters'] = st.slider('Кількість кластерів', 2, 10, 3)
        params['linkage'] = st.selectbox('Linkage', ['ward', 'complete', 'average', 'single'])

    calc_silh = st.checkbox("Розраховувати Silhouette Score (уповільнює роботу на великих даних)", value=True)


   # ============================================
    # МЕТОДИ ОПТИМІЗАЦІЇ ПАРАМЕТРІВ (ELBOW / K-DISTANCE / DENDROGRAM)
    # ============================================
    
    if method == 'K-Means':
        st.header("📈 Elbow Method")
        with st.expander("Показати графік ліктя (Elbow Method) для K-Means"):
            max_k = st.slider("Максимальна кількість кластерів", 3, 15, 10, key="elbow_k")
            with st.spinner("Завантаження графіка ліктя..."):
                inertias = calculate_elbow(X_scaled, max_k)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(2, max_k + 1), inertias, marker='o', color='#1f77b4')
            ax.set_xlabel('Кількість кластерів')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method')
            st.pyplot(fig)
            st.info("📌 Точка згину (ліктя) графіка допомагає визначити оптимальну кількість кластерів")

    elif method == 'DBSCAN':
        st.header("📈 Визначення оптимального eps (K-Distance Plot)")
        with st.expander("Показати K-Distance графік для DBSCAN", expanded=True):
            with st.spinner("Обчислення K-Distance графіку..."):
                from sklearn.neighbors import NearestNeighbors
                
                min_samples = params.get('min_samples', 5)
                neighbors = NearestNeighbors(n_neighbors=min_samples)
                neighbors_fit = neighbors.fit(X_scaled)
                
                distances, indices = neighbors_fit.kneighbors(X_scaled)
                distances = np.sort(distances, axis=0)
                distances = distances[:, min_samples - 1]
                
                fig_eps, ax_eps = plt.subplots(figsize=(8, 4))
                ax_eps.plot(distances, color='#2ca02c', label='K-distances')
                
                current_eps = params.get('eps', 0.5)
                ax_eps.axhline(y=current_eps, color='r', linestyle='--', alpha=0.8, 
                               label=f'Обраний у сидбарі eps ({current_eps})')
                
                ax_eps.set_title('K-Distance Plot with Elbow Detection', fontsize=12, fontweight='bold')
                ax_eps.set_xlabel('Точки даних, відсортовані за відстанню', fontsize=10)
                ax_eps.set_ylabel(f'{min_samples}-th Nearest Neighbor Distance (eps)', fontsize=10)
                ax_eps.grid(True, linestyle='--', alpha=0.6)
                ax_eps.legend(fontsize=10)
                
                st.pyplot(fig_eps)
                st.info("📌 **Як читати графік:** Оптимальне значення `eps` знаходиться в точці максимального згину "
                        "(найвищого підняття кривої перед різким вертикальним злетом).")

    elif method == 'Agglomerative':
            st.header("🌳 Ієрархічна структура даних (Дендрограма)")
            with st.expander("Показати дендрограму (Hierarchical Tree)", expanded=True):
                current_linkage = params.get('linkage', 'ward')
                
                with st.spinner(f"Обчислення матриці зв'язків за методом '{current_linkage}' для ВСІХ товарів..."):
                    from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
                    import matplotlib
                    
                    X_dendro_full = X_scaled

                    linkage_matrix = linkage(X_dendro_full, method=current_linkage)
                    
                    target_clusters = params.get('n_clusters', 3)
                    set2_palette = sns.color_palette("bone", target_clusters)
                    hex_colors = [matplotlib.colors.to_hex(rgb) for rgb in set2_palette]
                    
                    set_link_color_palette(hex_colors)
                    
                    if target_clusters > 1 and len(linkage_matrix) >= target_clusters:
                        threshold = linkage_matrix[-target_clusters + 1, 2]
                    else:
                        threshold = 0.7 * max(linkage_matrix[:, 2])

                    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
                    
                    dendrogram(
                        linkage_matrix,
                        truncate_mode='lastp',
                        p=12,
                        show_contracted=True,
                        color_threshold=threshold,
                        ax=ax_dendro
                    )
                    
                    ax_dendro.set_title(f"Дендрограма об’єднання товарів (Метод зв'язку: '{current_linkage}')", 
                                        fontsize=12, fontweight='bold')
                    ax_dendro.set_xlabel('Індекси сформованих суб-кластерів (або кількість точок у них)', fontsize=10)
                    ax_dendro.set_ylabel('Евклідова відстань об’єднання', fontsize=10)
                    ax_dendro.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    st.pyplot(fig_dendro)
                    
                    set_link_color_palette(None)
                    
                    st.info(f"📌 **Примітка:** Графік побудовано на основі 100% обсягу очищених даних. "
                            f"При виборі кількості кластерів та методу зв'язку '{current_linkage}' у меню ліворуч, "
                            f"структура та кольори дерева оновлюються автоматично.")
        
    # ============================================
    # ЗАПУСК МОДЕЛІ
    # ============================================

    if 'run_clicked' not in st.session_state:
        st.session_state.run_clicked = False

    if st.button("🚀 Запустити кластеризацію"):
        st.session_state.run_clicked = True
        model, labels = run_clustering(method, X_scaled, params)
        if labels is not None:
            st.session_state.labels = labels
            st.session_state.model = model
        else:
            st.session_state.run_clicked = False

    if st.session_state.run_clicked and 'labels' in st.session_state:
        labels = st.session_state.labels
        
        df_result = X_original.copy()
        df_result['Cluster'] = labels

        # ============================================
        # SILHOUETTE SCORE
        # ============================================
        st.header("📊 Метрики")
        
        if calc_silh:
            score = calculate_silhouette(X_scaled, labels)
            if score is not None:
                st.metric("Silhouette Score", round(score, 3))
                if score > 0.5:
                    st.success("Хороша якість кластеризації")
                elif score > 0.25:
                    st.warning("Середня якість кластеризації")
                else:
                    st.error("Слабке розділення кластерів")
            else:
                st.warning("Silhouette Score неможливо обчислити")
        else:
            st.info("Розрахунок Silhouette Score вимкнено користувачем.")

        # ============================================
        # PAIRPLOT ВІЗУАЛІЗАЦІЯ (ЗАМІСТЬ PCA)
        # ============================================
        st.header("🎨 Матриця розподілу кластерів (Pairplot)")

        with st.spinner("Побудова матриці Pairplot... Зачекайте кілька секунд."):
            pairplot_df = df_result[[
                'MonetaryValue',
                'Frequency',
                'Recency',
                'Cluster'
            ]].copy()

            # Зміна: Блок `if pairplot_df.shape[0] > 5000:` із df.sample() повністю видалено.
            # Тепер Seaborn будує графік по всьому датасету.

            pairplot_df['Cluster'] = pairplot_df['Cluster'].astype(str)
            sns.set_theme(style="ticks")
            
            grid = sns.pairplot(
                pairplot_df,
                hue='Cluster',
                diag_kind='kde',
                palette='Set2',
                plot_kws={'alpha': 0.6, 'edgecolor': 'none', 's': 20}
            )
            
            grid.fig.suptitle(
                f"Взаємозв'язки між RFM-метриками для {method}", 
                y=1.02, 
                fontsize=14, 
                fontweight='bold'
            )

            st.pyplot(grid.fig)

        # ============================================
        # 3D SCATTER PLOT ВІЗУАЛІЗАЦІЯ
        # ============================================
        st.header("🔮 3D Простір метрик RFM за кластерами")

        with st.spinner("Побудова 3D-графіка..."):
            fig_3d = plt.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(projection='3d')

            unique_clusters = sorted(list(set(labels)))
            base_palette = sns.color_palette("Set2", len(unique_clusters))
            cluster_colors = {str(clust): base_palette[i] for i, clust in enumerate(unique_clusters)}
            
            if '-1' in cluster_colors:
                cluster_colors['-1'] = (0.5, 0.5, 0.5) 

            labels_str = df_result['Cluster'].astype(str)
            colors_mapped = labels_str.map(cluster_colors)

            scatter_3d = ax_3d.scatter(
                df_result['MonetaryValue'], 
                df_result['Frequency'], 
                df_result['Recency'], 
                c=colors_mapped,
                marker='o',
                alpha=0.7,
                edgecolors='w',
                linewidth=0.3,
                s=35
            )

            ax_3d.set_xlabel('Monetary Value (Виторг)', fontsize=10, labelpad=10)
            ax_3d.set_ylabel('Frequency (Частота)', fontsize=10, labelpad=10)
            ax_3d.set_zlabel('Recency (Давність)', fontsize=10, labelpad=10)
            ax_3d.set_title(f'3D Scatter Plot: Розподіл товарів ({method})', fontsize=12, fontweight='bold', pad=15)

            from matplotlib.lines import Line2D
            
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=f'Кластер {clust}',
                       markerfacecolor=cluster_colors[str(clust)], markersize=10)
                for clust in unique_clusters if str(clust) != '-1'
            ]
            
            if '-1' in cluster_colors:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Шум (Аномалії)',
                                       markerfacecolor=cluster_colors['-1'], markersize=10))
                
            ax_3d.legend(handles=legend_elements, title="Сегменти", loc="upper left", bbox_to_anchor=(1.05, 1))
            ax_3d.view_init(elev=20, azim=135)

            st.pyplot(fig_3d)

        # ============================================
        # VIOLIN PLOTS ВІЗУАЛІЗАЦІЯ
        # ============================================
        st.header("🎻 Розподіл метрик за кластерами (Violin Plots)")

        with st.spinner("Побудова графіків розподілу Violin Plots..."):
            fig_violin, axes = plt.subplots(3, 1, figsize=(10, 14))

            violin_df = df_result.copy()
            violin_df['Cluster'] = violin_df['Cluster'].astype(str)

            unique_clusters_str = sorted(violin_df['Cluster'].unique())
            if 'cluster_colors' not in locals():
                base_palette = sns.color_palette("Set2", len(unique_clusters_str))
                cluster_colors = {clust: base_palette[i] for i, clust in enumerate(unique_clusters_str)}
                if '-1' in cluster_colors:
                    cluster_colors['-1'] = (0.5, 0.5, 0.5)

            sns.violinplot(
                data=violin_df,
                x='Cluster',
                y='MonetaryValue',
                palette=cluster_colors,
                hue='Cluster',
                legend=False,
                ax=axes[0]
            )
            axes[0].set_title('Monetary Value (Виторг) за кластерами', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Monetary Value')
            axes[0].set_xlabel('Кластер')

            sns.violinplot(
                data=violin_df,
                x='Cluster',
                y='Frequency',
                palette=cluster_colors,
                hue='Cluster',
                legend=False,
                ax=axes[1]
            )
            axes[1].set_title('Frequency (Частота) за кластерами', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Frequency')
            axes[1].set_xlabel('Кластер')

            sns.violinplot(
                data=violin_df,
                x='Cluster',
                y='Recency',
                palette=cluster_colors,
                hue='Cluster',
                legend=False,
                ax=axes[2]
            )
            axes[2].set_title('Recency (Давність) за кластерами', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Recency')
            axes[2].set_xlabel('Кластер')

            plt.tight_layout()
            st.pyplot(fig_violin)

        # ============================================
        # РОЗПОДІЛ КЛАСТЕРІВ
        # ============================================
        st.header("📦 Розподіл товарів по кластерах")

        with st.spinner("Побудова діаграми розподілу..."):
            cluster_counts = df_result['Cluster'].value_counts().sort_index()
            
            counts_df = pd.DataFrame({
                'Кластер': cluster_counts.index.astype(str),
                'Кількість товарів': cluster_counts.values
            })

            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            
            unique_clusters_str = sorted(df_result['Cluster'].astype(str).unique())
            if 'cluster_colors' not in locals():
                base_palette = sns.color_palette("Set2", len(unique_clusters_str))
                cluster_colors = {clust: base_palette[i] for i, clust in enumerate(unique_clusters_str)}
                if '-1' in cluster_colors:
                    cluster_colors['-1'] = (0.5, 0.5, 0.5)

            sns.barplot(
                data=counts_df,
                x='Кластер',
                y='Кількість товарів',
                hue='Кластер',
                palette=cluster_colors,
                legend=False,
                ax=ax3
            )
            
            for p in ax3.patches:
                if p.get_height() > 0:
                    ax3.annotate(
                        f"{int(p.get_height())}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 8), 
                        textcoords='offset points',
                        fontsize=10, fontweight='bold'
                    )

            ax3.set_xlabel('Кластер', fontsize=10)
            ax3.set_ylabel('Кількість товарів', fontsize=10)
            ax3.set_title('Розмір отриманих кластерів (кількість товарів у них)', fontsize=12, fontweight='bold')
            ax3.grid(axis='y', linestyle='--', alpha=0.5)
            ax3.set_ylim(0, counts_df['Кількість товарів'].max() * 1.15)

            st.pyplot(fig3)

        # ============================================
        # ВКЛАДКИ З КЛАСТЕРАМИ
        # ============================================
        st.header("📂 Товари по кластерах")

        available_clusters = sorted(df_result['Cluster'].unique())
        tabs = st.tabs([f"Кластер {cluster}" for cluster in available_clusters])

        for tab, cluster in zip(tabs, available_clusters):
            with tab:
                cluster_data = df_result[df_result['Cluster'] == cluster].copy()
                st.subheader(f"Товари кластеру {cluster}")

                description_col = next((c for c in ['Description', 'description', 'product_description', 'ProductName', 'product_name'] if c in cluster_data.columns), None)
                product_id_col = next((c for c in ['StockCode', 'product_id', 'ProductID', 'id', 'product_code'] if c in cluster_data.columns), None)

                display_columns = []
                if product_id_col: display_columns.append(product_id_col)
                if description_col: display_columns.append(description_col)
                
                for col in ["MonetaryValue", "Frequency", "Recency", "Quantity"]:
                    if col in cluster_data.columns:
                        display_columns.append(col)

                if not display_columns: 
                    display_columns = cluster_data.columns.tolist()

                st.dataframe(
                    cluster_data[display_columns].reset_index(drop=True),
                    use_container_width=True
                )
                st.write(f"Всього товарів у кластері: {len(cluster_data)}")

        # ============================================
        # ОПИС КЛАСТЕРІВ
        # ============================================
        st.header("🧠 Характеристика кластерів")

        numeric_cols = ["MonetaryValue", "Frequency", "Recency", "Quantity"]
        numeric_cols = [col for col in numeric_cols if col in df_result.columns]

        cluster_summary = df_result.groupby('Cluster')[numeric_cols].mean()
        st.dataframe(cluster_summary.style.format("{:.2f}"))

        # ============================================
        # МОЖЛИВІ ВИСНОВКИ
        # ============================================
        st.header("📌 Інтерпретація")
        st.markdown("Запустіть кластеризацію для генерації фінальних аналітичних інсайтів.")