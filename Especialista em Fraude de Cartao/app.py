# app.py
# -*- coding: utf-8 -*-
import os
import io
import re
import zipfile
import textwrap
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json  # Memória persistente em arquivo

# ====== OpenAI via LangChain (para INSIGHTS textuais com memória) ======
LLM_AVAILABLE = True
try:
    from langchain_openai import ChatOpenAI  # versões novas
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI  # fallback p/ versões antigas # type: ignore
        from langchain.prompts import PromptTemplate  # type: ignore
        from langchain.chains import LLMChain  # type: ignore
    except Exception:
        LLM_AVAILABLE = False

# ====== PDF backends (reportlab -> fpdf2) ======
PDF_BACKEND = None
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    PDF_BACKEND = "reportlab"
except Exception:
    try:
        from fpdf import FPDF  # pip install fpdf2
        PDF_BACKEND = "fpdf2"
    except Exception:
        PDF_BACKEND = None  # sem PDF

# ====== MEMÓRIA PERSISTENTE (JSON por usuário) ======
MEMORY_DIR = os.path.join(os.getcwd(), "memories")
os.makedirs(MEMORY_DIR, exist_ok=True)

def _slugify_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "", (name or "").strip().lower().replace(" ", "_")) or "anonimo"

def _mem_path(user_name: str) -> str:
    return os.path.join(MEMORY_DIR, f"{_slugify_name(user_name)}.json")

def _json_fallback(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

def load_user_memory(user_name: str):
    try:
        with open(_mem_path(user_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("qa_history", [])
    except Exception:
        return []

def save_user_memory(user_name: str, qa_history: list):
    try:
        with open(_mem_path(user_name), "w", encoding="utf-8") as f:
            json.dump({"qa_history": qa_history}, f, ensure_ascii=False, indent=2, default=_json_fallback)
    except Exception as e:
        st.warning(f"Não foi possível salvar a memória: {e}")

def clear_user_memory(user_name: str):
    try:
        p = _mem_path(user_name)
        if os.path.exists(p):
            os.remove(p)
    except Exception as e:
        st.warning(f"Não foi possível apagar a memória: {e}")

# ====== TEXTO-PADRÃO: "Framework escolhida" (usado na capa do PDF) ======
DEFAULT_FRAMEWORK_TEXT = (
    "Frontend: Streamlit. Análises: Pandas/NumPy/Matplotlib/Seaborn. "
    "IA: LangChain + OpenAI (ChatOpenAI) para insights detalhados, com memória por usuário (JSON em 'memories/'). "
    "PDF: reportlab (preferencial) ou fpdf2 (fallback)."
)

# ====== CONFIG UI ======
st.set_page_config(page_title="Agente EDA para CSV (I2A2)", page_icon="🤖", layout="wide")
st.title("🤖 Agente EDA para CSV — I2A2")
st.caption("Pergunte em linguagem natural. O app detecta a intenção, calcula métricas no CSV, gera gráficos profissionais e produz insights detalhados em PT-BR com memória cumulativa.")

# Sugestões de perguntas no início
st.markdown("""
### Sugestões de Perguntas para Começar:
- Qual é a distribuição da variável 'Amount'? Gere um histograma.
- Existem padrões temporais em 'Time' vs 'Class'?
- Quais as correlações entre variáveis V1-V5 e 'Class'?
- Detecte outliers em 'Amount' e sugira remoção.
- Quais conclusões gerais sobre fraudes no dataset?
""")

# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Configurações")

    def _default_api_key():
        env = os.getenv("OPENAI_API_KEY", "")
        project_secrets = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
        home_secrets = os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")
        if os.path.exists(project_secrets) or os.path.exists(home_secrets):
            try:
                return st.secrets.get("OPENAI_API_KEY", env)
            except Exception:
                return env
        return env

    api_key = st.text_input(
        "OpenAI API Key",
        value=_default_api_key(),
        type="password",
        help="Use .streamlit/secrets.toml (OPENAI_API_KEY) ou variável de ambiente.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model_name = st.selectbox("Modelo para INSIGHTS", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)

    st.markdown("---")
    st.markdown("### 👤 Identificação do Usuário")
    user_name = st.text_input(
        "Seu nome (obrigatório na primeira vez)",
        key="user_name",
        help="Serve para personalizar e persistir a memória do relatório."
    )

    col_mem1, col_mem2 = st.columns(2)
    with col_mem1:
        load_clicked = st.button("🔄 Carregar memória")
    with col_mem2:
        clear_clicked = st.button("🧹 Limpar memória")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "loaded_user" not in st.session_state:
        st.session_state.loaded_user = None

    if load_clicked and user_name.strip():
        st.session_state.qa_history = load_user_memory(user_name)
        st.session_state.loaded_user = user_name.strip()
        st.success(f"Memória carregada para: {user_name}")

    if clear_clicked:
        if user_name.strip():
            clear_user_memory(user_name)
        st.session_state.qa_history = []
        st.success("Memória limpa. Seu próximo PDF virá apenas com o que for feito a partir de agora.")

    st.markdown("---")
    sample_n = st.number_input("Amostragem (0 = usar tudo)", min_value=0, value=0, step=10000)
    show_head = st.checkbox("Mostrar head() do DataFrame", value=True)
    st.markdown("---")
    uploaded = st.file_uploader("Faça upload do CSV ou ZIP (ex.: Kaggle - Credit Card Fraud.zip)", type=["csv", "zip"])
    st.markdown("---")
    st.caption("Framework: Streamlit + Pandas + Matplotlib/Seaborn + LangChain/OpenAI")
    st.caption("Estrutura: Roteador de intenções → Métricas/Gráficos → Insights detalhados → PDF")

# ====== LEITURA DE DADOS ======
@st.cache_data(show_spinner=False)
def load_csv_from_zip(file_bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    return pd.read_csv(f)
    raise ValueError("ZIP não contém um CSV.")

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    if file.name.lower().endswith(".zip"):
        return load_csv_from_zip(file.getvalue())
    else:
        return pd.read_csv(file)

df = None
if uploaded:
    try:
        df = load_csv(uploaded)
        if sample_n and sample_n > 0 and len(df) > sample_n:
            df = df.sample(sample_n, random_state=42).reset_index(drop=True)
        st.success(f"Arquivo carregado: **{uploaded.name}** — shape: {df.shape}")
        if show_head:
            st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Falha ao ler o arquivo: {e}")
elif os.path.exists("creditcard.csv"):
    try:
        df = pd.read_csv("creditcard.csv")
        st.info("Usando creditcard.csv local (fallback).")
        if show_head:
            st.dataframe(df.head(), use_container_width=True)
    except Exception:
        pass

# ====== UTILIDADES ======
def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def find_columns_in_text(text: str, df: pd.DataFrame):
    norm_text = normalize(text or "")
    col_map = {normalize(c): c for c in df.columns}
    return [orig for ncol, orig in col_map.items() if ncol and ncol in norm_text]

def pick_numeric(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()

def pick_categorical(df: pd.DataFrame):
    nums = set(pick_numeric(df))
    return [c for c in df.columns if c not in nums]

def dataset_profile_text(df: pd.DataFrame, spec=None) -> str:
    parts = []
    parts.append(f"shape={df.shape[0]} linhas x {df.shape[1]} colunas")
    num_cols = pick_numeric(df); cat_cols = pick_categorical(df)
    parts.append(f"colunas_numericas({len(num_cols)}): {', '.join(num_cols[:8])}{'...' if len(num_cols)>8 else ''}")
    parts.append(f"colunas_categoricas({len(cat_cols)}): {', '.join(cat_cols[:8])}{'...' if len(cat_cols)>8 else ''}")
    na = df.isna().mean().sort_values(ascending=False).head(8)
    if na.max() > 0:
        parts.append("nulos_top8(%): " + ", ".join([f"{c}={round(v*100,2)}%" for c, v in na.items() if v > 0]))
    if "Class" in df.columns:
        cls = pd.to_numeric(df["Class"], errors="coerce")
        vc = cls.value_counts().to_dict()
        total = int(cls.notna().sum())
        pos = int(vc.get(1, 0)); neg = int(vc.get(0, 0))
        rate = (pos/total) if total else 0.0
        parts.append(f"balance_Class: 1={pos}, 0={neg}, taxa_fraude={rate:.4f}")
    if spec and getattr(spec, "cols", None):
        parts.append("colunas_escolhidas: " + ", ".join(spec.cols))
    return "\n".join(parts)

# ====== ROTEADOR ======
class VizSpec:
    def __init__(self, kind: str, cols=None, params=None, reason=""):
        self.kind = kind  # 'temporal', 'hist', 'corr', 'box', 'bar', 'scatter', 'class_summary', 'pie', 'auto'
        self.cols = cols or []
        self.params = params or {}
        self.reason = reason

def route_query_to_viz(q: str, df: pd.DataFrame) -> VizSpec:
    ql = (q or "").lower()
    found_cols = find_columns_in_text(ql, df)
    num_cols = pick_numeric(df); cat_cols = pick_categorical(df)
    if any(k in ql for k in ["tempo", "temporais", "janela", "rolling", "linha", "time series", "time", "sazonal", "evolucao"]):
        c_time = "Time" if "Time" in df.columns else next((c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)), None)
        if c_time:
            return VizSpec("temporal", [c_time], {"use_class": ("class" in ql or "fraud" in ql or "fraude" in ql or "Class" in df.columns)},
                           reason="Pergunta temporal detectada (ex.: padrões em fraudes ao longo do tempo).")
    if any(k in ql for k in ["correl", "heatmap", "matriz", "relacao", "associacao", "influencia"]):
        cols = [c for c in found_cols if c in num_cols]
        if "Class" in df.columns and "Class" not in cols: cols.append("Class")
        return VizSpec("corr", cols or num_cols[:], reason="Pedido de correlação/heatmap (incluindo com Class).")
    if any(k in ql for k in ["outlier", "atipic", "anomalia", "boxplot", "box plot", "remover", "transformar"]):
        cols = [c for c in found_cols if c in num_cols][:6] or num_cols[:6]
        return VizSpec("box", cols, reason="Pedido de outliers/anomalias.")
    if any(k in ql for k in ["distribu", "histogr", "densidade", "kde", "tendencia central", "variancia", "desvio", "padrao"]):
        cols = [c for c in found_cols if c in num_cols][:2]
        if not cols:
            if "Amount" in df.columns: cols = ["Amount"]
            elif num_cols: cols = [num_cols[0]]
        return VizSpec("hist", cols, reason="Pedido de distribuição/histograma.")
    if any(k in ql for k in ["frequencia", "categor", "barras", "contagem", "top", "mais comuns", "menos frequentes"]):
        cols = [c for c in found_cols if c in cat_cols][:1] or (cat_cols[:1] if cat_cols else [])
        if "Class" in df.columns and "Class" not in cols: cols.append("Class")
        return VizSpec("bar", cols, reason="Pedido de frequências/categorias.")
    if any(k in ql for k in ["dispers", "scatter", "relacao", "associacao"]):
        cols = [c for c in found_cols if c in num_cols][:2]
        if len(cols) < 2 and len(num_cols) >= 2: cols = num_cols[:2]
        return VizSpec("scatter", cols, reason="Pedido de relação entre variáveis.")
    if any(k in ql for k in ["class", "fraud", "fraude", "detec", "anomalia"]):
        cols = [c for c in ["Class", "Amount"] if c in df.columns]
        return VizSpec("class_summary", cols, reason="Pedido focado em fraude/Class.")
    return VizSpec("auto", reason="Roteador não encontrou match explícito — análise automática.")

# ====== GRÁFICOS + MÉTRICAS ======
def fig_to_bytes() -> bytes:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0); out = buf.read()
    plt.close("all")
    return out

def plot_temporal(df, time_col, use_class=False):
    metrics = {}
    s = df[time_col]
    if not np.issubdtype(s.dtype, np.datetime64):
        s = pd.to_numeric(s, errors="coerce"); mask = s.notna()
        span = float(s[mask].max() - s[mask].min())
        if span <= 0: raise ValueError("Coluna Time não possui variação suficiente.")
        bin_size = max(60.0, min(3600.0, span / 200.0))
        df2 = df.loc[mask, [time_col]].copy()
        df2["bin"] = (s[mask] // bin_size).astype(int) * bin_size
        plt.figure(figsize=(10, 6)); sns.set_style("whitegrid")
        if use_class and "Class" in df.columns:
            df2["Class"] = pd.to_numeric(df.loc[mask, "Class"], errors="coerce")
            agg = df2.groupby("bin")["Class"].mean().sort_index()
            sns.lineplot(x=agg.index, y=agg.values, color="red")
            plt.title("Taxa de Fraude por Janela de Tempo"); plt.xlabel("Tempo (segundos)"); plt.ylabel("Taxa de Fraude")
            metrics["media_taxa_fraude"] = float(agg.mean()); metrics["pico_taxa_fraude"] = float(agg.max())
        else:
            agg = df2.groupby("bin").size().sort_index()
            sns.lineplot(x=agg.index, y=agg.values, color="blue")
            plt.title("Transações por Janela de Tempo"); plt.xlabel("Tempo (segundos)"); plt.ylabel("Contagem")
            metrics["contagem_media_janela"] = float(agg.mean()); metrics["pico_contagem"] = int(agg.max())
        return fig_to_bytes(), metrics
    df2 = df.copy(); df2[time_col] = pd.to_datetime(df2[time_col], errors="coerce"); df2 = df2.dropna(subset=[time_col])
    if df2.empty: raise ValueError("Sem datas válidas após conversão.")
    df2 = df2.set_index(time_col).sort_index()
    total = (df2.index.max() - df2.index.min()).total_seconds()
    rule = f"{int(max(60, min(3600, total/200)))}S"
    plt.figure(figsize=(10, 6)); sns.set_style("whitegrid")
    if use_class and "Class" in df.columns:
        ser = pd.to_numeric(df2["Class"], errors="coerce").resample(rule).mean()
        sns.lineplot(x=ser.index, y=ser.values, color="red")
        plt.title("Taxa de Fraude por Janela de Tempo"); plt.ylabel("Taxa de Fraude")
        metrics["media_taxa_fraude"] = float(np.nanmean(ser.values)); metrics["pico_taxa_fraude"] = float(np.nanmax(ser.values))
    else:
        ser = df2.resample(rule).size()
        sns.lineplot(x=ser.index, y=ser.values, color="blue")
        plt.title("Transações por Janela de Tempo"); plt.ylabel("Contagem")
        metrics["contagem_media_janela"] = float(np.nanmean(ser.values)); metrics["pico_contagem"] = int(np.nanmax(ser.values))
    plt.xlabel("Tempo")
    return fig_to_bytes(), metrics

def plot_corr(df, cols=None):
    num = df.select_dtypes(include="number")
    if cols: num = num[cols]
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(10, 8)); sns.set_style("white")
    sns.heatmap(corr, annot=len(corr.columns) < 10, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Matriz de Correlação (V1-V28 PCA, Amount, Class)")
    corr_vals = corr.where(~np.eye(corr.shape[0], dtype=bool))
    top = corr_vals.abs().unstack().dropna().sort_values(ascending=False).head(5)
    metrics = {"top_correlacoes": [(str(a), str(b), float(v)) for (a, b), v in top.items()]}
    if "Class" in corr.columns:
        with_class = corr["Class"].drop("Class", errors="ignore").abs().sort_values(ascending=False).head(5)
        metrics["correl_com_Class_top5"] = [(str(k), float(v)) for k, v in with_class.items()]
    return fig_to_bytes(), metrics

def plot_hist(df, cols):
    col = cols[0] if cols else df.select_dtypes(include="number").columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    plt.figure(figsize=(8, 6)); sns.set_style("whitegrid")
    sns.histplot(s, bins=50, kde=True, color="skyblue")
    plt.title(f"Distribuição de {col}"); plt.xlabel(col); plt.ylabel("Frequência")
    metrics = {
        "coluna": col, "n": int(s.shape[0]), "media": float(s.mean()), "mediana": float(s.median()),
        "desvio_padrao": float(s.std(ddof=1)), "min": float(s.min()), "max": float(s.max()),
        "skewness": float(s.skew()), "kurtosis": float(s.kurt()),
    }
    return fig_to_bytes(), metrics

def plot_box(df, cols):
    use = cols if cols else df.select_dtypes(include="number").columns[:6]
    data = df[use].apply(pd.to_numeric, errors="coerce")
    use = list(data.columns)[:6]; data = data[use]
    plt.figure(figsize=(10, 6)); sns.set_style("whitegrid")
    sns.boxplot(data=data, orient="h", palette="Set2", showfliers=True)
    plt.title("Boxplots para Detecção de Outliers (V1-V28, Amount)"); plt.xlabel("Valores")
    out_counts = {}
    for c in data.columns:
        s = data[c].dropna()
        if s.empty: out_counts[c] = 0; continue
        q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_counts[c] = int(((s < low) | (s > high)).sum())
    metrics = {"colunas": use, "outliers_por_coluna": out_counts}
    return fig_to_bytes(), metrics

def plot_bar(df, cols):
    if cols: col = cols[0]
    else:
        cat = pick_categorical(df)
        if not cat: raise ValueError("Não há colunas categóricas para gráfico de barras.")
        col = cat[0]
    s = df[col].astype("category"); vc = s.value_counts().head(20)
    plt.figure(figsize=(10, 6)); sns.set_style("whitegrid")
    sns.barplot(x=vc.values, y=vc.index, palette="viridis")
    plt.title(f"Frequências de {col} (Top 20)"); plt.xlabel("Contagem"); plt.ylabel(col)
    metrics = {"coluna": col, "top_frequencias": [(str(k), int(v)) for k, v in vc.items()]}
    return fig_to_bytes(), metrics

def plot_scatter(df, cols):
    if len(cols) < 2: raise ValueError("Informe duas colunas numéricas para scatter.")
    x, y = cols[:2]
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] > 50000: data = data.sample(50000, random_state=42)
    plt.figure(figsize=(8, 6)); sns.set_style("whitegrid")
    sns.scatterplot(x=data[x], y=data[y], alpha=0.6, s=50, color="purple")
    plt.title(f"Dispersão: {x} vs {y} (PCA ou Amount)"); plt.xlabel(x); plt.ylabel(y)
    metrics = {"x": x, "y": y, "correlacao_pearson": float(data[x].corr(data[y]))}
    return fig_to_bytes(), metrics

def plot_class_summary(df):
    if "Class" not in df.columns: raise ValueError("Coluna 'Class' não encontrada.")
    s = pd.to_numeric(df["Class"], errors="coerce").dropna(); taxa = float(s.mean())
    metrics = {"taxa_fraude": taxa, "total_fraudes": int(s.sum()), "desbalanceamento": (taxa / (1 - taxa)) if 0 < taxa < 1 else "Todos fraudes/0"}
    plt.figure(figsize=(6, 4)); sns.set_style("whitegrid")
    counts = s.value_counts().sort_index()
    sns.barplot(x=[0, 1], y=[counts.get(0, 0), counts.get(1, 0)], palette=["blue", "red"])
    plt.xticks([0, 1], ["Normal (0)", "Fraude (1)"]); plt.title("Distribuição de Class (Fraudes vs Normais)"); plt.ylabel("Contagem")
    if "Amount" in df.columns:
        df2 = df[["Class", "Amount"]].copy()
        df2["Class"] = pd.to_numeric(df2["Class"], errors="coerce"); df2["Amount"] = pd.to_numeric(df2["Amount"], errors="coerce")
        medias = df2.groupby("Class")["Amount"].mean()
        metrics["amount_medio_por_classe"] = {int(k): float(v) for k, v in medias.items()}
    return fig_to_bytes(), metrics

def plot_pie(df, col, n_bins=None):
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and n_bins:
        s = pd.to_numeric(s, errors="coerce"); s = pd.cut(s, bins=n_bins)
    vc = s.value_counts().head(12)
    if vc.empty: raise ValueError("Sem dados para montar pizza nessa coluna.")
    plt.figure(figsize=(8, 8))
    plt.pie(vc.values, labels=[str(k) for k in vc.index], autopct="%1.1f%%", startangle=90, counterclock=False)
    plt.title(f"Distribuição de {col}")
    metrics = {"coluna": col, "categorias": [(str(k), int(v)) for k, v in vc.items()]}
    return fig_to_bytes(), metrics

def plot_auto(df):
    if "Class" in df.columns: return plot_class_summary(df)
    num = pick_numeric(df)
    if "Amount" in df.columns: return plot_hist(df, ["Amount"])
    if len(num) >= 2: return plot_corr(df, num)
    else: return plot_bar(df, None)

# ====== INSIGHTS (LLM) ======
def llm_insights(question: str, viz_kind: str, metrics: dict, model_name: str, history: str, context: str = "") -> str:
    summary = textwrap.dedent(f"""
    Contexto do dataset:
    {context}

    Pergunta atual:
    {question}

    Tipo de visualização:
    {viz_kind}

    Métricas calculadas (usar e explicar com números, percentuais e faixas):
    {json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_fallback)}

    Histórico de análises anteriores (incorporar aprendizados e evitar repetições):
    {history}
    """).strip()

    default_text = (
        "Insights detalhados (fallback):\n"
        "- Interprete as métricas acima de forma objetiva (médias, picos, correlações) e relacione com risco/negócio.\n"
        "- Destaque janelas/categorias críticas e impactos de outliers; comente qualidade do dado.\n"
        "- Recomende próximos passos (normalização, seleção de variáveis, limiares, validação estratificada)."
    )

    if not LLM_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return default_text

    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template=(
                "Você é um analista sênior de dados focado em detecção de fraudes. Responda EM PORTUGUÊS DO BRASIL.\n"
                "Escreva 6–12 bullets CLAROS e NUMERICAMENTE fundamentados (valores, percentuais, faixas, exemplos), "
                "incorporando histórico e contexto do dataset.\n"
                "{input}"
            )
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        resp = chain.run(input=summary)
        return resp.strip()
    except Exception:
        return default_text

# ====== EXECUÇÃO ======
def run_query(question: str, df: pd.DataFrame, model_name: str, history: str, override_spec=None):
    spec = override_spec if override_spec else route_query_to_viz(question, df)
    img_bytes, metrics = None, {}
    try:
        if spec.kind == "temporal":
            img_bytes, metrics = plot_temporal(df, spec.cols[0], use_class=spec.params.get("use_class", False))
        elif spec.kind == "corr":
            img_bytes, metrics = plot_corr(df, spec.cols)
        elif spec.kind == "hist":
            img_bytes, metrics = plot_hist(df, spec.cols)
        elif spec.kind == "box":
            img_bytes, metrics = plot_box(df, spec.cols)
        elif spec.kind == "bar":
            img_bytes, metrics = plot_bar(df, spec.cols)
        elif spec.kind == "scatter":
            img_bytes, metrics = plot_scatter(df, spec.cols)
        elif spec.kind == "class_summary":
            img_bytes, metrics = plot_class_summary(df)
        elif spec.kind == "pie":
            n_bins = spec.params.get("bins")
            img_bytes, metrics = plot_pie(df, spec.cols[0], n_bins=n_bins)
        else:
            img_bytes, metrics = plot_auto(df)

        context = dataset_profile_text(df, spec)
        insights = llm_insights(question, spec.kind, metrics, model_name, history, context=context)
        return spec, img_bytes, metrics, insights
    except Exception as e:
        try:
            img_bytes, metrics = plot_auto(df)
            insights = f"(Fallback após erro: {e})\n" + llm_insights(question, "auto", metrics, model_name, history)
            return VizSpec("auto", reason=f"Erro: {e}"), img_bytes, metrics, insights
        except Exception as e2:
            return VizSpec("none", reason=f"Erro geral: {e2}"), None, {}, f"Falha: {e2}"

# ====== PDF ======
def _wrap_text_lines(text, max_chars=95):
    lines = []
    for paragraph in (text or "").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append(""); continue
        lines.extend(textwrap.wrap(paragraph, width=max_chars))
    return lines

def _metrics_to_text(metrics: dict) -> str:
    try:
        return json.dumps(metrics or {}, ensure_ascii=False, indent=2, default=_json_fallback)
    except Exception:
        return str(metrics)

def build_pdf(qa_history: list, dataset_name: str, user_name: str = "", framework_text: str = DEFAULT_FRAMEWORK_TEXT) -> str:
    """Gera um PDF a partir do qa_history. Retorna o caminho do arquivo."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"relatorio_{_slugify_name(user_name) or 'anonimo'}_{ts}.pdf"
    out_path = os.path.join(tempfile.gettempdir(), fname)

    if PDF_BACKEND == "reportlab":
        # REPORTLAB
        c = canvas.Canvas(out_path, pagesize=A4)
        W, H = A4; margin = 40

        # Capa + Framework escolhida
        c.setFont("Helvetica-Bold", 20)
        c.drawString(margin, H - 80, "Relatório EDA — I2A2")
        c.setFont("Helvetica", 12)
        c.drawString(margin, H - 110, f"Usuário: {user_name or '—'}")
        c.drawString(margin, H - 130, f"Dataset: {dataset_name}")
        c.drawString(margin, H - 150, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        y = H - 180
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "Framework escolhida para este agente:")
        y -= 18
        c.setFont("Helvetica", 11)
        for line in _wrap_text_lines(framework_text, max_chars=95):
            c.drawString(margin, y, line); y -= 14
            if y < 80:
                c.showPage(); y = H - 60; c.setFont("Helvetica", 11)
        c.showPage()

        # Conteúdo
        for i, item in enumerate(qa_history, 1):
            c.setFont("Helvetica-Bold", 14); y = H - 60
            c.drawString(margin, y, f"{i}. Pergunta"); y -= 20
            c.setFont("Helvetica", 12)
            for line in _wrap_text_lines(item.get("pergunta", ""), max_chars=95):
                c.drawString(margin, y, line); y -= 14
                if y < 120: c.showPage(); y = H - 60

            meta_lines = []
            if item.get("spec_kind"):
                meta_lines.append(f"Tipo de análise: {item['spec_kind']} — {item.get('spec_reason','')}")
            if item.get("spec_cols"):
                meta_lines.append("Colunas usadas: " + ", ".join(item["spec_cols"]))
            if meta_lines:
                c.setFont("Helvetica-Oblique", 11)
                for line in meta_lines:
                    c.drawString(margin, y, line); y -= 14
                    if y < 120: c.showPage(); y = H - 60

            c.setFont("Helvetica-Bold", 14)
            if y < 120: c.showPage(); y = H - 60
            c.drawString(margin, y, "Insights"); y -= 20
            c.setFont("Helvetica", 12)
            for line in _wrap_text_lines(item.get("resposta", ""), max_chars=95):
                c.drawString(margin, y, line); y -= 14
                if y < 200: c.showPage(); y = H - 60

            c.setFont("Helvetica-Bold", 14)
            if y < 120: c.showPage(); y = H - 60
            c.drawString(margin, y, "Métricas calculadas (baseadas no CSV)"); y -= 20
            c.setFont("Helvetica", 12)
            for line in _wrap_text_lines(_metrics_to_text(item.get("metrics", {})), max_chars=95):
                c.drawString(margin, y, line); y -= 14
                if y < 200: c.showPage(); y = H - 60

            img_bytes = item.get("img_bytes")
            if img_bytes:
                try:
                    img = ImageReader(io.BytesIO(img_bytes))
                    iw, ih = img.getSize()
                    max_w = W - 2 * margin; max_h = H * 0.45
                    ratio = min(max_w / iw, max_h / ih); w, h = iw * ratio, ih * ratio
                    if y - h < 60: c.showPage(); y = H - 60
                    c.drawImage(img, margin, y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
                    y -= (h + 20)
                except Exception:
                    pass
            c.showPage()
        c.save()
        return out_path

    elif PDF_BACKEND == "fpdf2":
        # FPDF2
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=20)
        pdf.cell(0, 12, "Relatório EDA — I2A2", ln=1)
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 8, f"Usuário: {user_name or '—'}", ln=1)
        pdf.cell(0, 8, f"Dataset: {dataset_name}", ln=1)
        pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=1)

        pdf.ln(2)
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 8, "Framework escolhida para este agente:", ln=1)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, framework_text)

        for i, item in enumerate(qa_history, 1):
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(0, 8, f"{i}. Pergunta", ln=1)
            pdf.set_font("Helvetica", size=12)
            pdf.multi_cell(0, 6, item.get("pergunta", ""))

            meta = []
            if item.get("spec_kind"):
                meta.append(f"Tipo de análise: {item['spec_kind']} — {item.get('spec_reason','')}")
            if item.get("spec_cols"):
                meta.append("Colunas usadas: " + ", ".join(item["spec_cols"]))
            if meta:
                pdf.ln(1); pdf.set_font("Helvetica", style="I", size=11)
                pdf.multi_cell(0, 6, "\n".join(meta))

            pdf.ln(2); pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(0, 8, "Insights", ln=1)
            pdf.set_font("Helvetica", size=12)
            pdf.multi_cell(0, 6, item.get("resposta", ""))

            pdf.ln(2); pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(0, 8, "Métricas calculadas (baseadas no CSV)", ln=1)
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 5, _metrics_to_text(item.get("metrics", {})))

            img_bytes = item.get("img_bytes")
            if img_bytes:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(img_bytes); tmp_path = tmp.name
                    pdf.ln(2)
                    page_w = pdf.w - 2 * pdf.l_margin
                    pdf.image(tmp_path, w=page_w)
                    try: os.unlink(tmp_path)
                    except Exception: pass
                except Exception:
                    pass

        pdf.output(out_path)
        return out_path

    else:
        st.error("Nenhum backend de PDF disponível. Instale 'reportlab' ou 'fpdf2'.")
        return ""

# ====== UI PRINCIPAL ======
if df is not None:
    st.subheader("Faça sua pergunta")

    mem_vazia = len(st.session_state.get("qa_history", [])) == 0
    nome_preenchido = bool(st.session_state.get("user_name", "").strip())
    if mem_vazia and not nome_preenchido:
        st.warning("Para começar, informe seu nome na barra lateral. Isso permite criar relatórios personalizados e salvar sua memória.")
        st.stop()

    with st.expander("🎨 Tipo de gráfico (opcional) — escolha para sobrescrever a detecção automática"):
        viz_choice = st.selectbox(
            "Tipo de gráfico",
            ["Automático (com base na pergunta)", "Histograma", "Pizza", "Box plot", "Linha"],
            index=0
        )

        override_spec = None
        num_cols = pick_numeric(df); cat_cols = pick_categorical(df)

        if viz_choice == "Histograma":
            if not num_cols:
                st.error("Nenhuma coluna numérica disponível para histograma.")
            else:
                col = st.selectbox("Coluna numérica", num_cols, index=0, key="hist_col")
                override_spec = VizSpec("hist", [col], reason="Override: histograma escolhido pelo usuário.")
        elif viz_choice == "Pizza":
            mode = st.radio("Fonte de categorias", ["Coluna categórica", "Discretizar coluna numérica"], horizontal=True)
            if mode == "Coluna categórica":
                if not cat_cols and "Class" not in df.columns:
                    st.error("Não há coluna categórica disponível. Use a opção de discretizar numérica.")
                else:
                    options = ([c for c in ["Class"] if c in df.columns] + [c for c in cat_cols if c != "Class"])
                    col = st.selectbox("Coluna categórica", options, index=0, key="pie_cat_col")
                    override_spec = VizSpec("pie", [col], {"bins": None}, reason="Override: pizza categórica.")
            else:
                if not num_cols:
                    st.error("Não há colunas numéricas para discretização.")
                else:
                    col = st.selectbox("Coluna numérica", num_cols, index=0, key="pie_num_col")
                    n_bins = st.slider("Número de faixas (bins)", min_value=2, max_value=12, value=5, step=1, key="pie_bins")
                    override_spec = VizSpec("pie", [col], {"bins": n_bins}, reason="Override: pizza com discretização.")
        elif viz_choice == "Box plot":
            if not num_cols:
                st.error("Nenhuma coluna numérica disponível para box plot.")
            else:
                chosen = st.multiselect("Colunas numéricas (até 6)", num_cols, default=num_cols[:6], key="box_cols")
                if chosen and len(chosen) > 6: chosen = chosen[:6]
                if chosen:
                    override_spec = VizSpec("box", chosen, reason="Override: box plot escolhido pelo usuário.")
                else:
                    st.info("Selecione pelo menos uma coluna para o box plot.")
        elif viz_choice == "Linha":
            time_candidates = []
            if "Time" in df.columns: time_candidates.append("Time")
            time_candidates += [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64) and c not in time_candidates]
            fallback_nums = [c for c in num_cols if c not in time_candidates]
            options = (time_candidates + fallback_nums) if (time_candidates or fallback_nums) else []
            if not options:
                st.error("Não há colunas elegíveis para gráfico de linha.")
            else:
                col = st.selectbox("Coluna temporal/númerica para o eixo X", options, index=0, key="line_col")
                use_class = st.checkbox("Plotar taxa de fraude (usa 'Class' se existir)", value=("Class" in df.columns), key="line_use_class")
                override_spec = VizSpec("temporal", [col], {"use_class": use_class}, reason="Override: linha escolhida pelo usuário.")

    question = st.text_area(
        "Pergunta (PT-BR, ex.: 'Há padrões temporais em fraudes ao longo de Time? Agregue por janelas.')",
        height=100,
    )

    if st.button("▶️ Analisar", disabled=not question.strip()):
        with st.spinner("Interpretando pergunta, calculando métricas e gerando gráfico..."):
            history = "\n".join([f"Q: {item['pergunta']}\nA: {item['resposta']}" for item in st.session_state.get("qa_history", [])])
            spec, img_bytes, metrics, insights = run_query(
                question.strip(), df, model_name, history, override_spec=override_spec
            )

        st.markdown("#### Tipo de análise escolhido")
        st.write(f"**{spec.kind}** — {spec.reason or 'auto'}")
        if spec.cols:
            st.write(f"**Colunas usadas:** {', '.join(spec.cols)}")
        if img_bytes:
            st.image(img_bytes, caption="Visualização gerada a partir do CSV", use_column_width=True)
        else:
            st.warning("Nenhuma imagem foi gerada.")
        st.markdown("#### Métricas calculadas (baseadas no CSV)")
        st.json(metrics, expanded=False)
        st.markdown("#### Insights")
        st.write(insights)

        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []
        st.session_state.qa_history.append({
            "pergunta": question.strip(),
            "resposta": insights,
            "img_bytes": img_bytes,
            "metrics": metrics,
            "spec_kind": spec.kind,
            "spec_cols": spec.cols,
            "spec_reason": spec.reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "usuario": st.session_state.get("user_name", "").strip(),
        })

        if st.session_state.get("user_name", "").strip():
            save_user_memory(st.session_state["user_name"], st.session_state.qa_history)

    # ====== PDF ======
    st.markdown("---")
    left, right = st.columns([3, 1])
    with left:
        st.subheader("📄 Relatório")
        st.caption("Gera um PDF com as perguntas, insights detalhados, métricas e gráficos desta sessão.")
    with right:
        if st.session_state.get("qa_history"):
            if st.button("💾 Gerar PDF"):
                with st.spinner("Montando PDF..."):
                    out = build_pdf(
                        st.session_state.qa_history,
                        uploaded.name if uploaded else "creditcard.csv",
                        st.session_state.get("user_name", ""),
                        framework_text=DEFAULT_FRAMEWORK_TEXT
                    )
                if out and os.path.exists(out):
                    with open(out, "rb") as f:
                        base_name = os.path.basename(out)
                        st.download_button("⬇️ Baixar Relatório (PDF)", data=f.read(), file_name=base_name, mime="application/pdf")
                else:
                    st.error("Falha ao gerar o PDF. Verifique dependências (reportlab/fpdf2).")
        else:
            st.info("Faça pelo menos uma pergunta antes de gerar o PDF.")
else:
    st.info("Envie um arquivo CSV/ZIP na barra lateral para começar.")
