# 💳 Agente EDA Especialista em Fraude de Cartão (creditcard.csv)

[![Streamlit](https://img.shields.io/badge/Streamlit-%F0%9F%93%8A-red)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)

Um **agente de IA especialista** em **análise exploratória** de **fraudes em cartão de crédito**, otimizado para o dataset **`creditcard.csv`** do Kaggle (MLG-ULB).
Você pergunta em **PT-BR**, e o app:

* Gera **gráficos** pertinentes ao domínio (distribuições, outliers, evolução temporal da **taxa de fraude**, etc.),
* Calcula **métricas críticas** ao problema (desbalanceamento de classe, estatísticas de `Amount`, correlação com `Class`, contagem de outliers por IQR, etc.),
* Produz **insights detalhados e cumulativos**, com **memória por usuário**,
* Exporta um **PDF completo** (pergunta, colunas, métricas e gráficos),
* Mostra na capa do PDF: **“Framework escolhida para este agente”**.

> **Dataset alvo**: [Kaggle — Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
> Estrutura clássica: 31 colunas (`Time`, `V1`…`V28` via PCA, `Amount`, `Class`), sendo `Class` a variável alvo (0 = normal, 1 = fraude).

---

## 🎯 Por que especializado?

Fraude em cartão possui **sinais, métricas e armadilhas próprias**:

* **Desbalanceamento extremo** (poucas fraudes) — afeta leitura de métricas, thresholds e modelagem.
* **Features PCA (`V1`–`V28`)** — interpretabilidade limitada, mas úteis para padrões/score.
* **Tempo e sazonalidade** — picos de taxa de fraude por **janelas temporais**.
* **Montantes (`Amount`)** — caudas longas e outliers relevantes.

Este agente traz **rotas de visualização** e **insights** ajustados a esse contexto.

---

## 🧱 Framework escolhida (resumo)

* **Frontend**: Streamlit
* **Análises**: Pandas / NumPy / Matplotlib / Seaborn
* **IA**: LangChain + OpenAI (`ChatOpenAI`) — *insights numéricos e cumulativos (memória)*
* **Memória**: JSON por usuário em `memories/` (carregar/limpar via sidebar)
* **PDF**: `reportlab` (preferencial) ou `fpdf2` (fallback)

Fluxo: **Pergunta → Roteador de intenção (domínio) → Métricas + Gráfico → Insights (LLM) → Histórico/memória → PDF**

---

## 🔎 O que o agente responde (foco no domínio)

* **Descrição dos dados (domínio)**

  * Distribuição de `Amount` (assimetria, min–max, média/mediana, desvio).
  * Colunas PCA `V1`…`V28`: correlações internas e **com `Class`**.
* **Padrões e tendências (tempo)**

  * Evolução de **contagem de transações** por janela.
  * Evolução da **taxa de fraude** por janela quando `Class` existe.
* **Frequências e categorias**

  * Contagem de `Class` e distribuição (barras/pizza).
* **Outliers e anomalias**

  * **Box plots** com **contagem de outliers por IQR** (por coluna).
* **Relações entre variáveis**

  * **Correlação** no conjunto numérico (heatmap) e **dispersão** (Pearson).
* **Conclusões do agente**

  * Síntese acumulada do que foi visto (memória por usuário).

> **Clusters e crosstab** não estão ativados por padrão; podem entrar no roadmap.

---

## 🧩 Pré-requisitos

* Python **3.10+**
* **OPENAI_API_KEY** (para insights do LLM; sem chave, o app cai em *fallback* de texto)
* Dependências do `requirements.txt`

---

## ⚙️ Instalação e execução

**Configurar chave da OpenAI (uma das opções abaixo):**

* **Var. de ambiente (recomendado):**

  ```bash
  export OPENAI_API_KEY="sk-..."
  # Windows (PowerShell):
  # setx OPENAI_API_KEY "sk-..."
  ```
* **Streamlit secrets:**

  ```toml
  # .streamlit/secrets.toml  (NÃO versionar)
  OPENAI_API_KEY = "sk-..."
  ```

**Rodar:**

```bash
streamlit run app.py
```

Abra o link (geralmente `http://localhost:8501`).

---

## 🧭 Como usar (passo a passo)

1. **Sidebar**: informe seu **nome** (obrigatório quando a memória está vazia).
2. **Upload**: anexe o arquivo do Kaggle (`Kaggle - Credit Card Fraud.zip`) ou `creditcard.csv`.
3. (Opcional) **Carregar memória** do seu usuário; use **Limpar memória** para um relatório “do zero”.
4. (Opcional) **Escolha do gráfico** (override): Histograma, Pizza, Box plot, Linha.
5. **Pergunte** em PT-BR e clique **▶️ Analisar**.
6. **Gere o PDF** em “📄 Relatório → 💾 Gerar PDF”.

---

## 💡 Perguntas recomendadas (domínio fraude)

* “Qual a **distribuição de `Amount`**? Faça histograma e traga **média/mediana/min–max/desvio**.”
* “Qual a **taxa de fraude por janela de tempo** usando `Time` e `Class`?”
* “Quais **`V*` mais correlacionam com `Class`**? Traga o **top 5**.”
* “Mostre **outliers por IQR** nas colunas `Amount`, `V1`, `V2`, `V3` (box plot).”
* “Quais **conclusões gerais** o agente obteve até agora?”

---

## 🔒 Segurança

* **Nunca** exponha a **OPENAI_API_KEY** no código.
* **Não** comite `.streamlit/secrets.toml`.
* **Ignore** `memories/` no Git (guarde só um `.gitkeep`).
* Em caso de vazamento, **revogue** a chave imediatamente.

---

## 🙌 Agradecimentos

* **I2A2** pela proposta da atividade.
* **MLG-ULB/Kaggle** pelo dataset **Credit Card Fraud**.
---

Pronto. Agora o posicionamento está **claramente especializado** no **`creditcard.csv`** e no **problema de fraude**. Quer que eu adapte as *sugestões de perguntas* para o seu uso específico (turma/exercício)?

