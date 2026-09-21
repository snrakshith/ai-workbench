# 🧠 Prompt Engineering — Interactive Notebooks

> A hands-on, example-driven course covering all 8 core prompt engineering techniques.  
> Runs on **free API keys** (Groq · Gemini) — no paid account required.

---

## 📋 Course Overview

| # | Topic | Notebook |
|---|-------|----------|
| 1 | Anatomy of a Prompt | `01_prompt_foundations` |
| 2 | Zero-Shot, One-Shot & Few-Shot Prompting | `01_prompt_foundations` |
| 3 | System Prompt Design & Role Assignment | `01_prompt_foundations` |
| 4 | Chain-of-Thought & Step-Back Prompting | `02_reasoning_and_output` |
| 5 | Output Formatting & Structured Generation | `02_reasoning_and_output` |
| 6 | Prompt Sensitivity, Fragility & Robustness Testing | `02_reasoning_and_output` |
| 7 | Prompt Chaining & Decomposition Strategies | `03_advanced_strategies` |
| 8 | Meta-Prompting & Self-Refinement Loops | `03_advanced_strategies` |

---

## 🗂️ Repository Structure

```
prompt-engineering-notebooks/
│
├── 01_prompt_foundations.ipynb      ← Topics 1–3
├── 02_reasoning_and_output.ipynb    ← Topics 4–6
├── 03_advanced_strategies.ipynb     ← Topics 7–8
│
├── pyproject.toml                            ← All dependencies (uv)
├── .python-version                           ← Python 3.12
├── .env.example                              ← API key template
├── .env                                      ← Your keys (gitignored)
└── .gitignore
```

---

## ⚡ Quick Start

### 1 · Clone & enter the repo

```bash
git clone <repo-url>
cd prompt-engineering-notebooks
```

### 2 · Set up environment (Python 3.12 + uv)

```bash
# Install uv if you don't have it
curl -Ls https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3 · Install dependencies

```bash
uv pip install -r pyproject.toml
```

### 4 · Add your API key

```bash
cp .env.example .env
# Open .env and fill in ONE of the three provider sections
```

### 5 · Launch JupyterLab

```bash
jupyter lab
```

Open notebooks in order: `01 → 02 → 03`

---

## 🔑 API Keys — Free Options Available

You only need **one** provider. All three work identically inside the notebooks.

| Provider | Cost | Free Limits | Get Your Key |
|----------|------|-------------|--------------|
| **Groq** ✅ | Free, no credit card | Rate-limited free tier | [console.groq.com/keys](https://console.groq.com/keys) |
| **Gemini** ✅ | Free, no credit card | Generous daily quota | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| **OpenAI** | Paid | — | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

Add exactly one key to `.env`:

```dotenv
# Option A — Groq (recommended for students, fastest inference)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

# Option B — Gemini (most generous free quota)
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxx

# Option C — OpenAI (paid)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

The notebooks **auto-detect** which key you've set and configure the right model automatically. No other code changes needed.

### Models used per provider

| Provider | Model | Notes |
|----------|-------|-------|
| Groq | `openai/gpt-oss-120b` | GPT-class open model, 131k context |
| Gemini | `models/gemini-3.7-flash` | Latest Flash, fast & capable |
| OpenAI | `gpt-4o` | Best quality |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12 | Runtime |
| uv | latest | Package management |
| openai SDK | ≥ 3.8.0 | OpenAI, Groq & Gemini client |
| anthropic SDK | ≥ 1.3.0 | Claude (optional, XML examples) |
| pydantic | ≥ 2.12.0 | Structured output validation |
| python-dotenv | latest | `.env` key loading |
| rich | ≥ 14.0.0 | Formatted terminal output |
| jupyterlab | ≥ 4.6.3 | Notebook environment |
| ipykernel | ≥ 7.3.0 | Jupyter Python kernel |

---

## 🧪 Testing

All notebooks were executed end-to-end and verified:

```
Notebook 1 — Foundations          11/11 cells ✅
Notebook 2 — Reasoning & Output   14/14 cells ✅
Notebook 3 — Advanced Strategies  11/11 cells ✅

Total: 36/36 cells — zero errors
```

To re-run the test yourself:

```bash
source .venv/bin/activate

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=300 \
  --output executed.ipynb \
  01_prompt_foundations.ipynb
```

---

## 📚 Research References

Key findings embedded throughout the notebooks:

- **+34%** accuracy improvement with structured CoT vs direct prompting *(ibuidl.org, 2026)*
- **71% → 94%** JSON format compliance with 3 few-shot examples *(ibuidl.org, 2026)*
- **+17.9%** on GSM8K math benchmark with CoT + self-consistency *(Wang et al., 2022)*
- **+10–25%** quality improvement from self-refinement loops *(Lushbinary, 2026)*
- **+10–30%** accuracy from prompt chaining vs single mega-prompt *(SurePrompts, 2026)*
- System prompts **>800 tokens** dilute instruction adherence *(ibuidl.org, 2026)*
- Explicit CoT **hurts** on reasoning-native models (o-series, Gemini Thinking) *(promtable.com, 2026)*

---

## 🚀 Recommended Learning Path

```
Day 1  →  Notebook 01  (Foundations — ~45 min)
Day 2  →  Notebook 02  (Reasoning & Output — ~60 min)
Day 3  →  Notebook 03  (Advanced Strategies — ~75 min)
```

Each notebook follows: **Theory → Live Example → Side-by-Side Comparison → Student Exercise**

---

## ❓ Troubleshooting

**`EnvironmentError: No valid API key found`**  
→ Make sure `.env` exists (not just `.env.example`) and your key is not the placeholder value.

**`404 model not found` on Groq or Gemini**  
→ Model names change. List current models:
```python
for m in client.models.list().data:
    print(m.id)
```

**`uv: command not found`**  
→ Install uv: `curl -Ls https://astral.sh/uv/install.sh | sh` then restart terminal.

**Notebook cell times out**  
→ Self-refinement cells (NB3) make 10–15 API calls. Increase timeout or switch to Groq (fastest inference).

---

## 📄 License

MIT — free to use, adapt, and share for teaching purposes.
