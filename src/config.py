"""Central settings. Change values here instead of inside the scripts,
so later experiments (chunk size, models, top-k) are one-line edits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# ---- Models (served locally by Ollama) ----
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"          # used from Day 2 onwards
OLLAMA_HOST = "http://localhost:11434"

# ---- Chunking (in words; ~1.3 tokens per word) ----
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

# ---- Retrieval ----
TOP_K = 5
EMBED_BATCH_SIZE = 32

# ---- Data sources ----
# Common drugs across therapy areas, so questions about interactions,
# side effects and dosing have something to retrieve.
DRUGS = [
    "metformin", "sitagliptin", "empagliflozin", "insulin glargine",
    "atorvastatin", "rosuvastatin", "amlodipine", "lisinopril", "losartan",
    "metoprolol", "warfarin", "apixaban", "clopidogrel", "aspirin",
    "ibuprofen", "naproxen", "acetaminophen", "tramadol",
    "omeprazole", "pantoprazole", "levothyroxine", "prednisone",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "sertraline", "fluoxetine", "escitalopram", "bupropion",
    "alprazolam", "gabapentin", "levetiracetam", "montelukast",
    "albuterol", "cetirizine", "furosemide", "hydrochlorothiazide",
    "spironolactone", "sacubitril and valsartan",
]

# Label sections worth indexing (openFDA field names).
LABEL_SECTIONS = [
    "boxed_warning", "indications_and_usage", "dosage_and_administration",
    "contraindications", "warnings_and_cautions", "warnings",
    "adverse_reactions", "drug_interactions", "use_in_specific_populations",
    "mechanism_of_action",
]

PUBMED_QUERIES = [
    "metformin adverse effects",
    "SGLT2 inhibitors heart failure outcomes",
    "warfarin drug interactions",
    "statin muscle toxicity",
    "NSAID kidney injury",
    "SSRI serotonin syndrome",
    "fluoroquinolone tendon rupture",
    "direct oral anticoagulants bleeding risk",
    "sacubitril valsartan heart failure",
    "proton pump inhibitor long-term risks",
]
PUBMED_PER_QUERY = 20

# ---- Generation (Day 2) ----
# If the best retrieved chunk scores below this cosine similarity, we refuse
# to answer instead of letting the model guess. Tune it with the Day 4 eval.
MIN_RELEVANCE = 0.65
LLM_TEMPERATURE = 0.0

# ---- Agent (Day 3) ----
AGENT_MAX_STEPS = 4          # max think->act->observe rounds before forcing an answer
LABEL_SNIPPET_CHARS = 2500   # how much of a label section the lookup tool returns
