"""Central settings. Change values here instead of inside the scripts,
so later experiments (chunk size, models, top-k) are one-line edits."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# ---- Models (served locally by Ollama) ----
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")   # override to compare models, e.g. LLM_MODEL=qwen2.5:7b
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")   # overridden in Docker

# ---- Chunking (in words; ~1.3 tokens per word) ----
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

# ---- Retrieval ----
TOP_K = 5
EMBED_BATCH_SIZE = 32

# ---- Data sources ----
# ~300 drugs across therapy areas (cardiometabolic, pain, infection, CNS, immunology, oncology,
# endocrine, respiratory, plus much of the Novartis portfolio), so questions about interactions,
# side effects and dosing have something to retrieve. Deliberately NOT indexed, so out-of-scope tests stay
# valid: lithium, pembrolizumab, nivolumab, dupilumab, apremilast (adalimumab, semaglutide and tirzepatide
# appear in a few PubMed titles, so they are only partly covered).
DRUGS = [
    "metformin", "sitagliptin", "empagliflozin", "insulin glargine", "atorvastatin", "rosuvastatin",
    "amlodipine", "lisinopril", "losartan", "metoprolol", "warfarin", "apixaban", "clopidogrel", "aspirin",
    "ibuprofen", "naproxen", "acetaminophen", "tramadol", "omeprazole", "pantoprazole", "levothyroxine",
    "prednisone", "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "sertraline",
    "fluoxetine", "escitalopram", "bupropion", "alprazolam", "gabapentin", "levetiracetam", "montelukast",
    "albuterol", "cetirizine", "furosemide", "hydrochlorothiazide", "spironolactone",
    "sacubitril and valsartan", "glipizide", "glimepiride", "pioglitazone", "linagliptin", "saxagliptin",
    "dapagliflozin", "canagliflozin", "liraglutide", "dulaglutide", "insulin lispro", "insulin aspart",
    "simvastatin", "pravastatin", "ezetimibe", "fenofibrate", "gemfibrozil", "niacin", "icosapent ethyl",
    "valsartan", "irbesartan", "olmesartan", "telmisartan", "candesartan", "benazepril", "enalapril",
    "ramipril", "captopril", "carvedilol", "atenolol", "bisoprolol", "propranolol", "nebivolol",
    "diltiazem", "verapamil", "nifedipine", "hydralazine", "clonidine", "doxazosin",
    "isosorbide mononitrate", "nitroglycerin", "digoxin", "amiodarone", "ranolazine", "ivabradine",
    "eplerenone", "torsemide", "bumetanide", "chlorthalidone", "triamterene", "rivaroxaban", "dabigatran",
    "edoxaban", "prasugrel", "ticagrelor", "cilostazol", "enoxaparin", "heparin", "ketorolac",
    "diclofenac", "meloxicam", "celecoxib", "indomethacin", "oxycodone", "hydrocodone", "morphine",
    "fentanyl", "codeine", "buprenorphine", "methadone", "naloxone", "naltrexone", "cyclobenzaprine",
    "methocarbamol", "tizanidine", "baclofen", "esomeprazole", "lansoprazole", "rabeprazole", "famotidine",
    "ranitidine", "ondansetron", "metoclopramide", "loperamide", "bisacodyl", "mesalamine", "sucralfate",
    "amoxicillin and clavulanate", "cephalexin", "cefdinir", "ceftriaxone", "penicillin v", "clindamycin",
    "levofloxacin", "moxifloxacin", "sulfamethoxazole and trimethoprim", "nitrofurantoin", "metronidazole",
    "linezolid", "vancomycin", "clarithromycin", "minocycline", "rifampin", "isoniazid", "fluconazole",
    "terbinafine", "voriconazole", "acyclovir", "valacyclovir", "oseltamivir", "hydroxychloroquine",
    "ivermectin", "tenofovir", "emtricitabine", "dolutegravir", "sofosbuvir", "citalopram", "paroxetine",
    "venlafaxine", "duloxetine", "mirtazapine", "trazodone", "amitriptyline", "nortriptyline",
    "vortioxetine", "quetiapine", "olanzapine", "risperidone", "aripiprazole", "haloperidol", "clozapine",
    "ziprasidone", "lurasidone", "lamotrigine", "valproic acid", "carbamazepine", "oxcarbazepine",
    "topiramate", "phenytoin", "lacosamide", "pregabalin", "clonazepam", "lorazepam", "diazepam",
    "zolpidem", "buspirone", "hydroxyzine", "methylphenidate", "amphetamine", "atomoxetine", "modafinil",
    "donepezil", "memantine", "rivastigmine", "ropinirole", "pramipexole", "carbidopa and levodopa",
    "sumatriptan", "rizatriptan", "levocetirizine", "loratadine", "fexofenadine", "diphenhydramine",
    "fluticasone", "budesonide", "ipratropium", "tiotropium", "salmeterol", "theophylline", "benzonatate",
    "guaifenesin", "pseudoephedrine", "methylprednisolone", "dexamethasone", "hydrocortisone",
    "methotrexate", "azathioprine", "mycophenolate", "cyclosporine", "tacrolimus", "sirolimus",
    "everolimus", "leflunomide", "sulfasalazine", "tofacitinib", "etanercept", "infliximab", "secukinumab",
    "ustekinumab", "omalizumab", "canakinumab", "ofatumumab", "fingolimod", "siponimod",
    "dimethyl fumarate", "natalizumab", "ocrelizumab", "imatinib", "nilotinib", "dasatinib", "ruxolitinib",
    "ribociclib", "palbociclib", "letrozole", "anastrozole", "tamoxifen", "bicalutamide", "abiraterone",
    "capecitabine", "cyclophosphamide", "trastuzumab", "bevacizumab", "rituximab", "lenalidomide",
    "pazopanib", "sunitinib", "octreotide", "zoledronic acid", "deferasirox", "eltrombopag", "hydroxyurea",
    "alendronate", "raloxifene", "denosumab", "levonorgestrel and ethinyl estradiol", "estradiol",
    "medroxyprogesterone", "progesterone", "finasteride", "tamsulosin", "sildenafil", "tadalafil",
    "oxybutynin", "solifenacin", "mirabegron", "allopurinol", "febuxostat", "colchicine", "methimazole",
    "liothyronine", "cabergoline", "ranibizumab", "aflibercept", "timolol", "latanoprost", "brimonidine",
    "dorzolamide", "folic acid", "cyanocobalamin", "potassium chloride", "sodium bicarbonate",
    "calcium carbonate", "ferrous sulfate", "epinephrine", "atropine", "lidocaine", "propofol", "ketamine",
    "midazolam",
]

# Label sections worth indexing (openFDA field names).
LABEL_SECTIONS = [
    "boxed_warning", "indications_and_usage", "dosage_and_administration",
    "contraindications", "warnings_and_cautions", "warnings",
    "adverse_reactions", "drug_interactions", "use_in_specific_populations",
    "mechanism_of_action",
]

PUBMED_QUERIES = [
    "metformin adverse effects", "SGLT2 inhibitors heart failure outcomes", "warfarin drug interactions",
    "statin muscle toxicity", "NSAID kidney injury", "SSRI serotonin syndrome",
    "fluoroquinolone tendon rupture", "direct oral anticoagulants bleeding risk",
    "sacubitril valsartan heart failure", "proton pump inhibitor long-term risks",
    "GLP-1 receptor agonist cardiovascular outcomes", "DPP-4 inhibitor safety",
    "sulfonylurea hypoglycemia risk", "insulin therapy type 2 diabetes", "ACE inhibitor angioedema",
    "angiotensin receptor blocker hyperkalemia", "beta blocker heart failure mortality",
    "calcium channel blocker edema", "amiodarone toxicity", "digoxin toxicity", "statin diabetes risk",
    "PCSK9 inhibitor lipid lowering", "antiplatelet therapy after stent",
    "atrial fibrillation anticoagulation elderly", "diuretic electrolyte disturbance",
    "opioid overdose risk", "opioid use disorder buprenorphine", "gabapentinoid misuse",
    "acetaminophen hepatotoxicity", "NSAID gastrointestinal bleeding",
    "Clostridioides difficile antibiotic risk", "macrolide QT prolongation", "beta-lactam allergy",
    "antibiotic resistance outpatient prescribing", "antifungal drug interactions",
    "HIV antiretroviral drug interactions", "hepatitis C direct acting antivirals",
    "H2 blocker versus proton pump inhibitor", "antidepressant suicidality young adults",
    "benzodiazepine dependence elderly", "antipsychotic metabolic effects", "clozapine agranulocytosis",
    "valproate pregnancy teratogenicity", "antiepileptic drug interactions", "carbamazepine HLA-B*1502",
    "ADHD stimulant cardiovascular safety", "cholinesterase inhibitors dementia",
    "triptan cardiovascular risk", "zolpidem complex sleep behaviors", "methotrexate toxicity monitoring",
    "TNF inhibitor serious infection", "IL-17 inhibitor psoriasis", "JAK inhibitor safety",
    "multiple sclerosis disease-modifying therapy", "calcineurin inhibitor nephrotoxicity",
    "mTOR inhibitor transplant", "imatinib chronic myeloid leukemia",
    "tyrosine kinase inhibitor cardiotoxicity", "CDK4/6 inhibitor breast cancer",
    "aromatase inhibitor bone loss", "checkpoint inhibitor adverse events", "iron chelation deferasirox",
    "eltrombopag thrombocytopenia", "anti-VEGF intravitreal injection",
    "bisphosphonate atypical femur fracture", "levothyroxine dosing",
    "hormone therapy cardiovascular risk", "5-alpha reductase inhibitor",
    "phosphodiesterase 5 inhibitor nitrate interaction", "inhaled corticosteroid pneumonia risk",
    "asthma biologic therapy", "montelukast neuropsychiatric events",
    "drug use in pregnancy first trimester", "medication safety in older adults Beers criteria",
    "renal dosing adjustment", "drug induced liver injury", "QT prolonging drugs",
    "serotonin syndrome drug combinations", "adverse drug reactions polypharmacy",
]
PUBMED_PER_QUERY = 25

# ---- Generation (Day 2) ----
# If the best retrieved chunk scores below this cosine similarity, we refuse
# to answer instead of letting the model guess. Tune it with the Day 4 eval.
MIN_RELEVANCE = 0.65
LLM_TEMPERATURE = 0.0
MAX_NEW_TOKENS = 700      # hard cap on every answer: an unbounded small model once looped for 44,000 tokens (10 minutes)
LLM_OPTIONS = {"temperature": LLM_TEMPERATURE, "num_predict": MAX_NEW_TOKENS}

# ---- Agent (Day 3) ----
AGENT_MAX_STEPS = 4          # max think->act->observe rounds before forcing an answer
LABEL_SNIPPET_CHARS = 2500   # how much of a label section the lookup tool returns

# ---- Reranking ----
USE_RERANKER = True     # cross-encoder over the top 20 candidates; needs torch (falls back silently without it)
