import csv
import json
import os
import time

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "dataset.csv"
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "mistral_large_3_outputs")
MODEL_NAME  = "Mistral-Large-3"
MAX_TOKENS  = 4096
MAX_RETRIES = 4
OVERWRITE   = os.environ.get("OVERWRITE", "").lower() in {"1", "true", "yes"}

AZURE_ENDPOINT = "https://subha-mcsj09dy-eastus2.services.ai.azure.com/openai/v1"
# ─────────────────────────────────────────────────────────────────────────────

# Broad categories — well-separated, classifier-friendly
VISUALIZATION_CATEGORIES = [
    "Microscopy",
    "Diffraction",
    "Spectroscopy",
    "Thermal Analysis",
    "Phase/Equilibrium Diagram",
    "Mechanical Test",
    "Electrochemistry",
    "Magnetic/Electronic",
    "Optical/Photonic",
    "Tomography/3D",
    "Compositional Map",
    "Simulation",
    "Machine Learning",
    "Crystal Structure",
    "Generic Plot",
    "Schematic/Diagram",
    "Photograph",
    "Table",
    "other",
]

# Subtypes grouped by category — used both for prompt guidance and for building
# the flat subtype enum. Order within a category does not matter for the enum.
SUBTYPES_BY_CATEGORY = {
    "Microscopy": ["SEM", "TEM", "STEM", "HAADF-STEM", "BF-TEM", "DF-TEM",
                   "Optical Micrograph", "Confocal Microscopy", "AFM",
                   "Fluorescence Microscopy", "Live/Dead Staining"],
    "Diffraction": ["XRD Pattern", "SAED", "EBSD Map", "Pole Figure",
                    "Inverse Pole Figure", "Neutron Diffraction",
                    "Synchrotron Diffraction"],
    "Spectroscopy": ["XPS Spectrum", "Raman Spectrum", "FTIR Spectrum",
                     "EDX Spectrum", "EELS Spectrum", "NMR Spectrum",
                     "Mass Spectrum", "UV-Vis Spectrum",
                     "Photoluminescence Spectrum", "XANES Spectrum",
                     "EXAFS Spectrum", "Mössbauer Spectrum"],
    "Thermal Analysis": ["DSC Curve", "TGA Curve", "DMA Curve", "TMA Curve"],
    "Phase/Equilibrium Diagram": ["Binary Phase Diagram", "Ternary Phase Diagram",
                                  "TTT Diagram", "CCT Diagram",
                                  "CALPHAD Diagram", "Pourbaix Diagram"],
    "Mechanical Test": ["Stress-Strain Curve", "Load-Displacement Curve",
                        "Nanoindentation Curve", "Hardness Map",
                        "Fatigue/S-N Curve", "Creep Curve",
                        "Fracture Toughness Plot", "DIC Strain Map",
                        "Wear/Tribology Plot"],
    "Electrochemistry": ["Cyclic Voltammogram", "Charge-Discharge Curve",
                         "Capacity Retention Plot", "Coulombic Efficiency Plot",
                         "Nyquist Plot", "Bode Plot", "Tafel Plot",
                         "Polarization Curve", "GITT/PITT Curve",
                         "Rate Capability Plot"],
    "Magnetic/Electronic": ["M-H Hysteresis Loop", "M-T Curve", "ZFC/FC Curve",
                            "I-V Curve", "C-V Curve", "Band Structure",
                            "Density of States", "Hall Effect Plot"],
    "Optical/Photonic": ["Absorbance Spectrum", "Transmittance Spectrum",
                         "Reflectance Spectrum", "EQE/IQE Plot", "J-V Curve",
                         "Ellipsometry Plot", "Refractive Index Plot"],
    "Tomography/3D": ["APT Reconstruction", "Micro-CT", "FIB-SEM Tomography",
                      "3D Reconstruction"],
    "Compositional Map": ["EDS Map", "WDS Map", "EBSD IPF Map",
                          "Elemental Distribution Map"],
    "Simulation": ["DFT Result", "MD Snapshot", "MD Trajectory",
                   "Phase-Field Simulation", "FEA/FEM Result",
                   "Monte Carlo Result"],
    "Machine Learning": ["Parity Plot", "Confusion Matrix", "ROC Curve",
                         "Learning Curve", "Feature Importance Plot",
                         "SHAP Plot", "t-SNE/UMAP/PCA Plot"],
    "Crystal Structure": ["Unit Cell", "Atomic Model", "Supercell"],
    "Generic Plot": ["Bar Chart", "Scatter Plot", "Line Graph", "Box Plot",
                     "Contour Plot", "Heatmap", "Radar Chart", "Ashby Plot",
                     "Arrhenius Plot", "Histogram"],
    "Schematic/Diagram": ["Process Schematic", "Flowchart",
                          "Mechanism Diagram", "Experimental Setup"],
    "Photograph": ["Sample Photo", "Equipment Photo", "In-situ Photo"],
    "Table": ["Data Table"],
    "other": ["other"],
}

VISUALIZATION_SUBTYPES = list(dict.fromkeys(
    sub for subs in SUBTYPES_BY_CATEGORY.values() for sub in subs
))


def _build_taxonomy_block() -> str:
    return "\n".join(
        f"- {cat}: {', '.join(subs)}"
        for cat, subs in SUBTYPES_BY_CATEGORY.items()
    )


TAXONOMY_BLOCK = _build_taxonomy_block()

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "figure_panels",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["panels"],
            "additionalProperties": False,
            "properties": {
                "panels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["panel", "visualization_category",
                                     "visualization_subtype", "subcaption", "summary"],
                        "additionalProperties": False,
                        "properties": {
                            "panel": {"type": "string"},
                            "visualization_category": {
                                "type": "string",
                                "enum": VISUALIZATION_CATEGORIES,
                            },
                            "visualization_subtype": {"type": "string"},
                            "subcaption": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

PROMPT_TEMPLATE = """\
You are a scientific figure analysis expert specializing in materials science.

Using only the figure caption and the reference sentences from the paper, \
generate detailed sub-captions for each panel of the figure.

Figure caption:
{caption}

Reference sentences from the paper:
{reference_sentences}

INSTRUCTIONS:
- Generate a sub-caption for each panel (a, b, c, …); use panel id "main" for single-panel figures.
- Classify each panel with TWO fields, chosen ONLY from the allowed taxonomy below:
    * visualization_category: the broad category.
    * visualization_subtype: the specific technique/plot type, which MUST belong to the chosen category.
- Both fields are strict enums. Use "other" only if absolutely nothing matches.
- For plots/graphs: state axes, specimen/condition, test method, and any key trend or value from the caption or reference.
- For images/maps: state material, technique, orientation/view, processing condition, and key observation or structural feature.
- The caption is the ground truth for what this figure shows - never contradict it.
- Use a reference sentence only if it clearly supports the caption content. Discard any reference sentence that describes a different figure or unrelated content.
- Group-level references that support the caption apply to all panels; references naming a specific panel letter apply only to that panel.
- Do not start every sub-caption identically - vary sentence openings across panels. Strictly avoid repeatedly starting sentences with articles or determiners.
- Expand the caption into a precise scientific description - do not copy it verbatim and do not speculate beyond what the caption and references state.
- For each panel, write a summary of 40-60 words using only that panel's subcaption and its directly related reference sentences. Be concise and focus on subcaption's content and key findings. Exclude unrelated information. Don't include panel words like "panel a" or "the figure" in the summary.

ALLOWED TAXONOMY (category -> valid subtypes):
{taxonomy}
"""


def _load_rows() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pending_rows(rows: list[dict]) -> list[dict]:
    def _needs_run(r: dict) -> bool:
        if OVERWRITE:
            return True
        path = os.path.join(OUTPUT_DIR, f"{r['image_name']}.json")
        if not os.path.exists(path):
            return True
        with open(path, encoding="utf-8") as f:
            return "error" in json.load(f)
    return [r for r in rows if _needs_run(r)]


def _process_row(row: dict, client: OpenAI) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        caption=row["caption"],
        reference_sentences=row["reference"],
        taxonomy=TAXONOMY_BLOCK,
    )

    # Mistral and Llama models use max_tokens; OpenAI models use max_completion_tokens
    _is_openai = MODEL_NAME.lower().startswith("gpt") or MODEL_NAME.lower().startswith("o")
    token_kwarg = {"max_completion_tokens": MAX_TOKENS} if _is_openai else {"max_tokens": MAX_TOKENS}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                **token_kwarg,
                temperature=0.1,
                response_format=RESPONSE_FORMAT,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            print(f"    attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}", flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return {"error": str(last_error)}


def main():
    api_key = os.environ.get("AZURE_API_KEY")
    if not api_key:
        raise EnvironmentError("AZURE_API_KEY environment variable is not set.")

    client = OpenAI(base_url=AZURE_ENDPOINT, api_key=api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = _load_rows()
    pending = _pending_rows(rows)
    total = len(pending)

    print(f"Total rows   : {len(rows)}")
    print(f"Already done : {len(rows) - total}")
    print(f"To process   : {total}")
    print(f"Model        : {MODEL_NAME}\n")

    if not pending:
        print("Nothing to do.")
        return

    success = errors = 0
    for i, row in enumerate(pending, start=1):
        custom_id = row["image_name"]
        print(f"  [{i}/{total}] processing {custom_id} ...", end=" ", flush=True)
        parsed = _process_row(row, client)

        out_path = os.path.join(OUTPUT_DIR, f"{custom_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        if "error" in parsed:
            errors += 1
            print(f"ERROR: {parsed['error']}", flush=True)
        else:
            success += 1
            print(f"ok  (panels={len(parsed.get('panels', []))})", flush=True)

    print(f"\nDone. success={success}  errors={errors}")
    print(f"JSON files saved in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
