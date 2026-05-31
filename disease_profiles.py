"""Disease descriptions and treatment copy for the ALeaf-STENet UI."""

from __future__ import annotations


DEFAULT_PROFILE = {
    "scientific_name": "Apple foliage pathology",
    "symptoms": [
        "Visible changes in leaf color, texture, or lesion density detected by the classifier.",
        "Localized tissue stress patterns measured across the leaf blade.",
        "Model attention concentrated around abnormal foliage regions.",
    ],
    "preventive": [
        "Keep orchard rows clean and remove infected plant debris after pruning.",
        "Improve canopy airflow with regular pruning and balanced spacing.",
        "Monitor leaves after humid or rainy periods to catch early infection signals.",
    ],
    "organic": [
        "Use approved copper or sulfur-based protective sprays where appropriate.",
        "Apply biological fungicides such as Bacillus-based products during early disease pressure.",
    ],
    "chemical": [
        "Use a locally approved fungicide program matched to the detected disease and crop stage.",
        "Rotate fungicide mode of action groups to reduce resistance risk.",
    ],
    "research_notes": (
        "ALeaf-STENet combines EfficientNetV2-S texture features, CBAM attention, "
        "and Swin Transformer global context before producing class and severity heads."
    ),
}


DISEASE_PROFILES = {
    "Alternaria leaf spot": {
        "scientific_name": "Alternaria mali",
        "symptoms": [
            "Small brown to purplish spots that expand into necrotic lesions.",
            "Yellow halos around older lesions under warm and humid conditions.",
            "Severe cases can cause premature leaf drop and reduced tree vigor.",
        ],
        "preventive": [
            "Remove fallen infected leaves and prune crowded canopy areas.",
            "Avoid overhead irrigation during warm humid periods.",
            "Use balanced fertilization to reduce stress-driven susceptibility.",
        ],
        "organic": [
            "Apply copper-based protective sprays before heavy disease pressure.",
            "Use Bacillus subtilis or similar biological fungicide programs early.",
        ],
        "chemical": [
            "Apply labeled protectant fungicides such as mancozeb where permitted.",
            "Rotate systemic fungicides with different FRAC groups during repeated sprays.",
        ],
        "research_notes": (
            "The hybrid feature stack is useful for Alternaria because lesion margins and "
            "halo texture require both local CNN features and larger leaf-context attention."
        ),
    },
    "Brown spot": {
        "scientific_name": "Marssonina coronaria / Diplocarpon mali complex",
        "symptoms": [
            "Brown irregular spots that may merge across the leaf surface.",
            "Yellowing around infected areas followed by defoliation in severe outbreaks.",
            "Lesions often appear after extended wetness and dense canopy humidity.",
        ],
        "preventive": [
            "Collect and destroy fallen leaves to reduce overwintering inoculum.",
            "Improve sunlight penetration through pruning.",
            "Avoid excessive nitrogen that promotes tender susceptible growth.",
        ],
        "organic": [
            "Use copper sprays at early symptom appearance where label guidance allows.",
            "Support biological protection with compost extracts or approved biofungicides.",
        ],
        "chemical": [
            "Use labeled apple leaf spot fungicides during rainy infection windows.",
            "Repeat applications according to label interval and local extension guidance.",
        ],
        "research_notes": (
            "Brown spot prediction relies strongly on tissue discoloration and coalesced "
            "spot boundaries captured by the severity regression module."
        ),
    },
    "Frogeye leaf spot": {
        "scientific_name": "Botryosphaeria obtusa",
        "symptoms": [
            "Circular brown spots with tan centers and darker purple margins.",
            "Frog-eye ring patterns on mature lesions.",
            "Associated black rot cankers or mummified fruit may be present nearby.",
        ],
        "preventive": [
            "Prune and destroy cankered wood and mummified fruit.",
            "Avoid mechanical wounds on limbs and fruiting wood.",
            "Remove infected debris from orchard rows before wet spring periods.",
        ],
        "organic": [
            "Use copper products during dormant or early-season windows when appropriate.",
            "Add biological fungicides as a supplement in lower pressure blocks.",
        ],
        "chemical": [
            "Use captan or other labeled protectants during infection periods.",
            "Follow local recommendations for black rot and frog-eye leaf spot control.",
        ],
        "research_notes": (
            "Frog-eye lesions have strong ring-like morphology, which the Swin attention "
            "branch helps preserve even when spots are scattered across the blade."
        ),
    },
    "Grey spot": {
        "scientific_name": "Pestalotiopsis / apple leaf spot complex",
        "symptoms": [
            "Grey to ash-colored lesions with darker borders.",
            "Leaf tissue may become brittle around older necrotic areas.",
            "Symptoms intensify under high humidity and poor canopy airflow.",
        ],
        "preventive": [
            "Maintain open canopy structure and remove diseased leaves.",
            "Limit prolonged leaf wetness through irrigation management.",
            "Sanitize pruning tools when moving between infected trees.",
        ],
        "organic": [
            "Apply approved copper products as protectants during early disease pressure.",
            "Use biological fungicides in rotation where available.",
        ],
        "chemical": [
            "Use locally labeled broad-spectrum fungicides for leaf spot complexes.",
            "Rotate active ingredients and avoid repeated single-mode sprays.",
        ],
        "research_notes": (
            "Grey spot severity is inferred from necrotic coverage and contrast against "
            "remaining green tissue after feature fusion."
        ),
    },
    "Health": {
        "scientific_name": "Malus domestica - uninfected foliage",
        "symptoms": [
            "Uniform green chlorophyll distribution across the leaf blade.",
            "No visible fungal lesions, rust pustules, or chlorotic mosaic patterns.",
            "Normal vein structure and intact leaf margins.",
        ],
        "preventive": [
            "Continue periodic monitoring after rain, irrigation, or pest pressure.",
            "Maintain balanced nutrition and avoid water stress.",
            "Keep records of healthy baseline samples for future model comparison.",
        ],
        "organic": [
            "Support tree resilience with compost, mulch, and soil biology management.",
            "Avoid unnecessary pesticide use when the leaf remains healthy.",
        ],
        "chemical": [
            "No chemical treatment is recommended for a healthy prediction.",
            "Use fungicides only when disease pressure and local guidance justify protection.",
        ],
        "research_notes": (
            "The model registers low lesion density and low attention concentration, "
            "which drives the severity module toward the healthy range."
        ),
    },
    "Mosaic": {
        "scientific_name": "Apple mosaic virus",
        "symptoms": [
            "Irregular pale yellow, cream, or mosaic-like patches on leaves.",
            "Pattern often follows vein sectors rather than circular fungal lesions.",
            "Affected leaves may show reduced photosynthetic area and vigor.",
        ],
        "preventive": [
            "Use virus-tested nursery stock and certified propagation material.",
            "Remove severely affected trees if infection is confirmed and spreading.",
            "Sanitize grafting tools and avoid propagating from symptomatic trees.",
        ],
        "organic": [
            "There is no curative organic spray for viral infection.",
            "Improve tree vigor with balanced soil and water management.",
        ],
        "chemical": [
            "Chemical fungicides do not cure viral mosaic disease.",
            "Manage insect vectors only if local scouting confirms vector pressure.",
        ],
        "research_notes": (
            "Mosaic patterns are spatially diffuse, so the transformer branch contributes "
            "global pattern context beyond local lesion texture."
        ),
    },
    "Powdery mildew": {
        "scientific_name": "Podosphaera leucotricha",
        "symptoms": [
            "White to grey powdery fungal growth on leaves and young shoots.",
            "Leaf curling, distortion, and reduced shoot growth.",
            "Infected buds may weaken the following bloom cycle.",
        ],
        "preventive": [
            "Prune infected shoots during dormant pruning.",
            "Open dense canopy zones to improve airflow and sunlight.",
            "Plant resistant cultivars where mildew pressure is recurring.",
        ],
        "organic": [
            "Apply sulfur, potassium bicarbonate, or horticultural oil where label permits.",
            "Use biological products preventively during humid growth periods.",
        ],
        "chemical": [
            "Use labeled DMI or SDHI fungicides during tight cluster through early shoot growth.",
            "Rotate fungicide groups to manage resistance.",
        ],
        "research_notes": (
            "Powdery mildew is texture-heavy, so EfficientNetV2-S local filters provide "
            "strong features before CBAM recalibrates the infected regions."
        ),
    },
    "Rust": {
        "scientific_name": "Gymnosporangium juniperi-virginianae",
        "symptoms": [
            "Bright orange to yellow spots on upper leaf surfaces.",
            "Small dark fruiting bodies can appear within orange lesions.",
            "Tube-like aecia may form on the underside in favorable humidity.",
        ],
        "preventive": [
            "Reduce nearby cedar or juniper alternate hosts where practical.",
            "Choose rust-resistant cultivars for high-risk orchard blocks.",
            "Protect foliage during spring spore-release periods.",
        ],
        "organic": [
            "Use sulfur or copper products preventively at early infection risk.",
            "Maintain canopy airflow to reduce wetness duration.",
        ],
        "chemical": [
            "Use labeled sterol inhibitor fungicides during cedar rust infection windows.",
            "Follow local extension timing for green tip through petal fall protection.",
        ],
        "research_notes": (
            "Rust lesions are high-contrast orange structures, producing strong class "
            "confidence when the model sees clustered circular pustules."
        ),
    },
    "Scab": {
        "scientific_name": "Venturia inaequalis",
        "symptoms": [
            "Olive-green to brown velvety spots with feathery lesion edges.",
            "Leaf puckering, chlorosis, and premature leaf drop in severe infections.",
            "Older lesions may become necrotic and dark along the central blade.",
        ],
        "preventive": [
            "Remove fallen leaf litter to disrupt overwintering ascospore cycles.",
            "Prune canopy density to improve sunlight and airflow.",
            "Plant scab-resistant cultivars when establishing new blocks.",
        ],
        "organic": [
            "Apply sulfur or copper sprays early in wet spring conditions.",
            "Use Bacillus-based biocontrols as part of a preventive program.",
        ],
        "chemical": [
            "Use labeled DMI, dodine, captan, or strobilurin fungicides at risk periods.",
            "Time sprays according to infection forecasts and label intervals.",
        ],
        "research_notes": (
            "Scab combines subtle texture and lesion boundaries, matching the reason "
            "ALeaf-STENet fuses CNN texture extraction with transformer context."
        ),
    },
}


def get_profile(class_name: str) -> dict:
    """Return the closest profile for a predicted class name."""

    if not class_name:
        return DEFAULT_PROFILE

    normalized = class_name.strip().lower()
    for key, value in DISEASE_PROFILES.items():
        if key.lower() == normalized:
            return value

    for key, value in DISEASE_PROFILES.items():
        if key.lower() in normalized or normalized in key.lower():
            return value

    return DEFAULT_PROFILE


def severity_level(severity: float) -> str:
    """Map a severity percentage to the UI severity state."""

    if severity <= 3:
        return "Healthy"
    if severity < 30:
        return "Low"
    if severity < 60:
        return "Moderate"
    return "Severe"
