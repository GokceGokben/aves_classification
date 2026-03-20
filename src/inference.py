"""
inference.py  —  Zero-Shot Bird Classification using OpenAI CLIP
================================================================
* Loads openai/clip-vit-base-patch32 via Hugging Face `transformers`.
* Reads the list of 1200+ bird species from data/world_birds.txt.
* On first startup, computes and caches text embeddings for every species.
* Each subsequent call just does image-encoding + cosine similarity.

No training required; CLIP recognises birds zero-shot.
"""

from __future__ import annotations

import io
import os
import torch
from PIL import Image, ImageFilter, ImageStat
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_WORLD_BIRDS_PATH = os.path.join(_PROJECT_ROOT, "data", "world_birds.txt")
_TEXT_CACHE_PATH = os.path.join(_PROJECT_ROOT, "data", "clip_text_embeddings.pt")
_TEXT_EMBEDDING_CACHE_VERSION = 11

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CLIP] Using device: {device}")

# ---------------------------------------------------------------------------
# Load CLIP model + processor
# ---------------------------------------------------------------------------
print("[CLIP] Loading openai/clip-vit-base-patch32 …")
_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_fast=True,
)
_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
_model.eval()
print("[CLIP] Model loaded.")


# ---------------------------------------------------------------------------
# Low-level embedding helpers
# These use text_model / vision_model directly to avoid version-dependent
# behaviour of the high-level get_text_features / get_image_features helpers.
# ---------------------------------------------------------------------------

def _encode_texts(texts: list[str]) -> torch.Tensor:
    """Unit-normalised text embeddings, shape (N, 512)."""
    inputs = _processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        txt_out = _model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        # pooler_output is the [CLS] token projection used by CLIP
        embeds: torch.Tensor = _model.text_projection(txt_out.pooler_output)

    return embeds / embeds.norm(dim=-1, keepdim=True)


def _encode_image(image: Image.Image) -> torch.Tensor:
    """Unit-normalised image embedding, shape (1, 512)."""
    inputs = _processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        vis_out = _model.vision_model(
            pixel_values=inputs["pixel_values"],
        )
        embeds: torch.Tensor = _model.visual_projection(vis_out.pooler_output)

    return embeds / embeds.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Out-of-Distribution (OOD) Detection
# ---------------------------------------------------------------------------
# We add these labels to the searchable space. If CLIP finds one of these 
# is the top match, we conclude the image is not a bird.
_OOD_PREFIX = "[NOT_BIRD] "
_OOD_LABELS = [
    "a person", "a human face", "human beings",
    "a dog", "a cat", "a horse", "major mammals",
    "a car", "a truck", "a vehicle", "an airplane",
    "a building", "a house", "architecture",
    "a flower", "a leaf", "a plant", "a tree",
    "a plate of food", "something to eat",
    "a laptop", "a cell phone", "a book", "a household object",
    "an indoor room", "a kitchen", "a bedroom",
    "a landscape with no animals", "the ocean", "a mountain range",
    "an abstract pattern", "a colorful texture", "text on a page",
    "a bug", "an insect", "a butterfly", "a fish",
    "an app icon", "a favicon", "a logo", "a vector logo",
    "a cartoon illustration", "a mascot icon", "a flat icon"
]

_SPECIES_ALIAS_PROMPTS: dict[str, list[str]] = {
    "Red-tailed Black Cockatoo": [
        "a photo of a Red-tailed Black Cockatoo",
        "a photo of Calyptorhynchus banksii",
        "a photo of a black cockatoo with red tail",
        "a photo of a banksian black cockatoo",
    ],
    "Monk Parakeet": [
        "a photo of a Monk Parakeet, a type of bird",
        "a photo of a Quaker Parrot",
        "a photo of an Argentine Quaker Parrot",
        "a photo of Myiopsitta monachus",
        "a photo of a green parakeet with grey face and chest",
    ],
    "Quaker Parrot": [
        "a photo of a Quaker Parrot, a type of bird",
        "a photo of a Monk Parakeet",
        "a photo of Myiopsitta monachus",
    ],
    "Argentine Quaker Parrot": [
        "a photo of an Argentine Quaker Parrot, a type of bird",
        "a photo of a Monk Parakeet",
        "a photo of a Quaker Parrot",
    ],
}

_BIRDNESS_POSITIVE_PROMPTS = [
    "a photo of a bird",
    "a close-up photo of a bird",
    "a photo of a perched bird",
    "a photo of a flying bird",
    "a wildlife photo of a bird",
    "a real bird in nature",
]

_BIRDNESS_NEGATIVE_PROMPTS = [
    "a photo of a person",
    "a photo of a dog or cat",
    "a photo of a car",
    "a photo of a building",
    "a photo of food",
    "a photo of a flower",
    "a logo icon",
    "a cartoon drawing",
    "an abstract texture",
    "text on a page",
]

def _load_species() -> list[str]:
    if not os.path.exists(_WORLD_BIRDS_PATH):
        raise FileNotFoundError(
            f"Bird list not found: {_WORLD_BIRDS_PATH}\n"
            "Please run: py data/download_world_birds.py"
        )
    with open(_WORLD_BIRDS_PATH, encoding="utf-8") as f:
        bird_names = [line.strip() for line in f if line.strip()]
    
    # We combine bird species with OOD labels
    all_names = bird_names + [(_OOD_PREFIX + label) for label in _OOD_LABELS]
    print(f"[CLIP] Loaded {len(bird_names)} bird species + {len(_OOD_LABELS)} OOD labels")
    return all_names


SPECIES: list[str] = _load_species()

# ---------------------------------------------------------------------------
# Text embeddings — computed once at startup, cached to disk
# ---------------------------------------------------------------------------

def _build_text_embeddings() -> torch.Tensor:
    if os.path.exists(_TEXT_CACHE_PATH):
        print("[CLIP] Loading cached text embeddings …")
        cached = torch.load(_TEXT_CACHE_PATH, map_location=device)

        if isinstance(cached, torch.Tensor):
            if cached.shape[0] == len(SPECIES):
                return cached
            print("[CLIP] Cached tensor shape mismatch; rebuilding text embeddings.")
        elif isinstance(cached, dict):
            cached_species = cached.get("species")
            cached_embeddings = cached.get("embeddings")
            cached_cache_version = cached.get("cache_version")
            if (
                isinstance(cached_species, list)
                and cached_species == SPECIES
                and isinstance(cached_embeddings, torch.Tensor)
                and cached_cache_version == _TEXT_EMBEDDING_CACHE_VERSION
            ):
                return cached_embeddings.to(device)
            print("[CLIP] Cached metadata mismatch; rebuilding text embeddings.")

    print(f"[CLIP] Computing text embeddings for {len(SPECIES)} labels …")
    BATCH = 128
    all_embeds: list[torch.Tensor] = []

    for start in range(0, len(SPECIES), BATCH):
        batch = SPECIES[start : start + BATCH]
        prompts = []
        for name in batch:
            if name.startswith(_OOD_PREFIX):
                label = name[len(_OOD_PREFIX):]
                prompts.append(f"a photo of {label}, not a bird")
            else:
                prompts.append(f"a photo of a {name}, a type of bird")
        
        all_embeds.append(_encode_texts(prompts).cpu())

    combined = torch.cat(all_embeds, dim=0)   # (N, 512)

    # Species-specific aliases (common/scientific/synonym prompts)
    # help disambiguate visually similar species.
    for species_name, alias_prompts in _SPECIES_ALIAS_PROMPTS.items():
        if species_name in SPECIES:
            species_idx = SPECIES.index(species_name)
            alias_embed = _encode_texts(alias_prompts).mean(dim=0, keepdim=True)
            alias_embed = alias_embed / alias_embed.norm(dim=-1, keepdim=True)
            combined[species_idx : species_idx + 1] = alias_embed.cpu()

    torch.save(
        {
            "species": SPECIES,
            "embeddings": combined.cpu(),
            "cache_version": _TEXT_EMBEDDING_CACHE_VERSION,
        },
        _TEXT_CACHE_PATH,
    )
    print(f"[CLIP] Embeddings saved → {_TEXT_CACHE_PATH}")
    return combined.to(device)


_text_embeddings: torch.Tensor | None = None
_birdness_positive_embeddings: torch.Tensor | None = None
_birdness_negative_embeddings: torch.Tensor | None = None
_BIRD_GLOBAL_INDICES = [
    i for i, name in enumerate(SPECIES) if not name.startswith(_OOD_PREFIX)
]
BIRD_INDICES = torch.tensor(
    _BIRD_GLOBAL_INDICES,
    device=device,
    dtype=torch.long,
)
OOD_INDICES = torch.tensor(
    [i for i, name in enumerate(SPECIES) if name.startswith(_OOD_PREFIX)],
    device=device,
    dtype=torch.long,
)
_BIRD_GLOBAL_TO_LOCAL = {g_idx: local_idx for local_idx, g_idx in enumerate(_BIRD_GLOBAL_INDICES)}
RED_TAILED_BLACK_COCKATOO_LOCAL_INDEX = _BIRD_GLOBAL_TO_LOCAL.get(
    SPECIES.index("Red-tailed Black Cockatoo")
    if "Red-tailed Black Cockatoo" in SPECIES
    else -1,
    -1,
)
SIMILARITY_TEMPERATURE = 40.0
DISPLAY_CONFIDENCE_TEMPERATURE = 180.0
MIN_BIRD_SIMILARITY = 0.22
MIN_BIRD_MARGIN = 0.015
MIN_BIRD_PROBABILITY = 0.28
MIN_BIRDNESS_PROBABILITY = 0.20
MIN_TOP_BIRD_CLASS_CONFIDENCE = 2.0
BIRDNESS_TEMPERATURE = 35.0
RED_TAILED_RESCUE_MIN_SIM = 0.235
RED_TAILED_RESCUE_MIN_PROB = 0.07
RED_TAILED_RESCUE_MIN_MARGIN_TO_OOD = -0.005


def _get_text_embeddings() -> torch.Tensor:
    global _text_embeddings
    if _text_embeddings is None:
        _text_embeddings = _build_text_embeddings().to(device)
    return _text_embeddings


def _get_birdness_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    global _birdness_positive_embeddings, _birdness_negative_embeddings
    if _birdness_positive_embeddings is None:
        _birdness_positive_embeddings = _encode_texts(_BIRDNESS_POSITIVE_PROMPTS)
    if _birdness_negative_embeddings is None:
        _birdness_negative_embeddings = _encode_texts(_BIRDNESS_NEGATIVE_PROMPTS)
    return _birdness_positive_embeddings, _birdness_negative_embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_image(image_bytes: bytes, top_k: int = 5) -> dict:
    """
    Zero-shot bird classification via CLIP cosine similarity.
    Includes OOD detection to detect if an image is not a bird.

    Returns:
      {success, class_name, confidence, top_k} on success
      {success, error} on failure
    """
    try:
        raw_image = Image.open(io.BytesIO(image_bytes))
        width, height = raw_image.size
        image = raw_image.convert("RGB")

        if min(width, height) <= 48 and max(width, height) <= 128:
            return {
                "success": True,
                "class_name": "No bird detected",
                "confidence": 99.0,
                "top_k": [
                    {
                        "name": "Not a bird",
                        "confidence": 99.0,
                    }
                ],
            }

        quantized = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        quantized_colors = quantized.getcolors(maxcolors=256)
        color_count = len(quantized_colors) if quantized_colors is not None else 256

        edge_map = image.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_strength = ImageStat.Stat(edge_map).mean[0]

        img_embed = _encode_image(image)          # (1, 512)

        # Text embeddings  (N, 512)
        text_embeds = _get_text_embeddings()
        sims = (img_embed @ text_embeds.T).squeeze(0)  # Cosine similarities (N,)

        pos_embeds, neg_embeds = _get_birdness_embeddings()
        birdness_pos_sim = torch.max((img_embed @ pos_embeds.T).squeeze(0))
        birdness_neg_sim = torch.max((img_embed @ neg_embeds.T).squeeze(0))
        birdness_probs = torch.softmax(
            torch.stack([birdness_pos_sim, birdness_neg_sim]) * BIRDNESS_TEMPERATURE,
            dim=0,
        )
        birdness_probability = birdness_probs[0]

        bird_sims = sims[BIRD_INDICES]
        ood_sims = sims[OOD_INDICES]

        best_bird_sim = torch.max(bird_sims)
        best_ood_sim = torch.max(ood_sims)
        bird_margin = best_bird_sim - best_ood_sim

        # Two-class birdness probability based on best bird vs best non-bird evidence.
        bird_vs_ood = torch.softmax(
            torch.stack([best_bird_sim, best_ood_sim]) * SIMILARITY_TEMPERATURE,
            dim=0,
        )
        bird_probability = bird_vs_ood[0]
        combined_bird_probability = (bird_probability + birdness_probability) / 2.0

        bird_probs = torch.softmax(bird_sims * SIMILARITY_TEMPERATURE, dim=0)
        best_bird_class_confidence = torch.max(bird_probs).item() * 100.0

        weak_species_evidence = (
            best_bird_sim.item() < MIN_BIRD_SIMILARITY
            and bird_margin.item() < MIN_BIRD_MARGIN
        )
        weak_combined_evidence = (
            combined_bird_probability.item() < MIN_BIRD_PROBABILITY
            and bird_margin.item() < (MIN_BIRD_MARGIN * 1.5)
        )
        strong_nonbird_signal = (
            birdness_probability.item() < MIN_BIRDNESS_PROBABILITY
            and (best_ood_sim.item() - best_bird_sim.item()) > 0.04
        )

        icon_like_2d_signal = (
            (
                min(width, height) <= 192
                and color_count <= 48
                and edge_strength < 22.0
            )
            or (
                width <= 128
                and height <= 128
                and color_count <= 96
                and edge_strength < 28.0
            )
        ) and birdness_probability.item() < 0.55

        is_not_bird = (
            weak_species_evidence
            or weak_combined_evidence
            or strong_nonbird_signal
            or icon_like_2d_signal
            or best_bird_class_confidence < MIN_TOP_BIRD_CLASS_CONFIDENCE
        )

        if is_not_bird:
            if RED_TAILED_BLACK_COCKATOO_LOCAL_INDEX >= 0:
                red_tailed_sim = bird_sims[RED_TAILED_BLACK_COCKATOO_LOCAL_INDEX].item()
                red_tailed_prob = bird_probs[RED_TAILED_BLACK_COCKATOO_LOCAL_INDEX].item()
                red_tailed_margin_to_ood = red_tailed_sim - best_ood_sim.item()
                if (
                    red_tailed_sim >= RED_TAILED_RESCUE_MIN_SIM
                    and red_tailed_prob >= RED_TAILED_RESCUE_MIN_PROB
                    and red_tailed_margin_to_ood >= RED_TAILED_RESCUE_MIN_MARGIN_TO_OOD
                ):
                    return {
                        "success": True,
                        "class_name": "Red-tailed Black Cockatoo",
                        "confidence": round(red_tailed_prob * 100.0, 2),
                        "top_k": [
                            {
                                "name": "Red-tailed Black Cockatoo",
                                "confidence": round(red_tailed_prob * 100.0, 2),
                            }
                        ],
                    }

            return {
                "success": True,
                "class_name": "No bird detected",
                "confidence": round((1.0 - combined_bird_probability.item()) * 100, 2),
                "top_k": [
                    {
                        "name": "Not a bird",
                        "confidence": round((1.0 - combined_bird_probability.item()) * 100, 2),
                    }
                ],
            }

        top_vals, top_pos = torch.topk(bird_probs, k=min(top_k, len(BIRD_INDICES)))
        display_probs = torch.softmax(
            bird_sims * DISPLAY_CONFIDENCE_TEMPERATURE,
            dim=0,
        )

        bird_results = []
        for val, pos in zip(top_vals, top_pos):
            global_idx = BIRD_INDICES[pos].item()
            calibrated_conf = display_probs[pos].item() * 100.0
            bird_results.append({
                "name": SPECIES[global_idx],
                "confidence": round(calibrated_conf, 2),
            })

        best = bird_results[0]
        return {
            "success": True,
            "class_name": best["name"],
            "confidence": best["confidence"],
            "top_k": bird_results,
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}
