import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Support both `python -m LNIAGIA.DB.SQLLite.DataGenerator` and direct execution.
try:
    from LNIAGIA.DB.models import (
        STATIC_GLOBAL_FIELDS,
        TYPE_FIELDS,
        EXTRA_FIELD_VALUES,
        get_random_brand_for_type,
        generate_price_for_item,
        generate_age_groups,
        generate_body_types,
        get_valid_genders_for_type,
        filter_by_age_appropriateness,
        get_weighted_season_for_type,
        get_weighted_material_for_season,
        get_weighted_pattern_for_style,
        get_valid_occasion_for_type,
        get_valid_insulation_for_season,
    )
except ModuleNotFoundError:
    db_dir = Path(__file__).resolve().parent.parent
    if str(db_dir) not in sys.path:
        sys.path.insert(0, str(db_dir))

    from models import (
        STATIC_GLOBAL_FIELDS,
        TYPE_FIELDS,
        EXTRA_FIELD_VALUES,
        get_random_brand_for_type,
        generate_price_for_item,
        generate_age_groups,
        generate_body_types,
        get_valid_genders_for_type,
        filter_by_age_appropriateness,
        get_weighted_season_for_type,
        get_weighted_material_for_season,
        get_weighted_pattern_for_style,
        get_valid_occasion_for_type,
        get_valid_insulation_for_season,
    )

# ═══ CONFIG ═══
N = 10000
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataSources")


def generate_item():
    """
    Generates a single clothing item with realistic constraints.
    
    Applies 6 types of realistic constraints:
    1. Gender-Type: Skirts/dresses only female/unisex
    2. Age-Appropriate: Filters occasions, patterns, necklines by age
    3. Season-Type: Weights season choice by item type
    4. Material-Season: Weights materials appropriate for season
    5. Style-Pattern: Weights patterns that fit the style
    6. Occasion-Type: Filters occasions appropriate for type
    
    Returns:
        dict: A clothing item with all fields
    """
    # ═══ STEP 1: Pick type (drives many other choices) ═══
    item_type = random.choice(STATIC_GLOBAL_FIELDS["type"])
    
    # ═══ STEP 2: Pick gender (constrained by type) ═══
    # Constraint #1: Skirts/dresses only female or unisex
    valid_genders = get_valid_genders_for_type(item_type)
    gender = random.choice(valid_genders)
    
    # ═══ STEP 3: Generate age groups ═══
    # Uses smart probability logic (50% single, 30% two groups, etc.)
    age_group = generate_age_groups()
    
    # ═══ STEP 4: Pick season (weighted by type) ═══
    # Constraint #3: Summer items more likely in summer, winter items in winter
    season = get_weighted_season_for_type(item_type)
    
    # ═══ STEP 5: Pick material (weighted by season) ═══
    # Constraint #4: Lightweight materials in summer, warm materials in winter
    material = get_weighted_material_for_season(season)
    
    # ═══ STEP 6: Pick style ═══
    # Filter out age-inappropriate styles
    valid_styles = [s for s in STATIC_GLOBAL_FIELDS["style"] 
                    if filter_by_age_appropriateness("style", s, age_group)]
    style = random.choice(valid_styles) if valid_styles else random.choice(STATIC_GLOBAL_FIELDS["style"])
    
    # ═══ STEP 7: Pick pattern (weighted by style, filtered by age) ═══
    # Constraint #5: Patterns that fit the style (formal=plain/striped, bohemian=floral/tie-dye)
    # Constraint #2: Age-appropriate patterns (no graphic/tie-dye for babies)
    pattern = get_weighted_pattern_for_style(style)
    # Re-roll if not age-appropriate (max 10 attempts)
    for _ in range(10):
        if filter_by_age_appropriateness("pattern", pattern, age_group):
            break
        pattern = get_weighted_pattern_for_style(style)
    
    # ═══ STEP 8: Pick occasion (constrained by type, filtered by age) ═══
    # Constraint #6: Beach only for light items, formal only for dresses/trousers
    # Constraint #2: No work/parties for babies
    occasion = get_valid_occasion_for_type(item_type)
    # Re-roll if not age-appropriate (max 10 attempts)
    for _ in range(10):
        if filter_by_age_appropriateness("occasion", occasion, age_group):
            break
        occasion = get_valid_occasion_for_type(item_type)
    
    # ═══ STEP 9: Pick remaining simple fields ═══
    color = random.choice(STATIC_GLOBAL_FIELDS["color"])
    fit = random.choice(STATIC_GLOBAL_FIELDS["fit"])
    
    # ═══ STEP 10: Assemble the item ═══
    item = {
        "type": item_type,
        "color": color,
        "style": style,
        "pattern": pattern,
        "material": material,
        "fit": fit,
        "gender": gender,
        "age_group": age_group,
        "season": season,
        "occasion": occasion,
    }

    # ═══ STEP 11: Brand depends on type ═══
    brand = get_random_brand_for_type(item_type)
    item["brand"] = brand

    # ═══ STEP 12: Price depends on type + brand ═══
    item["price"] = generate_price_for_item(item_type, brand)

    # ═══ STEP 13: Type-specific extra fields ═══
    for field in TYPE_FIELDS.get(item_type, ()):
        # Special handling for age-filtered fields
        if field == "neckline":
            # Constraint #2: Age-appropriate necklines
            valid_necklines = [n for n in EXTRA_FIELD_VALUES[field]
                              if filter_by_age_appropriateness("neckline", n, age_group)]
            item[field] = random.choice(valid_necklines) if valid_necklines else random.choice(EXTRA_FIELD_VALUES[field])
        
        elif field == "insulation":
            # Constraint #4: Season-appropriate insulation
            item[field] = get_valid_insulation_for_season(season)
        
        else:
            # All other fields: random choice
            item[field] = random.choice(EXTRA_FIELD_VALUES[field])

    # ═══ STEP 14: Body type(s) the garment flatters ═══
    # Constraint #7: depends on gender pool + cut (fit/dress_style/leg/waist).
    # Computed last so it can read the type-specific fields set above.
    item["body_type"] = generate_body_types(item)

    return item


def generate_dataset(n):
    return [{"id": i, **generate_item()} for i in range(1, n + 1)]


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_FOLDER, f"{timestamp}.json")

    print(f"Generating {N} items...")
    data = generate_dataset(N)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {N} items to: {output_file}")


if __name__ == "__main__":
    main()
