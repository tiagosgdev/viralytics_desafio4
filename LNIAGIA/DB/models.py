# ═══ GLOBAL ═══
TYPE = (
    "short_sleeve_top", "long_sleeve_top", "long_sleeve_outwear", "vest",
    "shorts", "trousers", "skirt", "short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"
)

COLOR = (
    "black", "white", "gray", "navy", "blue", "red", "green",
    "yellow", "orange", "pink", "purple", "brown", "beige",
    "cream", "burgundy", "olive", "teal", "coral", "multicolor"
)

STYLE = (
    "casual", "formal", "smart casual", "sporty", "bohemian",
    "minimalist", "streetwear", "vintage", "elegant", "preppy"
)

PATTERN = (
    "plain", "striped", "checkered", "plaid", "floral",
    "polka dot", "geometric", "abstract", "animal print",
    "camouflage", "tie-dye", "graphic", "embroidered"
)

MATERIAL = (
    "cotton", "polyester", "linen", "silk", "wool", "denim",
    "leather", "suede", "velvet", "satin", "chiffon", "fleece",
    "cashmere", "nylon", "rayon", "spandex", "organic cotton"
)

FIT = (
    "slim fit", "regular", "relaxed", "oversized", "tailored",
    "loose", "fitted", "athletic", "baggy", "cropped"
)

GENDER = (
    "male", "female", "unisex"
)

# ═══ BODY TYPE ═══
# Body shape(s) a garment is designed to flatter. Values are gender-specific.
FEMALE_BODY_TYPES = (
    "hourglass", "pear", "triangle", "rectangle", "inverted_triangle", "apple"
)

MALE_BODY_TYPES = (
    "trapezoid", "rectangle", "inverted_triangle", "triangle", "oval"
)

# Shapes common to both pools — used for unisex items so the value stays neutral.
UNISEX_BODY_TYPES = tuple(bt for bt in FEMALE_BODY_TYPES if bt in MALE_BODY_TYPES)
# -> ("triangle", "rectangle", "inverted_triangle")

BODY_TYPES_BY_GENDER = {
    "female": FEMALE_BODY_TYPES,
    "male": MALE_BODY_TYPES,
    "unisex": UNISEX_BODY_TYPES,
}

# Full set of body-type values (union, no duplicates) — useful for validation.
BODY_TYPE = tuple(dict.fromkeys(FEMALE_BODY_TYPES + MALE_BODY_TYPES))

AGE_GROUP = (
    "baby", "child", "teenager", "young adult", "adult", "senior"
)

# Age ranges for each group (for future use)
AGE_GROUP_RANGES = {
    "baby": (0, 2),
    "child": (3, 12),
    "teenager": (13, 17),
    "young adult": (18, 29),
    "adult": (30, 59),
    "senior": (60, 120)
}

SEASON = (
    "spring", "summer", "autumn", "winter", "all-season"
)

OCCASION = (
    "everyday", "work", "party", "wedding", "beach",
    "sport", "date night", "travel", "lounge", "formal event"
)

# ═══ TOPS ═══
NECKLINE = (
    "crew neck", "v-neck", "scoop neck", "boat neck",
    "turtleneck", "mock neck", "off-shoulder", "square neck",
    "halter", "sweetheart", "cowl neck", "collared"
)

COLLAR = (
    "none", "pointed", "spread", "button-down", "mandarin",
    "cuban", "peter pan", "shawl", "notched"
)

SLEEVE_STYLE = (
    "regular", "puff", "bell", "raglan", "cap", "dolman",
    "bishop", "lantern", "rolled", "cuffed"
)

HEM_STYLE = (
    "straight", "curved", "cropped", "asymmetric", "raw edge",
    "knotted", "split", "longline"
)

# ═══ OUTWEAR ═══
CLOSURE = (
    "zipper", "buttons", "snap buttons", "velcro", "toggle",
    "belt", "open front", "double breasted"
)

HOOD = (
    "none", "attached", "detachable", "faux fur trim"
)

INSULATION = (
    "none", "light", "medium", "heavy", "down", "synthetic"
)

WATERPROOF = (
    "none", "water resistant", "waterproof"
)

OUTWEAR_POCKETS = (
    "none", "side pockets", "chest pocket", "interior pockets",
    "zippered pockets", "flap pockets", "patch pockets",
    "hand warmer pockets", "multiple pockets"
)

# ═══ BOTTOMS ═══
WAIST = (
    "xs", "small", "medium", "large", "xl", "xxl"
)

WAIST_STYLE = (
    "regular", "high-waisted", "mid-rise", "low-rise",
    "elastic", "drawstring", "belted", "paper bag"
)

RISE = (
    "low-rise", "mid-rise", "high-rise"
)

LENGTH = (
    "mini", "short", "knee-length", "midi", "ankle",
    "full length", "maxi", "cropped"
)

LEG_STYLE = (
    "straight", "skinny", "slim", "bootcut", "wide leg",
    "flared", "tapered", "jogger", "cargo"
)

BOTTOM_POCKETS = (
    "none", "side pockets", "back pockets", "cargo pockets",
    "zippered pockets", "hidden pockets", "coin pocket",
    "welt pockets", "patch pockets"
)

# ═══ DRESSES ═══
DRESS_STYLE = (
    "a-line", "bodycon", "shift", "wrap", "maxi",
    "midi", "mini", "shirt dress", "slip dress",
    "fit and flare", "empire", "sheath"
)


# ═══════════════════════════════════════════════════════
# ═══ REALISTIC CONSTRAINTS FOR DATA GENERATION ═══
# ═══════════════════════════════════════════════════════

# ═══ 1. GENDER-TYPE CONSTRAINTS ═══
# Certain types are only suitable for specific genders
GENDER_CONSTRAINTS_BY_TYPE = {
    "skirt": ("female", "unisex"),  # Never male
    "short_sleeve_dress": ("female", "unisex"),
    "long_sleeve_dress": ("female", "unisex"),
    "vest_dress": ("female", "unisex"),
    "sling_dress": ("female", "unisex"),
    # All other types allow all genders
}

# ═══ 2. AGE-APPROPRIATE CONSTRAINTS ═══
# Certain values should be excluded for specific age groups

# Occasions inappropriate for babies/children
AGE_INAPPROPRIATE_OCCASIONS = {
    "baby": ("work", "party", "wedding", "formal event", "date night"),
    "child": ("work", "date night", "formal event"),
    "teenager": ("formal event",),  # Rare but not impossible
}

# Patterns too complex or inappropriate for babies
AGE_INAPPROPRIATE_PATTERNS = {
    "baby": ("graphic", "tie-dye", "embroidered", "abstract"),
}

# Necklines inappropriate for young age groups
AGE_INAPPROPRIATE_NECKLINES = {
    "baby": ("halter", "off-shoulder", "sweetheart", "cowl neck", "v-neck"),
    "child": ("halter", "off-shoulder", "sweetheart", "cowl neck"),
    "teenager": ("off-shoulder",),  # Less common
}

# Styles less common for certain ages
AGE_INAPPROPRIATE_STYLES = {
    "baby": ("formal", "smart casual", "elegant", "vintage", "preppy"),
    "senior": ("streetwear",),  # Less common but not impossible
}

# ═══ 3. SEASON-TYPE PROBABILITIES ═══
# Certain types are more/less common in specific seasons
# Format: {season: {type: probability_weight}}
SEASON_TYPE_WEIGHTS = {
    "summer": {
        "short_sleeve_top": 3.0,
        "long_sleeve_top": 0.3,
        "long_sleeve_outwear": 0.2,
        "vest": 2.0,
        "shorts": 3.5,
        "trousers": 0.8,
        "skirt": 2.0,
        "short_sleeve_dress": 3.0,
        "long_sleeve_dress": 0.3,
        "vest_dress": 2.5,
        "sling_dress": 3.5,
    },
    "winter": {
        "short_sleeve_top": 0.4,
        "long_sleeve_top": 2.5,
        "long_sleeve_outwear": 3.5,
        "vest": 2.0,
        "shorts": 0.3,
        "trousers": 3.0,
        "skirt": 1.5,
        "short_sleeve_dress": 0.3,
        "long_sleeve_dress": 2.5,
        "vest_dress": 1.0,
        "sling_dress": 0.2,
    },
    "spring": {
        "short_sleeve_top": 2.0,
        "long_sleeve_top": 2.0,
        "long_sleeve_outwear": 2.0,
        "vest": 1.5,
        "shorts": 2.0,
        "trousers": 2.0,
        "skirt": 2.0,
        "short_sleeve_dress": 2.0,
        "long_sleeve_dress": 2.0,
        "vest_dress": 1.8,
        "sling_dress": 2.0,
    },
    "autumn": {
        "short_sleeve_top": 1.2,
        "long_sleeve_top": 2.5,
        "long_sleeve_outwear": 3.0,
        "vest": 2.0,
        "shorts": 1.0,
        "trousers": 2.5,
        "skirt": 2.0,
        "short_sleeve_dress": 1.2,
        "long_sleeve_dress": 2.5,
        "vest_dress": 1.5,
        "sling_dress": 0.8,
    },
    "all-season": {
        # All types equally likely for all-season
        "short_sleeve_top": 1.0,
        "long_sleeve_top": 1.0,
        "long_sleeve_outwear": 1.0,
        "vest": 1.0,
        "shorts": 1.0,
        "trousers": 1.0,
        "skirt": 1.0,
        "short_sleeve_dress": 1.0,
        "long_sleeve_dress": 1.0,
        "vest_dress": 1.0,
        "sling_dress": 1.0,
    }
}

# ═══ 4. MATERIAL-SEASON PREFERENCES ═══
# Certain materials are more suitable for specific seasons
# Format: {season: (preferred_materials)}
SEASON_PREFERRED_MATERIALS = {
    "summer": (
        "cotton", "linen", "chiffon", "satin", "organic cotton", 
        "rayon", "polyester"
    ),
    "winter": (
        "wool", "fleece", "cashmere", "velvet", "leather", 
        "suede", "nylon", "polyester"
    ),
    "spring": (
        "cotton", "linen", "denim", "polyester", "organic cotton",
        "rayon", "chiffon"
    ),
    "autumn": (
        "wool", "cotton", "denim", "suede", "velvet",
        "fleece", "polyester", "nylon"
    ),
    "all-season": MATERIAL,  # All materials valid
}

# Insulation should match season
SEASON_INSULATION_CONSTRAINTS = {
    "summer": ("none", "light"),
    "spring": ("none", "light", "medium"),
    "autumn": ("light", "medium", "heavy"),
    "winter": ("medium", "heavy", "down", "synthetic"),
    "all-season": ("none", "light", "medium"),
}

# ═══ 5. STYLE-PATTERN COHERENCE ═══
# Certain patterns fit better with specific styles
# Format: {style: (preferred_patterns)}
STYLE_PREFERRED_PATTERNS = {
    "formal": (
        "plain", "striped", "checkered", "plaid"
    ),
    "elegant": (
        "plain", "floral", "abstract", "embroidered"
    ),
    "casual": PATTERN,  # All patterns work
    "smart casual": (
        "plain", "striped", "checkered", "plaid", "geometric"
    ),
    "sporty": (
        "plain", "striped", "geometric", "graphic", "camouflage"
    ),
    "bohemian": (
        "floral", "tie-dye", "embroidered", "abstract", "geometric"
    ),
    "minimalist": (
        "plain", "geometric", "abstract"
    ),
    "streetwear": (
        "plain", "graphic", "camouflage", "tie-dye", "geometric", "abstract"
    ),
    "vintage": (
        "floral", "polka dot", "striped", "checkered", "plaid", "embroidered"
    ),
    "preppy": (
        "plain", "striped", "checkered", "plaid", "polka dot"
    ),
}

# ═══ 6. OCCASION-TYPE CONSTRAINTS ═══
# Certain types don't make sense for specific occasions
OCCASION_TYPE_CONSTRAINTS = {
    "beach": (
        "shorts", "short_sleeve_top", "vest",
        "short_sleeve_dress", "sling_dress", "skirt"
    ),
    "sport": (
        "short_sleeve_top", "long_sleeve_top", "shorts", 
        "trousers", "long_sleeve_outwear", "vest"
    ),
    "formal event": (
        "long_sleeve_dress", "short_sleeve_dress", "vest_dress",
        "trousers", "skirt", "long_sleeve_outwear", "long_sleeve_top"
    ),
    "wedding": (
        "long_sleeve_dress", "short_sleeve_dress", "vest_dress", "sling_dress",
        "trousers", "skirt", "long_sleeve_outwear"
    ),
    # Other occasions allow all types
}


# ═══ 7. BODY-TYPE FLATTERING LOGIC ═══
# Which body shapes a garment flatters, inferred from its cut.
# Each garment attribute (fit, dress_style, leg_style, waist_style) contributes
# affinity points to the body types it flatters. The best-scoring shapes within
# the item's gender pool are chosen. The names below span both gender pools;
# they are intersected with the gender-appropriate pool at selection time, so a
# shape that doesn't exist for the item's gender is simply ignored.

# How the overall fit of a garment flatters different builds.
FIT_BODYTYPE_AFFINITY = {
    "slim fit":  ("hourglass", "trapezoid", "rectangle", "inverted_triangle"),
    "slim":      ("hourglass", "trapezoid", "rectangle", "inverted_triangle"),
    "fitted":    ("hourglass", "trapezoid", "inverted_triangle", "rectangle"),
    "tailored":  ("hourglass", "trapezoid", "rectangle", "inverted_triangle", "pear", "triangle"),
    "athletic":  ("trapezoid", "inverted_triangle", "hourglass"),
    "regular":   ("rectangle", "trapezoid", "triangle", "oval", "pear", "hourglass"),
    "relaxed":   ("rectangle", "apple", "oval", "triangle", "pear"),
    "loose":     ("apple", "oval", "triangle", "rectangle", "pear"),
    "oversized": ("rectangle", "apple", "oval", "triangle"),
    "baggy":     ("rectangle", "apple", "oval", "triangle"),
    "cropped":   ("hourglass", "rectangle", "inverted_triangle", "trapezoid"),
}

# Dress silhouettes and the shapes they balance.
DRESS_STYLE_BODYTYPE_AFFINITY = {
    "a-line":        ("pear", "apple", "inverted_triangle", "hourglass", "rectangle"),
    "fit and flare": ("pear", "apple", "hourglass", "rectangle", "triangle"),
    "empire":        ("apple", "pear", "rectangle", "inverted_triangle"),
    "bodycon":       ("hourglass", "rectangle"),
    "sheath":        ("hourglass", "rectangle", "trapezoid"),
    "shift":         ("rectangle", "apple", "oval", "inverted_triangle"),
    "wrap":          ("hourglass", "pear", "apple", "triangle", "rectangle"),
    "maxi":          ("pear", "apple", "rectangle", "hourglass", "triangle"),
    "midi":          ("rectangle", "hourglass", "pear"),
    "mini":          ("hourglass", "rectangle", "inverted_triangle", "trapezoid"),
    "shirt dress":   ("rectangle", "apple", "oval", "triangle"),
    "slip dress":    ("hourglass", "rectangle", "inverted_triangle"),
}

# Trouser/short leg cuts and the builds they balance (esp. hips vs. shoulders).
LEG_STYLE_BODYTYPE_AFFINITY = {
    "skinny":   ("hourglass", "trapezoid", "rectangle", "inverted_triangle"),
    "slim":     ("hourglass", "trapezoid", "rectangle", "inverted_triangle"),
    "straight": ("rectangle", "trapezoid", "triangle", "oval", "hourglass", "pear"),
    "bootcut":  ("pear", "triangle", "inverted_triangle", "hourglass"),
    "wide leg": ("pear", "triangle", "inverted_triangle", "apple", "rectangle"),
    "flared":   ("pear", "triangle", "inverted_triangle", "hourglass"),
    "tapered":  ("trapezoid", "rectangle", "inverted_triangle", "oval", "triangle"),
    "jogger":   ("rectangle", "oval", "triangle", "trapezoid"),
    "cargo":    ("rectangle", "triangle", "oval", "pear"),
}

# Waistline treatments and the shapes they define/forgive.
WAIST_STYLE_BODYTYPE_AFFINITY = {
    "high-waisted": ("hourglass", "pear", "apple", "rectangle", "triangle"),
    "mid-rise":     ("rectangle", "hourglass", "trapezoid", "pear"),
    "low-rise":     ("hourglass", "trapezoid", "inverted_triangle", "rectangle"),
    "elastic":      ("apple", "oval", "triangle", "rectangle", "pear"),
    "drawstring":   ("apple", "oval", "triangle", "rectangle"),
    "belted":       ("hourglass", "rectangle", "apple", "pear"),
    "paper bag":    ("rectangle", "apple", "pear"),
    # "regular" intentionally omitted — carries no strong flattering signal.
}

# Maps an item field to the affinity table that scores it.
BODYTYPE_AFFINITY_SOURCES = {
    "fit": FIT_BODYTYPE_AFFINITY,
    "dress_style": DRESS_STYLE_BODYTYPE_AFFINITY,
    "leg_style": LEG_STYLE_BODYTYPE_AFFINITY,
    "waist_style": WAIST_STYLE_BODYTYPE_AFFINITY,
}

# How many shapes a garment flatters (weighted; capped at the gender-pool size).
BODYTYPE_COUNT_WEIGHTS = {1: 0.20, 2: 0.40, 3: 0.30, 4: 0.10}


# ═══ FIELD DEFINITIONS ═══

# All global field names (including dynamic ones)
GLOBAL_FIELDS = (
    "type", "color", "style", "pattern", "material", "fit",
    "gender", "body_type", "age_group", "season", "occasion", "brand", "price",
)

# Static global fields (have predefined value options)
STATIC_GLOBAL_FIELDS = {
    "type": TYPE,
    "color": COLOR,
    "style": STYLE,
    "pattern": PATTERN,
    "material": MATERIAL,
    "fit": FIT,
    "gender": GENDER,
    "age_group": AGE_GROUP,
    "season": SEASON,
    "occasion": OCCASION,
}

# Dynamic global fields (values depend on type or are generated)
DYNAMIC_GLOBAL_FIELDS = (
    "brand",  # depends on type
    "price",  # depends on type + brand
)

TYPE_FIELDS = {
    # ═══ TOPS ═══
    "short_sleeve_top": (
        "neckline", "collar", "sleeve_style", "hem_style"
    ),
    "long_sleeve_top": (
        "neckline", "collar", "sleeve_style", "hem_style"
    ),
    "vest": (
        "neckline", "hem_style"
    ),

    # ═══ OUTWEAR ═══
    "long_sleeve_outwear": (
        "closure", "hood", "insulation", "waterproof", 
        "outwear_pockets", "collar", "sleeve_style"
    ),
    
    # ═══ BOTTOMS ═══
    "shorts": (
        "waist", "waist_style", "rise", "length", 
        "leg_style", "bottom_pockets"
    ),
    "trousers": (
        "waist", "waist_style", "rise", "length", 
        "leg_style", "bottom_pockets"
    ),
    "skirt": (
        "waist", "waist_style", "length"
    ),
    
    # ═══ DRESSES ═══
    "short_sleeve_dress": (
        "dress_style", "neckline", "length", "waist_style", 
        "sleeve_style", "hem_style"
    ),
    "long_sleeve_dress": (
        "dress_style", "neckline", "length", "waist_style", 
        "sleeve_style", "hem_style"
    ),
    "vest_dress": (
        "dress_style", "neckline", "length", "waist_style", "hem_style"
    ),
    "sling_dress": (
        "dress_style", "neckline", "length", "waist_style", "hem_style"
    )
}

# Extra field values (for type-specific fields)
EXTRA_FIELD_VALUES = {
    # Tops
    "neckline": NECKLINE,
    "collar": COLLAR,
    "sleeve_style": SLEEVE_STYLE,
    "hem_style": HEM_STYLE,
    
    # Outwear
    "closure": CLOSURE,
    "hood": HOOD,
    "insulation": INSULATION,
    "waterproof": WATERPROOF,
    "outwear_pockets": OUTWEAR_POCKETS,
    
    # Bottoms
    "waist": WAIST,
    "waist_style": WAIST_STYLE,
    "rise": RISE,
    "length": LENGTH,
    "leg_style": LEG_STYLE,
    "bottom_pockets": BOTTOM_POCKETS,
    
    # Dresses
    "dress_style": DRESS_STYLE,
}

# ═══ BRAND TIERS ═══
# Each brand belongs to a pricing tier that affects the final price

BRAND_TIERS = {
    # ═══ BUDGET (0.5x - 0.8x multiplier) ═══
    "budget": (
        "H&M", "Primark", "Shein", "Pull&Bear", "Bershka",
        "Missguided", "PrettyLittleThing", "Boohoo", "ASOS",
        "Forever 21", "Old Navy", "Hollister", "American Eagle"
    ),
    
    # ═══ MID-RANGE (0.9x - 1.2x multiplier) - baseline ═══
    "mid": (
        "Zara", "Mango", "Uniqlo", "Gap", "Topshop",
        "& Other Stories", "COS", "Everlane", "Levi's",
        "Wrangler", "Lee", "Dockers", "Carhartt", "Dickies",
        "Nike", "Adidas", "Puma", "New Balance", "Under Armour",
        "Columbia", "Superdry", "Abercrombie & Fitch"
    ),
    
    # ═══ PREMIUM (1.5x - 2.5x multiplier) ═══
    "premium": (
        "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein", "Lacoste",
        "Hugo Boss", "Ted Baker", "Reiss", "Karen Millen",
        "Banana Republic", "J.Crew", "Massimo Dutti", "Bonobos",
        "Theory", "Vince", "AllSaints", "The North Face", "Patagonia",
        "Arc'teryx", "Lululemon", "Sandro", "Maje", "Ba&sh",
        "Ganni", "Reformation", "Free People", "Anthropologie",
        "Barbour", "Schott NYC", "Alpha Industries", "Woolrich"
    ),
    
    # ═══ LUXURY (3x - 6x multiplier) ═══
    "luxury": (
        "Burberry", "Moncler", "Canada Goose", "Armani Exchange",
        "Diane von Furstenberg", "Self-Portrait", "Zimmermann",
        "Mackage", "Moose Knuckles", "Herno", "Belstaff",
        "Brioni", "Incotex", "PT Torino", "Brooks Brothers",
        "Charles Tyrwhitt", "Eileen Fisher"
    ),
    
    # ═══ ULTRA LUXURY (6x - 12x multiplier) ═══
    "ultra_luxury": (
        "Gucci", "Prada", "Versace", "Dolce & Gabbana",
        "Saint Laurent", "Balenciaga", "Givenchy", "Valentino",
        "Bottega Veneta", "Loro Piana", "Brunello Cucinelli",
        "Max Mara", "Acne Studios"
    )
}

# Multiplier ranges per tier (min_mult, max_mult)
TIER_MULTIPLIERS = {
    "budget": (0.4, 0.7),
    "mid": (0.85, 1.3),
    "premium": (1.8, 3.0),
    "luxury": (3.5, 6.0),
    "ultra_luxury": (7.0, 15.0)
}

# ═══ BASE PRICE RANGES PER TYPE (EUR) ═══
# These are "mid-tier" baseline prices
BASE_PRICE_PER_TYPE = {
    # ═══ TOPS ═══
    "short_sleeve_top": (15.00, 45.00),
    "long_sleeve_top": (20.00, 55.00),
    "vest": (18.00, 40.00),

    # ═══ OUTWEAR ═══
    "long_sleeve_outwear": (60.00, 180.00),
    
    # ═══ BOTTOMS ═══
    "shorts": (18.00, 45.00),
    "trousers": (30.00, 75.00),
    "skirt": (22.00, 55.00),
    
    # ═══ DRESSES ═══
    "short_sleeve_dress": (35.00, 85.00),
    "long_sleeve_dress": (45.00, 100.00),
    "vest_dress": (38.00, 80.00),
    "sling_dress": (30.00, 75.00)
}

DEFAULT_BASE_PRICE = (20.00, 60.00)

# ═══ BRANDS PER TYPE ═══
BRANDS_PER_TYPE = {
    # ═══ TOPS ═══
    "short_sleeve_top": (
        # Budget
        "H&M", "Primark", "Pull&Bear", "Bershka", "ASOS",
        "Forever 21", "Hollister", "American Eagle",
        # Mid
        "Zara", "Mango", "Uniqlo", "Gap", "Nike", "Adidas",
        "Puma", "Levi's", "Superdry", "Abercrombie & Fitch",
        # Premium
        "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein", "Lacoste",
        "Hugo Boss", "Massimo Dutti", "J.Crew", "COS", "Everlane",
        # Luxury
        "Armani Exchange", "Burberry",
        # Ultra Luxury
        "Gucci", "Prada", "Balenciaga"
    ),
    "long_sleeve_top": (
        # Budget
        "H&M", "Primark", "Pull&Bear", "Bershka", "ASOS",
        "Forever 21", "Hollister", "American Eagle",
        # Mid
        "Zara", "Mango", "Uniqlo", "Gap", "Nike", "Adidas",
        "Superdry", "Abercrombie & Fitch",
        # Premium
        "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein", "Lacoste",
        "Hugo Boss", "Massimo Dutti", "J.Crew", "Theory", "AllSaints",
        "COS", "Everlane", "Bonobos",
        # Luxury
        "Armani Exchange", "Burberry", "Brooks Brothers",
        # Ultra Luxury
        "Gucci", "Prada", "Brunello Cucinelli"
    ),
    "vest": (
        # Budget
        "H&M", "Primark", "ASOS",
        # Mid
        "Uniqlo", "Gap", "Zara",
        # Premium
        "Patagonia", "The North Face", "Ralph Lauren",
        "Tommy Hilfiger", "COS", "J.Crew", "Everlane",
        # Luxury
        "Canada Goose", "Moncler"
    ),

    # ═══ OUTWEAR ═══
    "long_sleeve_outwear": (
        # Budget
        "H&M", "Primark", "ASOS", "Boohoo",
        # Mid
        "Zara", "Uniqlo", "Nike", "Adidas", "Puma", "New Balance",
        "Levi's", "Carhartt", "Columbia", "Superdry",
        # Premium
        "The North Face", "Patagonia", "Arc'teryx", "Ralph Lauren",
        "Tommy Hilfiger", "Hugo Boss", "AllSaints", "Barbour",
        "Schott NYC", "Alpha Industries", "Woolrich", "COS",
        # Luxury
        "Canada Goose", "Moncler", "Burberry", "Mackage",
        "Moose Knuckles", "Herno", "Belstaff",
        # Ultra Luxury
        "Gucci", "Prada", "Balenciaga", "Saint Laurent",
        "Acne Studios", "Max Mara"
    ),
    
    # ═══ BOTTOMS ═══
    "shorts": (
        # Budget
        "H&M", "Primark", "ASOS", "Pull&Bear", "Hollister",
        "American Eagle",
        # Mid
        "Zara", "Uniqlo", "Gap", "Nike", "Adidas", "Puma",
        "Levi's", "Wrangler", "Carhartt",
        # Premium
        "Patagonia", "The North Face", "Ralph Lauren",
        "Tommy Hilfiger", "Lululemon", "J.Crew",
        # Luxury
        "Armani Exchange"
    ),
    "trousers": (
        # Budget
        "H&M", "Primark", "ASOS", "Pull&Bear", "Bershka",
        # Mid
        "Zara", "Uniqlo", "Gap", "Levi's", "Dockers", "Lee",
        "Wrangler", "Mango",
        # Premium
        "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein",
        "Hugo Boss", "Massimo Dutti", "J.Crew", "Bonobos",
        "Theory", "COS", "Reiss",
        # Luxury
        "Incotex", "PT Torino", "Burberry",
        # Ultra Luxury
        "Brioni", "Gucci", "Prada"
    ),
    "skirt": (
        # Budget
        "H&M", "Primark", "ASOS", "Missguided", "PrettyLittleThing",
        "Boohoo", "Shein",
        # Mid
        "Zara", "Mango", "Topshop", "& Other Stories",
        # Premium
        "COS", "Reformation", "Free People", "Anthropologie",
        "Massimo Dutti", "Sandro", "Maje", "Ba&sh", "Reiss",
        "Ted Baker", "Karen Millen",
        # Luxury
        "Self-Portrait", "Zimmermann",
        # Ultra Luxury
        "Gucci", "Prada", "Max Mara"
    ),
    
    # ═══ DRESSES ═══
    "short_sleeve_dress": (
        # Budget
        "H&M", "Primark", "ASOS", "Missguided", "PrettyLittleThing",
        "Boohoo", "Shein",
        # Mid
        "Zara", "Mango", "Topshop", "& Other Stories",
        # Premium
        "COS", "Reformation", "Free People", "Anthropologie",
        "Sandro", "Maje", "Ba&sh", "Ted Baker", "Karen Millen",
        "Reiss", "Ganni",
        # Luxury
        "Diane von Furstenberg", "Self-Portrait", "Zimmermann",
        # Ultra Luxury
        "Gucci", "Valentino", "Dolce & Gabbana"
    ),
    "long_sleeve_dress": (
        # Budget
        "H&M", "Primark", "ASOS", "Missguided", "PrettyLittleThing",
        "Boohoo",
        # Mid
        "Zara", "Mango", "& Other Stories",
        # Premium
        "COS", "Reformation", "Anthropologie", "Sandro", "Maje",
        "Ba&sh", "Ted Baker", "Karen Millen", "Reiss", "Ganni",
        # Luxury
        "Diane von Furstenberg", "Self-Portrait", "Zimmermann",
        # Ultra Luxury
        "Gucci", "Valentino", "Max Mara", "Acne Studios"
    ),
    "vest_dress": (
        # Budget
        "H&M", "Primark", "ASOS",
        # Mid
        "Zara", "Mango", "Topshop", "& Other Stories",
        # Premium
        "COS", "Reformation", "Everlane", "Theory", "Vince",
        "Karen Millen", "Eileen Fisher",
        # Luxury
        "Self-Portrait"
    ),
    "sling_dress": (
        # Budget
        "H&M", "Primark", "ASOS", "Missguided", "PrettyLittleThing",
        "Boohoo", "Shein",
        # Mid
        "Zara", "Mango", "Topshop",
        # Premium
        "Reformation", "Free People", "Rat & Boa",
        "Sandro", "Maje",
        # Luxury
        "Zimmermann", "Self-Portrait",
        # Ultra Luxury
        "Gucci", "Balenciaga"
    )
}

# ═════════ USER PROFILE SCHEMA ═════════

USER_AGE_GROUP = AGE_GROUP  # reuse existing system

USER_GENDER = GENDER

# Users don't need all values, but we sample from REAL item space
USER_PROFILE_FIELDS = {
    "favorite_colors": COLOR,
    "favorite_styles": STYLE,
    "favorite_materials": MATERIAL,
    "preferred_seasons": SEASON,
    "preferred_occasions": OCCASION,
}

# ═════════ REAL-WORLD DISTRIBUTION WEIGHTS ═════════

AGE_GROUP_WEIGHTS = {
    "baby": 0.05,
    "child": 0.10,
    "teenager": 0.15,
    "young adult": 0.30,
    "adult": 0.30,
    "senior": 0.10
}

GENDER_WEIGHTS = {
    "male": 0.49,
    "female": 0.49,
    "unisex": 0.02
}

STYLE_WEIGHTS = {
    "casual": 0.25,
    "streetwear": 0.15,
    "smart casual": 0.12,
    "sporty": 0.12,
    "minimalist": 0.10,
    "elegant": 0.10,
    "formal": 0.08,
    "vintage": 0.05,
    "bohemian": 0.02,
    "preppy": 0.01
}

COLOR_WEIGHTS = {
    "black": 0.12,
    "white": 0.10,
    "gray": 0.10,
    "navy": 0.10,
    "blue": 0.10,
    "beige": 0.08,
    "cream": 0.06,
    "green": 0.06,
    "red": 0.05,
    "brown": 0.05,
    "pink": 0.05,
    "yellow": 0.04,
    "orange": 0.03,
    "purple": 0.03,
    "multicolor": 0.02
}

# ═══ TYPE → AGE-GROUP AFFINITY (coherence) ═══
# Primary-age affinity per type. Keys MUST be exact TYPE values; the inner
# keys MUST be exact AGE_GROUP values. Weights are relative (normalized at
# draw time). This replaces the old uniform primary-age pick so dresses /
# work tops / outwear never get a baby/senior primary, while broad casual
# basics keep a real spread (incl. kids).
TYPE_AGE_GROUP_AFFINITY = {
    # dresses: overwhelmingly young-adult/adult, a little teen; never baby/child/senior
    "short_sleeve_dress": {"young adult": 5, "adult": 4, "teenager": 1.5},
    "long_sleeve_dress":  {"young adult": 5, "adult": 4, "teenager": 1.5},
    "vest_dress":         {"young adult": 5, "adult": 4, "teenager": 1.5},
    "sling_dress":        {"young adult": 6, "adult": 3, "teenager": 1.5},  # party-ish, no senior
    # smart/work tops & outwear: adult-centric
    "long_sleeve_top":    {"adult": 5, "young adult": 4, "teenager": 1.5, "senior": 1.0},
    "long_sleeve_outwear": {"adult": 5, "young adult": 4, "senior": 1.5, "teenager": 1.0},
    "trousers":           {"adult": 5, "young adult": 4, "senior": 1.5, "teenager": 1.0},
    # broadly-aged casual basics: keep a real spread (incl. kids) so not everything is adult
    "short_sleeve_top":   {"young adult": 4, "adult": 4, "teenager": 2, "child": 1.5, "senior": 1, "baby": 0.5},
    "vest":               {"young adult": 4, "adult": 3, "teenager": 2, "child": 1.5, "senior": 1},
    "shorts":             {"young adult": 4, "adult": 3, "teenager": 2, "child": 2, "senior": 0.8, "baby": 0.5},
    "skirt":              {"young adult": 5, "adult": 3, "teenager": 2, "child": 1, "senior": 0.8},
}
# Fallback (item_type=None) = the global age marginal (back-compat for other callers).
DEFAULT_AGE_AFFINITY = {a: AGE_GROUP_WEIGHTS[a] for a in AGE_GROUP}

# ═══ TYPE → STYLE BOOST (coherence) ═══
# Multiplies STYLE_WEIGHTS so smart types lean smart/minimalist and dress
# types lean elegant/streetwear. Keys MUST be exact TYPE values, inner keys
# exact STYLE values (note "smart casual" has a space). Styles not listed for
# a type keep boost 1.0, so every age-valid style stays possible (spread is
# preserved for Workstream B's veto_batch).
STYLE_TYPE_BOOST = {
    "long_sleeve_top":     {"minimalist": 2.5, "smart casual": 2.5, "formal": 1.5, "casual": 1.2},
    "long_sleeve_outwear": {"smart casual": 2, "minimalist": 2, "casual": 1.5, "sporty": 1.2},
    "trousers":            {"smart casual": 2, "minimalist": 1.8, "casual": 1.8, "formal": 1.5},
    "skirt":               {"casual": 1.8, "minimalist": 1.8, "elegant": 1.5, "vintage": 1.3},
    "shorts":              {"casual": 2.5, "sporty": 2, "streetwear": 1.5},
    "short_sleeve_dress":  {"elegant": 2.5, "streetwear": 1.8, "vintage": 1.5, "casual": 1.2},
    "long_sleeve_dress":   {"elegant": 2.5, "vintage": 1.8, "formal": 1.5},
    "vest_dress":          {"elegant": 2, "streetwear": 1.8, "casual": 1.5},
    "sling_dress":         {"elegant": 2.5, "streetwear": 2, "vintage": 1.5},
}

# ═══ HELPER FUNCTIONS ═══

import random

# ═══ BRAND & PRICE HELPERS ═══

def get_brand_tier(brand):
    """Returns the pricing tier for a brand"""
    for tier, brands in BRAND_TIERS.items():
        if brand in brands:
            return tier
    return "mid"


def get_brands_for_type(item_type):
    """Returns available brands for a type"""
    return BRANDS_PER_TYPE.get(item_type, ())


def get_random_brand_for_type(item_type):
    """Returns a random brand for a type"""
    brands = get_brands_for_type(item_type)
    return random.choice(brands) if brands else None


def get_base_price_range(item_type):
    """Returns base (min, max) price range for a type in EUR"""
    return BASE_PRICE_PER_TYPE.get(item_type, DEFAULT_BASE_PRICE)


def get_price_range_for_brand_and_type(item_type, brand):
    """Returns the realistic (min, max) price range for a specific brand + type combination."""
    base_min, base_max = get_base_price_range(item_type)
    tier = get_brand_tier(brand)
    mult_min, mult_max = TIER_MULTIPLIERS[tier]
    
    return (
        round(base_min * mult_min, 2),
        round(base_max * mult_max, 2)
    )


def round_to_retail_price(price):
    """Rounds price to common retail price endings"""
    if price < 10:
        endings = [0.95, 0.99]
    elif price < 100:
        endings = [0.00, 0.95, 0.99]
    else:
        endings = [0.00, 0.99]
    
    base = int(price)
    ending = random.choice(endings)
    
    return round(base + ending, 2)


def generate_price_for_item(item_type, brand):
    """Generates a realistic price based on type AND brand tier."""
    base_min, base_max = get_base_price_range(item_type)
    tier = get_brand_tier(brand)
    mult_min, mult_max = TIER_MULTIPLIERS[tier]
    
    base_price = random.uniform(base_min, base_max)
    multiplier = random.uniform(mult_min, mult_max)
    final_price = base_price * multiplier
    
    return round_to_retail_price(final_price)


# ═══ AGE GROUP HELPERS ═══

def generate_age_groups(item_type=None):
    """
    Generates a list of age groups for an item.

    Rules:
    - Most items (50%) have 1 age group
    - 30% have 2 adjacent groups
    - 15% have 3 adjacent groups
    - 4% have 4 adjacent groups
    - 0.8% have 5 adjacent groups
    - 0.2% have all 6 groups (universal items)
    - Groups are ordered by popularity (primary first)
    - Only adjacent groups are combined (no baby+senior)
    - The PRIMARY group is drawn from TYPE_AGE_GROUP_AFFINITY[item_type]
      (type-conditioned coherence); item_type=None falls back to the global
      age marginal (DEFAULT_AGE_AFFINITY) for back-compat. Only the primary
      selection changed; the multi-group count + adjacency logic is unchanged.

    Args:
        item_type (str | None): Item type used to condition the primary age.

    Returns:
        str: Comma-separated age groups (e.g., "adult, young adult")
    """
    age_groups_list = list(AGE_GROUP)
    
    # Determine number of groups with weighted probability
    rand = random.random()
    if rand < 0.50:  # 50% - single group
        num_groups = 1
    elif rand < 0.80:  # 30% - two groups
        num_groups = 2
    elif rand < 0.95:  # 15% - three groups
        num_groups = 3
    elif rand < 0.99:  # 4% - four groups
        num_groups = 4
    elif rand < 0.998:  # 0.8% - five groups
        num_groups = 5
    else:  # 0.2% - all six groups (universal item)
        num_groups = 6
    
    # If all groups, return them all
    if num_groups == 6:
        return ", ".join(age_groups_list)
    
    # Pick a primary age group from the type-conditioned affinity table.
    affinity = TYPE_AGE_GROUP_AFFINITY.get(item_type, DEFAULT_AGE_AFFINITY)
    primary_group = random.choices(
        list(affinity.keys()), weights=list(affinity.values()), k=1
    )[0]
    primary_idx = age_groups_list.index(primary_group)
    selected_indices = [primary_idx]
    
    # Add adjacent groups if needed
    if num_groups > 1:
        # Determine how many on each side
        remaining = num_groups - 1
        
        # Try to spread evenly, but respect boundaries
        left_available = primary_idx
        right_available = len(age_groups_list) - 1 - primary_idx
        
        # Distribute remaining slots
        for _ in range(remaining):
            # Randomly decide to go left or right, but respect boundaries
            can_go_left = left_available > 0 and (primary_idx - left_available) not in selected_indices
            can_go_right = right_available > 0 and (primary_idx + (len(selected_indices) - left_available)) not in selected_indices
            
            if can_go_left and can_go_right:
                if random.random() < 0.5:
                    selected_indices.append(primary_idx - (len([i for i in selected_indices if i < primary_idx]) + 1))
                else:
                    selected_indices.append(primary_idx + (len([i for i in selected_indices if i > primary_idx]) + 1))
            elif can_go_left:
                selected_indices.append(primary_idx - (len([i for i in selected_indices if i < primary_idx]) + 1))
            elif can_go_right:
                selected_indices.append(primary_idx + (len([i for i in selected_indices if i > primary_idx]) + 1))
        
        # Ensure we have contiguous indices
        selected_indices.sort()
        # Make them contiguous around primary
        min_idx = max(0, primary_idx - (num_groups // 2))
        max_idx = min(len(age_groups_list), min_idx + num_groups)
        if max_idx - min_idx < num_groups:
            min_idx = max(0, max_idx - num_groups)
        selected_indices = list(range(min_idx, max_idx))[:num_groups]
        
        # Reorder to put primary first
        if primary_idx in selected_indices:
            ordered_indices = [primary_idx]
            # Add adjacent groups in alternating pattern (closer ones first)
            left = [i for i in selected_indices if i < primary_idx]
            right = [i for i in selected_indices if i > primary_idx]
            left.reverse()  # Closest first
            
            # Alternate between left and right
            while left or right:
                if right:
                    ordered_indices.append(right.pop(0))
                if left:
                    ordered_indices.append(left.pop(0))
            selected_indices = ordered_indices
    
    # Convert indices to age group names
    selected_groups = [age_groups_list[i] for i in selected_indices]
    
    return ", ".join(selected_groups)


# ═══ REALISTIC CONSTRAINT HELPERS ═══

def get_valid_genders_for_type(item_type):
    """
    Returns valid genders for a given item type.
    
    Constraint #1: Gender-Type Rules
    - Skirts and dresses are only for female/unisex (never male)
    - Slings are predominantly female/unisex
    - All other types allow all genders
    
    Args:
        item_type (str): The type of clothing item
        
    Returns:
        tuple: Valid gender options for this type
    """
    return GENDER_CONSTRAINTS_BY_TYPE.get(item_type, GENDER)


def generate_body_types(item, rng=random):
    """
    Picks the body shape(s) a garment is designed to flatter, based on its cut.

    Constraint #7: Body-Type Flattering Logic
    - Values are gender-specific: female pool, male pool, or — for unisex items —
      only the shapes shared by both pools, so the value stays neutral.
    - Garment attributes (fit, dress_style, leg_style, waist_style) add affinity
      points to the body types they flatter; the best-scoring shapes win, with a
      base score so every shape in the pool stays possible.
    - Returns 1-4 shapes (weighted, capped at pool size), ordered best-fit first
      so the primary shape leads — mirroring how age_group lists its primary first.

    Args:
        item (dict): Item with at least 'gender'; optionally fit / dress_style /
            leg_style / waist_style for smarter, cut-aware assignment.
        rng (random.Random): RNG to use. Pass a seeded instance (e.g.
            random.Random(item_id)) for reproducible, store-consistent output.

    Returns:
        str: Comma-separated body types (e.g. "pear, apple, hourglass"),
             or "" if the gender has no body-type pool.
    """
    gender = item.get("gender", "unisex")
    pool = BODY_TYPES_BY_GENDER.get(gender, UNISEX_BODY_TYPES)
    if not pool:
        return ""

    # Base score keeps every pool member eligible even with no garment signal.
    scores = {bt: 1.0 for bt in pool}

    # Add affinity points from each garment attribute that carries a signal.
    for field, table in BODYTYPE_AFFINITY_SOURCES.items():
        value = item.get(field)
        if not value:
            continue
        for body_type in table.get(value, ()):  # value -> flattered shapes
            if body_type in scores:
                scores[body_type] += 2.0

    # Decide how many shapes to assign (never more than the pool holds).
    counts = [c for c in BODYTYPE_COUNT_WEIGHTS if c <= len(pool)]
    weights = [BODYTYPE_COUNT_WEIGHTS[c] for c in counts]
    num = rng.choices(counts, weights=weights, k=1)[0]

    # Weighted sampling without replacement — higher-scoring shapes favored.
    candidates = list(scores.keys())
    candidate_weights = [scores[bt] for bt in candidates]
    chosen = []
    for _ in range(num):
        pick = rng.choices(candidates, weights=candidate_weights, k=1)[0]
        idx = candidates.index(pick)
        candidates.pop(idx)
        candidate_weights.pop(idx)
        chosen.append(pick)

    # Order best-fit first (primary shape leads), like age_group.
    chosen.sort(key=lambda bt: scores[bt], reverse=True)
    return ", ".join(chosen)


def filter_by_age_appropriateness(field_name, value, age_groups_str):
    """
    Checks if a value is appropriate for the given age groups.
    
    Constraint #2: Age-Appropriate Filters
    - Filters out inappropriate occasions, patterns, necklines, and styles
    - Based on the primary (first) age group in the list
    
    Args:
        field_name (str): The field being checked (e.g., "occasion", "pattern")
        value: The value to check
        age_groups_str (str): Comma-separated age groups (e.g., "adult, young adult")
        
    Returns:
        bool: True if appropriate, False if should be filtered out
    """
    # Get primary age group (first in the list)
    primary_age = age_groups_str.split(", ")[0]
    
    # Check each constraint type
    if field_name == "occasion":
        inappropriate = AGE_INAPPROPRIATE_OCCASIONS.get(primary_age, ())
        return value not in inappropriate
    
    elif field_name == "pattern":
        inappropriate = AGE_INAPPROPRIATE_PATTERNS.get(primary_age, ())
        return value not in inappropriate
    
    elif field_name == "neckline":
        inappropriate = AGE_INAPPROPRIATE_NECKLINES.get(primary_age, ())
        return value not in inappropriate
    
    elif field_name == "style":
        inappropriate = AGE_INAPPROPRIATE_STYLES.get(primary_age, ())
        return value not in inappropriate
    
    # If no constraint for this field, allow all values
    return True


def get_weighted_style_for_type(item_type, age_groups_str):
    """
    Picks a style coherent with the item type, weighted by STYLE_WEIGHTS and a
    per-type boost (STYLE_TYPE_BOOST), restricted to age-appropriate styles.

    - Builds the set of age-valid styles (via filter_by_age_appropriateness).
    - Weights each by STYLE_WEIGHTS[style] * STYLE_TYPE_BOOST[type][style]
      (boost defaults to 1.0 when not listed, so spread is preserved).
    - Falls back to all styles if age filtering leaves nothing.

    Args:
        item_type (str): The type of clothing item.
        age_groups_str (str): Comma-separated age groups (primary first).

    Returns:
        str: A style appropriate for this type + age.
    """
    valid = [s for s in STYLE
             if filter_by_age_appropriateness("style", s, age_groups_str)]
    if not valid:
        valid = list(STYLE)
    boost = STYLE_TYPE_BOOST.get(item_type, {})
    weights = [STYLE_WEIGHTS.get(s, 0.01) * boost.get(s, 1.0) for s in valid]
    return random.choices(valid, weights=weights, k=1)[0]


def get_weighted_season_for_type(item_type):
    """
    Picks a season with realistic weighting based on item type.
    
    Constraint #3: Season-Type Logic
    - Summer items (shorts, slings, short sleeves) more likely in summer
    - Winter items (long outwear, long sleeves) more likely in winter
    - Uses probability weights to make realistic seasonal choices
    
    Args:
        item_type (str): The type of clothing item
        
    Returns:
        str: A season appropriate for this type
    """
    seasons = list(SEASON)
    weights = []
    
    for season in seasons:
        # Get weight for this type in this season (default 1.0 if not specified)
        weight = SEASON_TYPE_WEIGHTS.get(season, {}).get(item_type, 1.0)
        weights.append(weight)
    
    # Weighted random choice
    return random.choices(seasons, weights=weights, k=1)[0]


def get_weighted_material_for_season(season):
    """
    Picks a material appropriate for the given season.
    
    Constraint #4: Material-Season Coherence
    - Summer: lightweight materials (cotton, linen, chiffon)
    - Winter: warm materials (wool, fleece, cashmere)
    - Spring/Autumn: transitional materials
    - Preferred materials have 6x higher probability
    
    Args:
        season (str): The season
        
    Returns:
        str: A material appropriate for this season
    """
    preferred = SEASON_PREFERRED_MATERIALS.get(season, MATERIAL)

    # Create weighted list: preferred materials 6x more likely
    materials = list(MATERIAL)
    weights = [6.0 if mat in preferred else 1.0 for mat in materials]
    
    return random.choices(materials, weights=weights, k=1)[0]


def get_weighted_pattern_for_style(style):
    """
    Picks a pattern that fits well with the given style.
    
    Constraint #5: Style-Pattern Coherence  
    - Formal/elegant: clean patterns (plain, striped, floral)
    - Sporty: geometric, graphic, plain
    - Bohemian: floral, tie-dye, embroidered
    - Preferred patterns have 6x higher probability
    
    Args:
        style (str): The style of the item
        
    Returns:
        str: A pattern that fits this style
    """
    preferred = STYLE_PREFERRED_PATTERNS.get(style, PATTERN)

    # Create weighted list: preferred patterns 6x more likely
    patterns = list(PATTERN)
    weights = [6.0 if pat in preferred else 1.0 for pat in patterns]
    
    return random.choices(patterns, weights=weights, k=1)[0]


def get_valid_occasion_for_type(item_type):
    """
    Picks an occasion appropriate for the given item type.
    
    Constraint #6: Occasion-Type Constraints
    - Beach: only suitable for shorts, dresses, light tops
    - Sport: no dresses or skirts
    - Formal/wedding: dresses, trousers, formal pieces
    - Filters out incompatible type-occasion combinations
    
    Args:
        item_type (str): The type of clothing item
        
    Returns:
        str: An occasion appropriate for this type
    """
    # Filter occasions to only those compatible with this type
    valid_occasions = []
    
    for occasion in OCCASION:
        # Check if this occasion has type constraints
        if occasion in OCCASION_TYPE_CONSTRAINTS:
            # Only add if type is in the allowed list
            if item_type in OCCASION_TYPE_CONSTRAINTS[occasion]:
                valid_occasions.append(occasion)
        else:
            # No constraints for this occasion, all types allowed
            valid_occasions.append(occasion)
    
    # If no valid occasions (shouldn't happen), return random
    if not valid_occasions:
        valid_occasions = list(OCCASION)
    
    return random.choice(valid_occasions)


def get_valid_insulation_for_season(season):
    """
    Picks insulation level appropriate for the season.
    
    Part of Constraint #4: Material-Season Coherence
    - Summer: none or light insulation only
    - Winter: medium to heavy insulation
    - Ensures outwear has realistic insulation for the season
    
    Args:
        season (str): The season
        
    Returns:
        str: An insulation level appropriate for this season
    """
    valid_options = SEASON_INSULATION_CONSTRAINTS.get(season, INSULATION)
    return random.choice(valid_options)


# ══════════════════════════════════════════════════════════════
# SEARCH & FILTER CONFIGURATION
# ══════════════════════════════════════════════════════════════
#
# FILTERABLE_FIELDS — single source of truth for what can be filtered.
#
# This list drives TWO things simultaneously:
#   1. description_generator.py  — every field listed here (plus all other
#      item keys except "id") is stored in the Qdrant payload metadata,
#      making it available for filtering at query time.
#   2. llm_query_parser.py       — the LLM is told exactly these fields and
#      their valid values, so it only returns filters Qdrant can act on.
#
# Rules for adding/removing fields:
#   - Global fields (present on every item) are always safe to add.
#   - Type-specific fields (e.g. neckline, closure) are fine in metadata
#     but keep the list short for the LLM — small models get confused by
#     very long prompts. Prefer user-facing ones a shopper would mention.
#   - Never add "brand" or "price" here — price is handled via price_range
#     and brand filtering is not yet wired to the vector search.
# ══════════════════════════════════════════════════════════════

FILTERABLE_FIELDS = [
    # ─── Global fields (every item has these) ───
    "type", "color", "style", "pattern", "material", "fit",
    "gender", "age_group", "season", "occasion", "body_type",
    # ─── Type-specific (most user-facing; skips technical ones
    "neckline", "collar", "sleeve_style", "hem_style",
    "closure", "hood", "insulation", "waterproof", "outwear_pockets",
    "waist", "waist_style", "rise", "length", "leg_style", "bottom_pockets",
    "dress_style"
]

# Free-text filter fields — the LLM extracts these as-is from the query
# (no closed set of valid values to list in the prompt / validate against).
# "brand" belongs here because there are 80+ brand names — injecting them
# all into the prompt would overwhelm a small model. The user just says
# "Nike" or "Gucci" and we trust the LLM to copy the name verbatim.
FREE_TEXT_FILTER_FIELDS = ["brand"]

# Fields excluded from the Qdrant payload metadata.
# Everything else present in the raw item JSON is stored as metadata.
METADATA_EXCLUDE_FIELDS = {"id"}

