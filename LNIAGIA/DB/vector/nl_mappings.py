# ════════════════════════════════════════════════════════════
# nl_mappings.py
# Natural Language mappings for clothing attributes
# Used by ClothingDescriptionGenerator to enrich embeddings
# with synonyms and human-friendly descriptions.
# ════════════════════════════════════════════════════════════


# ═══ TYPE MAPPINGS (code → natural language) ═══
TYPE_NAMES = {
    "short_sleeve_top": "short sleeve top (t-shirt, tee)",
    "long_sleeve_top": "long sleeve top (shirt, sweater, long-sleeved shirt)",
    "long_sleeve_outwear": "long sleeve jacket (coat, hoodie, outerwear)",
    "vest": "sleeveless vest (tank top, sleeveless top)",
    "shorts": "shorts (short pants)",
    "trousers": "trousers (pants, jeans, slacks)",
    "skirt": "skirt",
    "short_sleeve_dress": "short sleeve dress",
    "long_sleeve_dress": "long sleeve dress",
    "vest_dress": "sleeveless dress (vest dress)",
    "sling_dress": "spaghetti strap dress (slip dress, strappy dress)",
}

# ═══ COLOR DESCRIPTIONS ═══
COLOR_DESCRIPTIONS = {
    "black": "black (dark, noir)",
    "white": "white (bright, clean)",
    "gray": "gray (grey, neutral)",
    "navy": "navy blue (dark blue, maritime blue)",
    "blue": "blue",
    "red": "red (crimson, scarlet)",
    "green": "green",
    "yellow": "yellow (golden, sunny)",
    "orange": "orange (tangerine)",
    "pink": "pink (rose, blush)",
    "purple": "purple (violet, plum)",
    "brown": "brown (chocolate, tan)",
    "beige": "beige (cream, sand, off-white)",
    "cream": "cream (ivory, off-white)",
    "burgundy": "burgundy (wine red, maroon, dark red)",
    "olive": "olive (olive green, military green)",
    "teal": "teal (blue-green, turquoise)",
    "coral": "coral (salmon, peachy pink)",
    "multicolor": "multicolor (colorful, multi-colored, patterned)",
}

# ═══ STYLE DESCRIPTIONS ═══
STYLE_DESCRIPTIONS = {
    "casual": "casual (everyday, relaxed, informal)",
    "formal": "formal (dressy, elegant, professional)",
    "smart casual": "smart casual (business casual, polished casual)",
    "sporty": "sporty (athletic, active, sport-inspired)",
    "bohemian": "bohemian (boho, free-spirited, artistic)",
    "minimalist": "minimalist (simple, clean, understated)",
    "streetwear": "streetwear (urban, hip-hop inspired, trendy)",
    "vintage": "vintage (retro, classic, old-school)",
    "elegant": "elegant (sophisticated, refined, chic)",
    "preppy": "preppy (collegiate, classic American, polished)",
}

# ═══ PATTERN DESCRIPTIONS ═══
PATTERN_DESCRIPTIONS = {
    "plain": "plain (solid color, no pattern, simple)",
    "striped": "striped (stripes, lined)",
    "checkered": "checkered (checked, gingham)",
    "plaid": "plaid (tartan, checked pattern)",
    "floral": "floral (flower pattern, botanical print)",
    "polka dot": "polka dot (dotted, spotted)",
    "geometric": "geometric (shapes, abstract geometric)",
    "abstract": "abstract (artistic pattern, modern print)",
    "animal print": "animal print (leopard, zebra, snake print)",
    "camouflage": "camouflage (camo, military pattern)",
    "tie-dye": "tie-dye (psychedelic, rainbow swirl)",
    "graphic": "graphic (printed design, logo, artwork)",
    "embroidered": "embroidered (stitched design, decorative stitching)",
}

# ═══ MATERIAL DESCRIPTIONS ═══
MATERIAL_DESCRIPTIONS = {
    "cotton": "cotton (soft, breathable, natural fiber)",
    "polyester": "polyester (durable, synthetic, easy-care)",
    "linen": "linen (lightweight, breathable, natural)",
    "silk": "silk (luxurious, smooth, elegant)",
    "wool": "wool (warm, cozy, natural fiber)",
    "denim": "denim (jeans material, sturdy cotton)",
    "leather": "leather (genuine leather, premium)",
    "suede": "suede (soft leather, velvety texture)",
    "velvet": "velvet (plush, luxurious, soft)",
    "satin": "satin (smooth, shiny, silky)",
    "chiffon": "chiffon (sheer, lightweight, flowy)",
    "fleece": "fleece (soft, warm, cozy)",
    "cashmere": "cashmere (luxury wool, ultra-soft)",
    "nylon": "nylon (synthetic, durable, lightweight)",
    "rayon": "rayon (semi-synthetic, silky feel)",
    "spandex": "spandex (stretchy, elastic, flexible)",
    "organic cotton": "organic cotton (eco-friendly, sustainable, natural)",
}

# ═══ FIT DESCRIPTIONS ═══
FIT_DESCRIPTIONS = {
    "slim fit": "slim fit (fitted, body-hugging, narrow cut)",
    "regular": "regular fit (standard, normal fit)",
    "relaxed": "relaxed fit (comfortable, easy fit)",
    "oversized": "oversized (big, roomy, loose, baggy)",
    "tailored": "tailored fit (structured, precise fit)",
    "loose": "loose fit (flowy, relaxed, roomy)",
    "fitted": "fitted (form-fitting, close to body)",
    "athletic": "athletic fit (sporty, performance fit)",
    "baggy": "baggy (very loose, oversized, roomy)",
    "cropped": "cropped (shortened, above natural length)",
}

# ═══ SEASON DESCRIPTIONS ═══
SEASON_DESCRIPTIONS = {
    "spring": "spring (mild weather, transitional)",
    "summer": "summer (hot weather, warm season)",
    "autumn": "autumn (fall, cool weather)",
    "winter": "winter (cold weather, warm clothing)",
    "all-season": "all-season (year-round, versatile)",
}

# ═══ OCCASION DESCRIPTIONS ═══
OCCASION_DESCRIPTIONS = {
    "everyday": "everyday wear (daily, casual occasions)",
    "work": "work (office, professional, business)",
    "party": "party (celebration, night out, festive)",
    "wedding": "wedding (formal event, ceremony)",
    "beach": "beach (vacation, resort, seaside)",
    "sport": "sport (exercise, gym, athletic activities)",
    "date night": "date night (romantic, evening out)",
    "travel": "travel (comfortable, versatile, journey)",
    "lounge": "lounge (home, relaxation, comfort)",
    "formal event": "formal event (gala, ceremony, special occasion)",
}

# ═══ GENDER DESCRIPTIONS ═══
GENDER_DESCRIPTIONS = {
    "male": "men's (male, masculine)",
    "female": "women's (female, feminine)",
    "unisex": "unisex (gender-neutral, for everyone)",
}

# ═══ AGE GROUP DESCRIPTIONS ═══
AGE_GROUP_DESCRIPTIONS = {
    "baby": "baby (infant, newborn, 0-2 years)",
    "child": "child (kids, children, 3-12 years)",
    "teenager": "teenager (teen, adolescent, 13-17 years)",
    "young adult": "young adult (young, 18-29 years)",
    "adult": "adult (grown-up, 30-59 years)",
    "senior": "senior (elderly, mature, 60+ years)",
}


# ══════════════════════════════════════════════════════════════
# TYPE-SPECIFIC FIELD MAPPINGS
# (derived from the values defined in models.py)
# ══════════════════════════════════════════════════════════════

# ═══ NECKLINE DESCRIPTIONS (Tops & Dresses) ═══
NECKLINE_DESCRIPTIONS = {
    "crew neck": "crew neck (round neck, classic neckline)",
    "v-neck": "V-neck (V-shaped neckline)",
    "scoop neck": "scoop neck (wide, low round neckline)",
    "boat neck": "boat neck (bateau, wide horizontal neckline)",
    "turtleneck": "turtleneck (high neck, roll neck, polo neck)",
    "mock neck": "mock neck (short turtleneck, stand-up collar)",
    "off-shoulder": "off-shoulder (bare shoulders, strapless top)",
    "square neck": "square neck (angular, straight-across neckline)",
    "halter": "halter neck (halter top, tied behind neck)",
    "sweetheart": "sweetheart neckline (heart-shaped, romantic neckline)",
    "cowl neck": "cowl neck (draped, loose-fold neckline)",
    "collared": "collared neckline (shirt collar, classic collar)",
}

# ═══ COLLAR DESCRIPTIONS (Tops & Outwear) ═══
COLLAR_DESCRIPTIONS = {
    "none": "no collar",
    "pointed": "pointed collar (classic shirt collar)",
    "spread": "spread collar (wide collar, cutaway)",
    "button-down": "button-down collar (casual, preppy collar)",
    "mandarin": "mandarin collar (band collar, nehru collar, stand-up)",
    "cuban": "cuban collar (camp collar, open revere collar)",
    "peter pan": "peter pan collar (round, flat collar)",
    "shawl": "shawl collar (rolled, wrap-around collar)",
    "notched": "notched collar (lapel collar, blazer collar)",
}

# ═══ SLEEVE STYLE DESCRIPTIONS (Tops, Outwear & Dresses) ═══
SLEEVE_STYLE_DESCRIPTIONS = {
    "regular": "regular sleeves (standard, classic sleeves)",
    "puff": "puff sleeves (puffy, voluminous, balloon sleeves)",
    "bell": "bell sleeves (wide, flared sleeves)",
    "raglan": "raglan sleeves (baseball-style, diagonal seam sleeves)",
    "cap": "cap sleeves (short, minimal coverage sleeves)",
    "dolman": "dolman sleeves (batwing, wide loose sleeves)",
    "bishop": "bishop sleeves (full, gathered-at-cuff sleeves)",
    "lantern": "lantern sleeves (rounded, voluminous mid-arm sleeves)",
    "rolled": "rolled sleeves (folded-up, casual cuff)",
    "cuffed": "cuffed sleeves (turned-up cuff, banded sleeves)",
}

# ═══ HEM STYLE DESCRIPTIONS (Tops & Dresses) ═══
HEM_STYLE_DESCRIPTIONS = {
    "straight": "straight hem (even, clean-cut hem)",
    "curved": "curved hem (rounded, scoop hem)",
    "cropped": "cropped hem (short, above-waist hem)",
    "asymmetric": "asymmetric hem (uneven, high-low hem)",
    "raw edge": "raw edge hem (unfinished, frayed hem)",
    "knotted": "knotted hem (tied-up, front-knot hem)",
    "split": "split hem (side-slit, vented hem)",
    "longline": "longline hem (extended, long-cut hem)",
}

# ═══ CLOSURE DESCRIPTIONS (Outwear) ═══
CLOSURE_DESCRIPTIONS = {
    "zipper": "zipper closure (zip-up, zip front)",
    "buttons": "button closure (button-up, button front)",
    "snap buttons": "snap button closure (press-stud, popper)",
    "velcro": "velcro closure (hook-and-loop fastener)",
    "toggle": "toggle closure (toggle buttons, duffle-coat closure)",
    "belt": "belt closure (belted, tie-waist)",
    "open front": "open front (no closure, cardigan-style)",
    "double breasted": "double breasted closure (overlapping front, formal)",
}

# ═══ HOOD DESCRIPTIONS (Outwear) ═══
HOOD_DESCRIPTIONS = {
    "none": "no hood",
    "attached": "attached hood (fixed hood, built-in hood)",
    "detachable": "detachable hood (removable hood, zip-off hood)",
    "faux fur trim": "faux fur trim hood (fur-lined hood, plush hood)",
}

# ═══ INSULATION DESCRIPTIONS (Outwear) ═══
INSULATION_DESCRIPTIONS = {
    "none": "no insulation (unlined, shell only)",
    "light": "light insulation (lightly lined, thin padding)",
    "medium": "medium insulation (moderately warm, padded)",
    "heavy": "heavy insulation (thick, very warm, heavily padded)",
    "down": "down insulation (down-filled, goose down, ultra-warm)",
    "synthetic": "synthetic insulation (synthetic fill, polyester fill)",
}

# ═══ WATERPROOF DESCRIPTIONS (Outwear) ═══
WATERPROOF_DESCRIPTIONS = {
    "none": "not waterproof",
    "water resistant": "water resistant (light rain protection, splash-proof)",
    "waterproof": "waterproof (fully waterproof, rain-proof, sealed)",
}

# ═══ OUTWEAR POCKET DESCRIPTIONS ═══
OUTWEAR_POCKET_DESCRIPTIONS = {
    "none": "no pockets",
    "side pockets": "side pockets (hip pockets, hand pockets)",
    "chest pocket": "chest pocket (breast pocket, upper pocket)",
    "interior pockets": "interior pockets (inside pockets, inner pockets)",
    "zippered pockets": "zippered pockets (zip pockets, secure pockets)",
    "flap pockets": "flap pockets (covered pockets, buttoned flap)",
    "patch pockets": "patch pockets (sewn-on pockets, external pockets)",
    "hand warmer pockets": "hand warmer pockets (fleece-lined, warm pockets)",
    "multiple pockets": "multiple pockets (many pockets, utility pockets)",
}

# ═══ WAIST SIZE DESCRIPTIONS (Bottoms) ═══
WAIST_SIZE_DESCRIPTIONS = {
    "xs": "extra small (XS)",
    "small": "small (S)",
    "medium": "medium (M)",
    "large": "large (L)",
    "xl": "extra large (XL)",
    "xxl": "double extra large (XXL, 2XL)",
}

# ═══ WAIST STYLE DESCRIPTIONS (Bottoms & Dresses) ═══
WAIST_STYLE_DESCRIPTIONS = {
    "regular": "regular waist (standard waist, natural waist)",
    "high-waisted": "high-waisted (high waist, above-navel)",
    "mid-rise": "mid-rise waist (sits at hip, medium waist)",
    "low-rise": "low-rise waist (low waist, hip-hugger)",
    "elastic": "elastic waist (stretch waistband, pull-on)",
    "drawstring": "drawstring waist (adjustable, tie waist)",
    "belted": "belted waist (with belt, cinched waist)",
    "paper bag": "paper bag waist (gathered, pleated high waist)",
}

# ═══ RISE DESCRIPTIONS (Bottoms) ═══
RISE_DESCRIPTIONS = {
    "low-rise": "low-rise (sits below hips, low waist)",
    "mid-rise": "mid-rise (sits at hips, standard rise)",
    "high-rise": "high-rise (sits above waist, high waist)",
}

# ═══ LENGTH DESCRIPTIONS (Bottoms & Dresses) ═══
LENGTH_DESCRIPTIONS = {
    "mini": "mini length (very short, above mid-thigh)",
    "short": "short length (above knee)",
    "knee-length": "knee-length (at the knee)",
    "midi": "midi length (below knee, mid-calf)",
    "ankle": "ankle length (ankle-grazing, just above ankle)",
    "full length": "full length (floor-length, to the floor)",
    "maxi": "maxi length (long, floor-sweeping)",
    "cropped": "cropped length (above ankle, shortened)",
}

# ═══ LEG STYLE DESCRIPTIONS (Bottoms) ═══
LEG_STYLE_DESCRIPTIONS = {
    "straight": "straight leg (even width, classic cut)",
    "skinny": "skinny leg (tight, skin-tight, narrow)",
    "slim": "slim leg (narrow, tapered, close-fitting)",
    "bootcut": "bootcut leg (slightly flared from knee, boot-friendly)",
    "wide leg": "wide leg (palazzo, flowing, roomy leg)",
    "flared": "flared leg (wide from knee, bell-bottom)",
    "tapered": "tapered leg (narrows toward ankle, carrot fit)",
    "jogger": "jogger style (elastic cuff, sporty, ribbed ankle)",
    "cargo": "cargo leg (utility, multi-pocket, military-inspired)",
}

# ═══ BOTTOM POCKET DESCRIPTIONS ═══
BOTTOM_POCKET_DESCRIPTIONS = {
    "none": "no pockets",
    "side pockets": "side pockets (hip pockets, trouser pockets)",
    "back pockets": "back pockets (rear pockets, seat pockets)",
    "cargo pockets": "cargo pockets (thigh pockets, utility pockets)",
    "zippered pockets": "zippered pockets (zip pockets, secure pockets)",
    "hidden pockets": "hidden pockets (concealed, invisible pockets)",
    "coin pocket": "coin pocket (small pocket, watch pocket)",
    "welt pockets": "welt pockets (slit pockets, piped pockets)",
    "patch pockets": "patch pockets (sewn-on, external pockets)",
}

# ═══ DRESS STYLE DESCRIPTIONS ═══
DRESS_STYLE_DESCRIPTIONS = {
    "a-line": "A-line dress (flared from waist, triangular silhouette)",
    "bodycon": "bodycon dress (body-conscious, tight-fitting, figure-hugging)",
    "shift": "shift dress (straight, loose, boxy silhouette)",
    "wrap": "wrap dress (crossover, tie-waist, V-shape front)",
    "maxi": "maxi dress (long, floor-length, flowing)",
    "midi": "midi dress (mid-length, below-knee)",
    "mini": "mini dress (short, above-knee)",
    "shirt dress": "shirt dress (button-front, collared, casual dress)",
    "slip dress": "slip dress (lingerie-inspired, slinky, minimal)",
    "fit and flare": "fit and flare dress (fitted bodice, flared skirt)",
    "empire": "empire dress (high-waist seam, just below bust)",
    "sheath": "sheath dress (tailored, column silhouette, fitted)",
}

# ═══ BODY TYPE DESCRIPTIONS ═══
# Body shape a garment is designed to flatter. Keys MUST match
# DB.models.BODY_TYPE exactly (used for filtering and validation).
BODY_TYPE_DESCRIPTIONS = {
    "hourglass": "hourglass shape (balanced bust and hips, defined waist, curvy)",
    "pear": "pear shape (hips wider than shoulders, triangle bottom-heavy)",
    "triangle": "triangle shape (narrow shoulders, wider hips, bottom-heavy)",
    "rectangle": "rectangle shape (straight, balanced shoulders and hips, little waist definition)",
    "inverted_triangle": "inverted triangle shape (broad shoulders, narrow hips, top-heavy, athletic V-shape)",
    "apple": "apple shape (fuller midsection, weight around waist, round)",
    "trapezoid": "trapezoid shape (broad shoulders tapering to narrower waist, fit athletic male build)",
    "oval": "oval shape (fuller midsection, rounded torso, broader middle)",
}


# ══════════════════════════════════════════════════════════════
# CONVENIENCE: single dict mapping field name → its lookup table
# ══════════════════════════════════════════════════════════════

ALL_MAPPINGS = {
    # Global fields
    "type": TYPE_NAMES,
    "color": COLOR_DESCRIPTIONS,
    "style": STYLE_DESCRIPTIONS,
    "pattern": PATTERN_DESCRIPTIONS,
    "material": MATERIAL_DESCRIPTIONS,
    "fit": FIT_DESCRIPTIONS,
    "season": SEASON_DESCRIPTIONS,
    "occasion": OCCASION_DESCRIPTIONS,
    "gender": GENDER_DESCRIPTIONS,
    "age_group": AGE_GROUP_DESCRIPTIONS,
    "body_type": BODY_TYPE_DESCRIPTIONS,
    # Type-specific fields
    "neckline": NECKLINE_DESCRIPTIONS,
    "collar": COLLAR_DESCRIPTIONS,
    "sleeve_style": SLEEVE_STYLE_DESCRIPTIONS,
    "hem_style": HEM_STYLE_DESCRIPTIONS,
    "closure": CLOSURE_DESCRIPTIONS,
    "hood": HOOD_DESCRIPTIONS,
    "insulation": INSULATION_DESCRIPTIONS,
    "waterproof": WATERPROOF_DESCRIPTIONS,
    "outwear_pockets": OUTWEAR_POCKET_DESCRIPTIONS,
    "waist": WAIST_SIZE_DESCRIPTIONS,
    "waist_style": WAIST_STYLE_DESCRIPTIONS,
    "rise": RISE_DESCRIPTIONS,
    "length": LENGTH_DESCRIPTIONS,
    "leg_style": LEG_STYLE_DESCRIPTIONS,
    "bottom_pockets": BOTTOM_POCKET_DESCRIPTIONS,
    "dress_style": DRESS_STYLE_DESCRIPTIONS,
}
