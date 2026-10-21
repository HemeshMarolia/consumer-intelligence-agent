"""Step 1: create a synthetic skincare dataset (products, sales, reviews).

The data is fake but has planted patterns you can check your app against:
  - Vitamin C Serum: many packaging (pump leaks) complaints, sales fall from April.
  - SPF 50 Sunscreen: many texture complaints (sticky, white cast).
  - Tinted Sunscreen: sales grow month over month.
"""
import random

import pandas as pd

from config import DATA_DIR

random.seed(42)
DATA_DIR.mkdir(exist_ok=True)

PRODUCTS = [
    (1, "GlowLab Vitamin C Serum", "Serum", 599),
    (2, "GlowLab Niacinamide Serum", "Serum", 549),
    (3, "PureLeaf Rice Face Wash", "Face Wash", 249),
    (4, "PureLeaf Tea Tree Face Wash", "Face Wash", 229),
    (5, "SunShield SPF 50 Sunscreen", "Sunscreen", 449),
    (6, "SunShield Tinted Sunscreen", "Sunscreen", 499),
    (7, "HydraBloom Moisturizer", "Moisturizer", 399),
    (8, "HydraBloom Night Cream", "Moisturizer", 499),
    (9, "ClearSkin Salicylic Gel", "Treatment", 349),
    (10, "RoseDew Face Toner", "Toner", 299),
]
CHANNELS = ["E-commerce", "Quick Commerce", "Modern Trade", "General Trade"]
REGIONS = ["North", "South", "East", "West"]
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]

POSITIVE = [
    "Love this, my skin feels great after a couple of weeks.",
    "Really good product, I can see a visible difference.",
    "Gentle on my skin and works well. Will repurchase.",
    "Absorbs quickly and my skin looks fresh and glowing.",
    "Worth the money, it does exactly what it promises.",
    "Great texture and a pleasant feel. Highly recommend.",
]
NEUTRAL = [
    "It is okay, nothing special but does the job.",
    "Decent product, results are average so far.",
    "Not bad, though I expected a bit more for the price.",
]
NEGATIVE = {
    "packaging": [
        "The pump leaked inside the box and half the product was wasted.",
        "Packaging is poor, the bottle cap broke within a week.",
        "Arrived with the seal open and product spilled everywhere.",
    ],
    "fragrance": [
        "The fragrance is way too strong and gave me a headache.",
        "Smells artificial and the scent lingers for hours.",
    ],
    "texture": [
        "Feels very sticky and leaves a white cast on my skin.",
        "Texture is greasy and heavy, makes my face feel oily.",
        "Pills under makeup and feels tacky even after ten minutes.",
    ],
    "breakouts_irritation": [
        "Caused breakouts on my cheeks within three days.",
        "My skin turned red and started itching after using it.",
    ],
    "dryness": [
        "Left my face feeling tight and dry after every wash.",
        "Too harsh, my skin got flaky and dry.",
    ],
    "price": [
        "Way too expensive for the quantity you get.",
        "Not worth the price, cheaper options work better.",
    ],
    "delivery": [
        "Delivery was delayed by two weeks and support did not respond.",
        "Received a damaged parcel and the replacement took ages.",
    ],
    "ineffective": [
        "Used it for a month and saw zero improvement.",
        "Did nothing for my skin, a waste of money.",
    ],
}
THEMES = list(NEGATIVE)

# Probability that a review is negative, per product.
NEG_RATE = {1: 0.35, 5: 0.35, 3: 0.20, 4: 0.20}
DEFAULT_NEG_RATE = 0.12

# Which complaint themes dominate for each product (weights).
THEME_WEIGHTS = {
    1: {"packaging": 6, "delivery": 2, "price": 2, "ineffective": 1, "breakouts_irritation": 1},
    5: {"texture": 7, "fragrance": 1, "price": 1, "ineffective": 2},
    3: {"dryness": 5, "fragrance": 3, "price": 1},
    4: {"dryness": 5, "fragrance": 2, "breakouts_irritation": 2},
}


def pick_negative_theme(pid: int) -> str:
    weights = THEME_WEIGHTS.get(pid, {t: 1 for t in THEMES})
    return random.choices(list(weights), weights=list(weights.values()))[0]


def make_reviews(n: int = 600) -> pd.DataFrame:
    rows = []
    for rid in range(1, n + 1):
        pid, name, _, _ = random.choice(PRODUCTS)
        month = random.choice(MONTHS)
        day = random.randint(1, 28)
        if random.random() < NEG_RATE.get(pid, DEFAULT_NEG_RATE):
            rating = random.choice([1, 1, 2, 2, 2])
            text = random.choice(NEGATIVE[pick_negative_theme(pid)])
            if random.random() < 0.3:  # sometimes a second complaint
                text += " " + random.choice(NEGATIVE[pick_negative_theme(pid)])
        elif random.random() < 0.15:
            rating, text = 3, random.choice(NEUTRAL)
        else:
            rating, text = random.choice([4, 5, 5]), random.choice(POSITIVE)
        rows.append(
            {
                "review_id": rid,
                "product_id": pid,
                "rating": rating,
                "review_date": f"{month}-{day:02d}",
                "review_text": text,
            }
        )
    return pd.DataFrame(rows)


def make_sales() -> pd.DataFrame:
    rows, sid = [], 1
    for month_idx, month in enumerate(MONTHS):
        for pid, _, _, price in PRODUCTS:
            for channel in CHANNELS:
                for region in REGIONS:
                    base = random.randint(60, 220)
                    if pid == 1 and month_idx >= 3:  # leak complaints hurt sales
                        base = int(base * 0.6)
                    if pid == 6:  # steady growth
                        base = int(base * (1 + 0.15 * month_idx))
                    if channel == "Quick Commerce":
                        base = int(base * 1.2)
                    units = max(base, 1)
                    discount = random.choice([0, 0, 0.05, 0.10, 0.15])
                    rows.append(
                        {
                            "sale_id": sid,
                            "sale_month": month,
                            "product_id": pid,
                            "channel": channel,
                            "region": region,
                            "units": units,
                            "revenue": round(units * price * (1 - discount), 2),
                        }
                    )
                    sid += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    products = pd.DataFrame(PRODUCTS, columns=["product_id", "product_name", "category", "price"])
    products.to_csv(DATA_DIR / "products.csv", index=False)
    make_sales().to_csv(DATA_DIR / "sales.csv", index=False)
    make_reviews().to_csv(DATA_DIR / "reviews.csv", index=False)
    print("Created products.csv, sales.csv and reviews.csv in", DATA_DIR)
