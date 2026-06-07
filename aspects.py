# aspects.py  — keep this as a separate file, you'll import it everywhere

ASPECT_VOCAB = {
    "battery": ["battery", "charge", "charging", "life", "drain", "power"],
    "screen":  ["screen", "display", "brightness", "resolution", "pixels"],
    "camera":  ["camera", "photo", "picture", "image", "lens", "zoom"],
    "price":   ["price", "cost", "value", "cheap", "expensive", "worth"],
    "delivery":["delivery", "shipping", "packaging", "arrived", "box"]
}

# Build reverse lookup: word -> aspect
WORD_TO_ASPECT = {}
for aspect, keywords in ASPECT_VOCAB.items():
    for word in keywords:
        WORD_TO_ASPECT[word] = aspect