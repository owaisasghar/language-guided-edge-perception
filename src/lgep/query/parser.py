"""Rule-based parsing of natural-language object queries into a structured query.

"a man in a blue shirt and black jeans on cam02 in the last 10 minutes" ->
    StructuredQuery(categories=["person"], clothing=[Clothing("shirt", "upper", "blue"),
                    Clothing("jeans", "lower", "black")], cameras=["cam02"], time=TimeFilter(last_s=600), ...)

No language model and no network: it runs on the edge device, is deterministic and testable, and every
decision can be explained. What it does not understand stays in ``free_text`` and is still used by the
CLIP text-image similarity at retrieval time.

* **Category**: longest-match lookup of the 80 COCO class names plus synonyms and plurals ("guy",
  "pedestrian" -> person; "lorry" -> truck; "motorbike" -> motorcycle). Ambiguous words map to several
  categories ("vehicle" -> car / truck / bus / motorcycle; "bike" -> bicycle / motorcycle; "animal", "food").
  The *head* object is the first category mentioned before a relation word ("next to", "holding", "with",
  ...): in "man holding umbrella" the target is the man; in "large item next to car" it is unknown (not the
  car). In a noun-noun compound the last noun is the head ("baby elephant", "carrot bowl"). A query that
  names only clothing or body parts ("white shirt on the left", "red hair") refers to a person.
* **Colours** (11 basic colour terms + silver, with synonyms) attach to the next clothing item ("blue shirt"),
  to the head object ("white truck"), or, after "in" / "wearing" without an item, to the whole outfit.
* **Clothing** items map to a body region (upper / lower / full / head) for region-specific verification.
* **Cameras**: "cam02", "camera 3", "on cam 2". **Time**: "last 10 minutes", "past hour", "after 14:30",
  "before 9:15", "between 14:00 and 15:00".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COCO_CATEGORIES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush")
VEHICLES = ("car", "truck", "bus", "motorcycle")

# word or phrase -> categories (a tuple: some words are ambiguous, e.g. "vehicle")
_SYNONYMS: dict[str, tuple[str, ...]] = {c: (c,) for c in COCO_CATEGORIES}
for words, cats in {
    ("people", "man", "men", "woman", "women", "guy", "guys", "lady", "ladies", "girl", "boy", "kid", "kids", "child",
     "children", "pedestrian", "player", "baby", "dude", "gentleman", "adult", "worker", "officer", "persons", "human",
     "skier", "surfer", "batter", "catcher", "umpire", "rider", "cyclist", "biker"): ("person",),
    ("cycle",): ("bicycle",),
    ("bike", "bikes"): ("bicycle", "motorcycle"),  # "bike" is used for both
    ("sedan", "suv", "taxi", "cab", "van", "jeep", "hatchback", "minivan", "wagon", "station wagon", "estate car",
     "saloon", "crossover", "mpv", "coupe"): ("car",),
    ("motorbike", "scooter", "moped"): ("motorcycle",),
    ("lorry", "pickup", "pick up", "pickup truck", "semi"): ("truck",),
    ("coach", "minibus"): ("bus",),
    ("vehicle", "vehicles", "automobile"): VEHICLES,
    ("plane", "jet", "aircraft"): ("airplane",),
    ("ship",): ("boat",),
    ("sofa",): ("couch",),
    ("television", "monitor", "screen"): ("tv",),
    ("computer", "notebook"): ("laptop",),
    ("phone", "cellphone", "mobile", "smartphone"): ("cell phone",),
    ("table", "desk"): ("dining table",),
    ("plant", "flowerpot"): ("potted plant",),
    ("doughnut",): ("donut",),
    ("purse",): ("handbag",),
    ("bag", "luggage"): ("handbag", "backpack", "suitcase"),
    ("ball",): ("sports ball",),
    ("racket", "racquet"): ("tennis racket",),
    ("glove", "mitt"): ("baseball glove",),
    ("bat",): ("baseball bat",),
    ("hydrant",): ("fire hydrant",),
    ("fridge",): ("refrigerator",),
    ("puppy",): ("dog",), ("kitten", "kitty"): ("cat",), ("calf", "cattle"): ("cow",), ("lamb",): ("sheep",),
    ("teddy", "stuffed animal"): ("teddy bear",), ("glass",): ("wine glass", "cup"), ("mug",): ("cup",),
    ("bear", "bears"): ("bear", "teddy bear"),
    ("hotdog",): ("hot dog",),
    ("animal", "animals", "creature"): ("elephant", "giraffe", "zebra", "cow", "sheep", "horse", "dog", "cat", "bear", "bird"),
    ("food", "meal", "dish"): ("sandwich", "pizza", "cake", "donut", "hot dog", "broccoli", "carrot", "banana", "apple",
                               "orange", "bowl"),
    ("slice", "slices"): ("pizza", "cake", "sandwich"),
}.items():
    for w in words:
        _SYNONYMS[w] = cats

COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "silver")
_COLOUR_SYNONYMS = {"grey": "gray", "navy": "blue", "maroon": "red", "crimson": "red", "burgundy": "red",
                    "tan": "brown", "beige": "brown", "violet": "purple", "purp": "purple", "teal": "green",
                    "olive": "green", "golden": "yellow", "gold": "yellow", "blk": "black", "wht": "white"}
# Vehicle types (sub-types of the detector's car / truck classes), checked with CLIP like colours.
# canonical name -> (words in a query, the CLIP prompt noun)
VEHICLE_TYPES = {
    "sedan": (("sedan", "sedans", "saloon"), "sedan car"),
    "SUV": (("suv", "suvs", "jeep", "crossover"), "SUV"),
    "pickup": (("pickup truck", "pick-up truck", "pick up truck", "pickup", "pick-up", "pick up"), "pickup truck"),
    "minivan": (("minivan", "mini van", "mpv"), "minivan"),
    "van": (("van", "vans"), "van"),
    "hatchback": (("hatchback", "hatchbacks"), "hatchback car"),
    "wagon": (("station wagon", "wagon", "estate car"), "station wagon"),
}
# the alternatives a type is compared against (softmax): the types plus other vehicles
VEHICLE_TYPE_VOCAB = {**{k: v[1] for k, v in VEHICLE_TYPES.items()}, "bus": "bus", "truck": "box truck",
                      "motorcycle": "motorcycle"}
_TYPE_WORDS = sorted(((w, k) for k, (ws, _) in VEHICLE_TYPES.items() for w in ws), key=lambda x: -len(x[0]))
CLOTHING_REGIONS = {
    **dict.fromkeys(("shirt", "tshirt", "t-shirt", "tee", "jacket", "coat", "sweater", "hoodie", "top", "blouse", "vest",
                     "jersey", "sweatshirt", "cardigan", "uniform", "shirts", "jackets", "coats", "tank"), "upper"),
    **dict.fromkeys(("pants", "trousers", "jeans", "shorts", "skirt", "leggings", "slacks"), "lower"),
    **dict.fromkeys(("dress", "outfit", "suit", "clothes", "clothing", "robe", "gown", "overalls"), "full"),
    **dict.fromkeys(("hat", "cap", "helmet", "hood", "beanie", "scarf"), "head"),
}
_RELATIONS = ("next to", "close to", "in front of", "on top of", "behind", "beside", "near", "by", "under", "above",
              "below", "holding", "carrying", "with", "riding", "on", "against", "between", "touching", "in", "inside",
              "at", "from")
# Words that describe a person without naming one ("white shirt on the left", "red hair").
_PERSON_PARTS = {"hair", "head", "face", "hand", "hands", "arm", "arms", "leg", "legs", "back", "glasses", "sunglasses",
                 "beard", "bald", "blonde", "blond", "tie"}
_NOT_CATEGORIES = {"glasses", "sunglasses"}  # spectacles, not wine glasses
_SKIP = {"a", "an", "the", "of", "and", "is", "are", "that", "who", "which", "very", "all", "any", "find", "show", "me",
         "dark", "light", "bright", "pale", "large", "big", "small", "little", "tall", "short", "colored", "coloured"}


@dataclass(frozen=True)
class Clothing:
    item: str | None  # None: the colour of the outfit ("a woman in red")
    region: str  # upper | lower | full | head
    colour: str


@dataclass
class TimeFilter:
    last_s: float | None = None  # only the last N seconds before the query
    after: str | None = None  # "HH:MM" local clock time
    before: str | None = None


@dataclass
class StructuredQuery:
    text: str
    categories: list[str] = field(default_factory=list)  # empty: category unknown -> search all
    object_colours: list[str] = field(default_factory=list)
    clothing: list[Clothing] = field(default_factory=list)
    carried: list[str] = field(default_factory=list)  # categories in relations: "holding an umbrella" -> umbrella
    cameras: list[str] = field(default_factory=list)
    time: TimeFilter | None = None
    free_text: str = ""  # the description without camera / time filters (used for CLIP similarity)
    vehicle_types: list[str] = field(default_factory=list)  # "white SUV" -> ["SUV"] (VEHICLE_TYPES)

    @property
    def detector_prompt(self) -> str:
        """Phrases for an open-vocabulary detector: the head categories, or the description if none was found."""
        return " . ".join(self.categories) if self.categories else self.free_text


_CAMERA = re.compile(r"\b(?:on|in|from|at)?\s*(?:camera|cam)\s*[-_ ]?(?P<id>[a-z]*\d+[a-z0-9_-]*)\b")
_LAST = re.compile(r"\b(?:in\s+)?(?:the\s+)?(?:last|past)\s+(?P<n>\d+(?:\.\d+)?\s*)?(?P<unit>seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b")
_CLOCK = r"(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))"  # 14:30, 2:30pm, 2pm (not a bare "27")
_BETWEEN = re.compile(rf"\bbetween\s+{_CLOCK}\s+and\s+{_CLOCK}")
_AFTER = re.compile(rf"\b(?:after|since)\s+{_CLOCK}")
_BEFORE = re.compile(rf"\b(?:before|until)\s+{_CLOCK}")
_UNIT_S = {"s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60, "h": 3600, "hr": 3600, "hour": 3600}


def _clock(s: str) -> str:
    s = s.strip()
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    if not (0 <= h < 24 and 0 <= mi < 60):
        raise ValueError(f"invalid time {s!r}")
    return f"{h:02d}:{mi:02d}"


def _colour(word: str) -> str | None:
    w = _COLOUR_SYNONYMS.get(word, word)
    return w if w in COLOURS else None


def _singular(word: str) -> str:
    if word in _SYNONYMS or word in CLOTHING_REGIONS or word in _NOT_CATEGORIES or word in _PERSON_PARTS or len(word) < 4:
        return word
    for suffix, repl in (("ies", "y"), ("ses", "s"), ("es", "e"), ("s", "")):
        if word.endswith(suffix) and (word[: -len(suffix)] + repl) in _SYNONYMS:
            return word[: -len(suffix)] + repl
    return word


def parse_query(text: str) -> StructuredQuery:
    q = StructuredQuery(text=text)
    s = " " + text.lower().strip() + " "
    # camera and time filters (removed from the description)
    for m in _CAMERA.finditer(s):
        cid = m.group("id")
        q.cameras.append(cid if not cid.isdigit() else f"cam{int(cid):02d}")
    s = _CAMERA.sub(" ", s)
    tf = TimeFilter()
    if m := _LAST.search(s):
        unit = m.group("unit").rstrip("s") or "s"
        unit = {"sec": "sec", "min": "min", "hr": "hr"}.get(unit, unit)
        tf.last_s = float(m.group("n") or 1) * _UNIT_S.get(unit, _UNIT_S.get(unit[0], 1))
        s = _LAST.sub(" ", s)
    try:  # an invalid clock time ("25:00") is left in the text instead of failing the query
        if m := _BETWEEN.search(s):
            tf.after, tf.before = _clock(m.group(1)), _clock(m.group(2))
            s = _BETWEEN.sub(" ", s)
        else:
            if m := _AFTER.search(s):
                tf.after = _clock(m.group(1))
                s = _AFTER.sub(" ", s)
            if m := _BEFORE.search(s):
                tf.before = _clock(m.group(1))
                s = _BEFORE.sub(" ", s)
    except ValueError:
        tf.after = tf.before = None
    if tf != TimeFilter():
        q.time = tf
    tokens = [_singular(t) for t in re.findall(r"[a-z]+(?:-[a-z]+)?", s)]
    q.free_text = " ".join(t for t in re.findall(r"[a-z0-9]+(?:[-'][a-z]+)?", s))

    # positions of relation words: categories after the first one are not the head
    joined = " " + " ".join(tokens) + " "
    rel_positions = [len(joined[: joined.find(f" {r} ")].split()) for r in _RELATIONS if f" {r} " in joined]
    first_relation = min(rel_positions) if rel_positions else len(tokens)

    def category_at(j: int):
        if j >= len(tokens) or tokens[j] in _NOT_CATEGORIES:
            return None
        return next(((" ".join(tokens[j:j + n]), n) for n in (3, 2, 1)
                     if j + n <= len(tokens) and " ".join(tokens[j:j + n]) in _SYNONYMS), None)

    person_words = any(_SYNONYMS.get(t) == ("person",) for t in tokens)
    i, head_found, pending, in_or_wearing = 0, False, [], False
    while i < len(tokens):
        tok = tokens[i]
        colour = _colour(tok)
        match = category_at(i)
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if match and colour and (nxt in CLOTHING_REGIONS or category_at(i + 1)):  # "orange shirt", not the fruit
            match = None
        if match and match[0] == "tie" and person_words:  # "man in a red tie": clothing, not the target
            match = None
            tok = "tie"
        if match:  # an object category (longest match first: "pickup truck", "traffic light")
            phrase, n = match
            if not head_found and i < first_relation and (follow := category_at(i + n)) and i + n < first_relation:
                pending, i = [], i + n  # noun-noun compound ("baby elephant", "carrot bowl"): the last noun is the head
                continue
            cats = _SYNONYMS[phrase]
            if not head_found and i < first_relation:
                q.categories, head_found = list(cats), True
                q.object_colours += pending
            else:  # mentioned in a relation ("holding an umbrella", "next to a car")
                q.carried += [c for c in cats if c not in q.carried and c not in q.categories]
            pending = []
            i += n
            continue
        if tok in CLOTHING_REGIONS or tok == "tie":
            q.clothing += [Clothing(tok, CLOTHING_REGIONS.get(tok, "upper"), c) for c in pending]
            pending, in_or_wearing = [], False
        elif colour is not None:
            pending.append(colour)
        elif tok in ("in", "wearing", "dressed"):
            in_or_wearing = True
        i += 1
    if not head_found and (q.clothing or any(t in CLOTHING_REGIONS or t in _PERSON_PARTS for t in tokens)):
        q.categories, head_found = ["person"], True  # clothing or body parts only: a person
    if pending:  # colours not followed by an item or an object: "a woman in red", "the car that is white"
        if in_or_wearing or q.categories == ["person"]:
            q.clothing += [Clothing(None, "full", c) for c in pending]
        elif head_found:
            q.object_colours += pending
    q.object_colours = list(dict.fromkeys(q.object_colours))
    # Vehicle types of the head object only: "a red sedan followed by a green van" is about the sedan.
    head = " " + q.free_text + " "
    for rel in _RELATIONS:
        i = head.find(f" {rel} ", 1)
        if i > 0 and any(f" {w} " in head[:i + 1] for w, _ in _TYPE_WORDS):
            head = head[:i + 1]
    rest = head
    for word, kind in _TYPE_WORDS:  # longest first: "pickup truck" before "pickup", "minivan" before "van"
        if f" {word} " in rest:
            q.vehicle_types.append(kind)
            # a colour right before the type word is the vehicle's colour ("silver station wagon"), if the
            # head had none ("a green wagon")
            before = rest.split(f" {word} ", 1)[0].split()
            colour = _colour(before[-1]) if before else None
            if colour and not q.object_colours and all(c.colour != colour for c in q.clothing):
                q.object_colours.append(colour)
            rest = rest.replace(f" {word} ", " ")
    q.vehicle_types = list(dict.fromkeys(q.vehicle_types))
    return q
