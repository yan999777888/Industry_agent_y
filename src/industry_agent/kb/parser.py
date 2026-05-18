"""Manual parsing and text normalization."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from industry_agent.kb.models import ManualDocument

PIC_TOKEN = "<PIC>"
PIC_RE = re.compile(r"<PIC>", flags=re.IGNORECASE)
UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+\*?")
TAIL_IMAGE_LIST_RE = re.compile(
    r',\s*(\[(?:"[^"]+"\s*,\s*)*"[^"]*"\s*\])\s*\]\s*$',
    flags=re.DOTALL,
)
OCR_WORD_FIXES: tuple[tuple[str, str], ...] = (
    ("Donot", "Do not"),
    ("donot", "do not"),
    ("Itis", "It is"),
    ("itis", "it is"),
    ("isnot", "is not"),
    ("arenot", "are not"),
    ("isnotrecommended", "is not recommended"),
    ("canaccidentally", "can accidentally"),
    ("accidentallychoke", "accidentally choke"),
    ("chokethechild", "choke the child"),
    ("chokethechildorgiveanelectricalshock", "choke the child or give an electrical shock"),
    ("giveanelectricalshock", "give an electrical shock"),
    ("electricalshock", "electrical shock"),
    ("electricalshock.", "electrical shock."),
    ("poweroutlet", "power outlet"),
    ("thepoweroutlet", "the power outlet"),
    ("thepower", "the power"),
    ("theback-up", "the back-up"),
    ("thecamera", "the camera"),
    ("arebased", "are based"),
    ("Usenon", "Use non"),
    ("tocleanany", "to clean any"),
    ("sup ply", "supply"),
    ("beforesetting", "before setting"),
    ("anyhome-made", "any home-made"),
    ("orback-upbattery", "or back-up battery"),
    ("thebattery", "the battery"),
    ("fireorwater", "fire or water"),
    ("orwater", "or water"),
    ("fireor", "fire or "),
    ("preventafire", "prevent a fire"),
    ("togetherwith", "together with"),
    ("Beforeusing", "Before using"),
    ("beforeusing", "before using"),
    ("Connectthepowercord", "Connect the power cord"),
    ("thepowercord", "the power cord"),
    ("intoplace", "into place"),
    ("maybecome", "may become"),
    ("physicianimmediately", "physician immediately"),
    ("insulationandcauseafire", "insulation and cause a fire"),
    ("oraccessorywhennot", "or accessory when not"),
    ("Electromagneticwaves", "Electromagnetic waves"),
    ("nottwistortiethecords", "not twist or tie the cords"),
    ("thesurroundingisdusty", "the surrounding is dusty"),
    ("darkroomorchemical", "darkroom or chemical"),
    ("technologywithover", "technology with over"),
    ("activepixels", "active pixels"),
    ("deadpixels", "dead pixels"),
    ("amongtheremaining", "among the remaining"),
    ("canwarpthecardsandmakethemunusable", "can warp the cards and make them unusable"),
    ("alsocompatiblewith", "also compatible with"),
    ("turnsoffautomaticaly", "turns off automatically"),
    ("turnsoffautomatically", "turns off automatically"),
    ("automaticallyafterasettimeofidleoperation", "automatically after a set time of idle operation"),
    ("justpresstheshutterbuttonhalfwaytoturnitonagain", "just press the shutter button halfway to turn it on again"),
    ("lithiumbatteryasdescribedbelow", "lithium battery as described below"),
    ("closeanditmaydamagetheshuttercurtainsandimagesensor", "close and it may damage the shutter curtains and image sensor"),
    ("aresetautomaticallytosuit", "are set automatically to suit"),
    ("besettoobtainthebestresults", "be set to obtain the best results"),
    ("themeteringmodewillbesetto", "the metering mode will be set to"),
    ("themeteringmodewillbeset", "the metering mode will be set"),
    ("theAFmodewillbeset", "the AF mode will be set"),
    ("modessimultaneously", "modes simultaneously"),
    ("JPEGformatscannotbeselected", "JPEG formats cannot be selected"),
    ("processingparameters", "processing parameters"),
    ("numberofpossibleshotswillbehigher", "number of possible shots will be higher"),
    ("temperaturereading", "temperature reading"),
    ("willshowthebiasdirectionand", "will show the bias direction and"),
    ("onthewhitebalancemode", "on the white balance mode"),
    ("whitebalancebracketing", "white balance bracketing"),
    ("downtheshutterbutton", "down the shutter button"),
    ("effectiveforbacklitsubjects", "effective for backlit subjects"),
    ("properlybytheflash", "properly by the flash"),
    ("cameraisclosertothesubject", "camera is closer to the subject"),
    ("autoflashwithmultiple", "auto flash with multiple"),
    ("Sinceconnectioncords", "Since connection cords"),
    ("synccordregardlessofitspolarity", "sync cord regardless of its polarity"),
    ("thecapacityindicatedon", "the capacity indicated on"),
    ("automaticallyafteryouresolvetheproblem", "automatically after you resolve the problem"),
    ("imagewasrecomposed", "image was recomposed"),
    ("individuallyforeachimage", "individually for each image"),
    ("yourselfdependingon", "yourself depending on"),
    ("Thischapterexplainshowtousethe", "This chapter explains how to use the"),
    ("OThelensfocusmodeswitch", "The lens focus mode switch"),
    ("ThisaccommodatestwoBP", "This accommodates two BP"),
    ("TROUBLEHSOOTINGvacuum", "TROUBLESHOOTING vacuum"),
    ("inwaterorifwaterormetalfragmentsenterinsidethe", "in water or if water or metal fragments enter inside the"),
    ("Slensmountindexandtur", "lens mount index and turn"),
    ("Slensmountindexandturn", "lens mount index and turn"),
    ("camerabacktoproceedto", "camera back to proceed to"),
    ("cannotbesetforboththeB", "cannot be set for both the B"),
    ("whiteimagesontotheCFcard", "white images onto the CF card"),
    ("maketheimagelookmoreimpressive", "make the image look more impressive"),
    ("camerawilldetectthe", "camera will detect the"),
    ("detectionistwiceassensitiveashorizontal", "detection is twice as sensitive as horizontal"),
    ("eightAFpointsarehorizontal", "eight AF points are horizontal"),
    ("linesensitiveorvertical", "line sensitive or vertical"),
    ("speedoraperturevalueto", "speed or aperture value to"),
    ("Customwhitebalanceselection", "Custom white balance selection"),
    ("indicatesththexposrelevelis", "indicates that the exposure level is"),
    ("thedesireddepthoffield", "the desired depth of field"),
    ("Speedlitesprovidesall", "Speedlites provides all"),
    ("tomakesureitsynchronizesproperlywiththe", "to make sure it synchronizes properly with the"),
    ("whichhasbeenwrittento", "which has been written to"),
    ("Transmissionsbetweenthe", "Transmissions between the"),
    ("Printerscapableofdirectprintingfrom", "Printers capable of direct printing from"),
    ("andtheimagesspecifiedfortheprintordermight", "and the images specified for the print order might"),
    ("currentshutterspeedoraperturebecomesunsuitable", "current shutter speed or aperture becomes unsuitable"),
    ("automaticallytoobtaina", "automatically to obtain a"),
    ("thisremoteswitchhas", "this remote switch has"),
    ("SIRwithaCMOSsensor", "SLR with a CMOS sensor"),
    ("Space atleast", "Space at least"),
    ("was her", "washer"),
    ("thendrain", "then drain"),
    ("orwax", "or wax"),
    ("Ioprevent", "To prevent"),
    ("tollow", "follow"),
    ("safeguardsbelow", "safeguards below"),
    ("excessiveheat", "excessive heat"),
    ("anexplosion", "an explosion"),
    ("oroily", "or oily"),
    ("thedustonthe", "the dust on the"),
    ("becomemoist", "become moist"),
    ("theoutlet", "the outlet"),
    ("tocausea", "to cause a"),
    ("oraanic", "organic"),
    ("eauioment", "equipment"),
    ("OPERATINGINSTRUCTIONS", "OPERATING INSTRUCTIONS"),
    ("INSTANTACTION", "INSTANT ACTION"),
    ("con rm ation", "confirmation"),
    ("sternanchored", "stern anchored"),
    ("andprevent", "and prevent"),
    ("batterypack", "battery pack"),
    ("emiting", "emitting"),
    ("Thefigures", "The figures"),
    ("in dicates", "indicates"),
    ("lt is", "It is"),
    ("formenu", "for menu"),
    ("theEF-Slenswiththe", "the EF-S lens with the"),
    ("turnn", "turn"),
    ("Whenthe", "When the"),
    ("pwer", "power"),
    ("dialtoset", "dial to set"),
)


@dataclass
class ImageAttachmentResult:
    marked_text: str
    attached_image_ids: list[str]
    unmatched_pic_count: int
    strategy: str
    pic_count: int
    image_count: int
    attached_count: int
    suppressed_image_count: int

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def load_manuals(path: Path) -> list[ManualDocument]:
    """Load one or more ManualDocuments from a file.

    Supports:
      - JSONL: one JSON array [text, image_ids] per line → multiple documents
      - Single JSON: one JSON array per file → one document
      - Python literal: via ast.literal_eval
      - Tail-recovery: regex fallback for malformed payloads
    """

    raw = path.read_text(encoding="utf-8")
    lines = raw.strip().splitlines()

    # JSONL detection: try to parse every non-empty line as a JSON array
    if len(lines) > 1:
        documents: list[ManualDocument] = []
        all_valid = True
        for line_num, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                text, image_ids = _validate_manual_payload(data)
            except Exception:
                all_valid = False
                break
            cleaned_text = normalize_manual_text(text)
            product_name = _product_name_from_path(path)
            docs_id = f"{path.stem}_{line_num + 1:03d}"
            documents.append(
                ManualDocument(
                    manual_id=docs_id,
                    product_name=product_name,
                    source_path=path,
                    text=cleaned_text,
                    image_ids=image_ids,
                    pic_count=len(PIC_RE.findall(cleaned_text)),
                    parse_mode="jsonl",
                )
            )

        if all_valid and documents:
            return documents

    # Fall back to single-document parsing
    return [_load_manual_single(path, raw)]


def load_manual(path: Path) -> ManualDocument:
    """Load one manual file (legacy — prefer load_manuals for new code)."""
    return load_manuals(path)[0]


def _load_manual_single(path: Path, raw: str) -> ManualDocument:
    """Parse a single manual document from raw text using structured parsers."""
    text, image_ids, parse_mode = _parse_structured_manual(raw)
    cleaned_text = normalize_manual_text(text)
    return ManualDocument(
        manual_id=path.stem,
        product_name=_product_name_from_path(path),
        source_path=path,
        text=cleaned_text,
        image_ids=image_ids,
        pic_count=len(PIC_RE.findall(cleaned_text)),
        parse_mode=parse_mode,
    )


def _product_name_from_heading(text: str) -> str | None:
    """Extract a short product name from the first heading in the text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            name = stripped.lstrip("# ").strip()
            words = name.split()[:4]
            result = " ".join(words)
            # Skip if the heading is too generic (product-neutral)
            lower = result.lower()
            if any(
                kw in lower
                for kw in (
                    "introduction", "item check", "table of contents",
                    "content", "user manual", "owner", "safety", "product",
                    "getting started", "welcome",
                )
            ):
                continue
            return result if result else None
    return None


def _parse_structured_manual(raw: str) -> tuple[str, list[str], str]:
    for parser_name, parser in (("json", json.loads), ("literal", ast.literal_eval)):
        try:
            data = parser(raw)
        except Exception:
            continue
        text, image_ids = _validate_manual_payload(data)
        return text, image_ids, parser_name

    match = TAIL_IMAGE_LIST_RE.search(raw)
    if not match:
        raise ValueError("cannot parse manual payload or recover tail image list")

    image_ids = [str(item) for item in json.loads(match.group(1))]
    text = raw[: match.start()].strip()
    if text.startswith('["'):
        text = text[2:]
    elif text.startswith("["):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    return _decode_common_escapes(text), image_ids, "tail-recovery"


def _validate_manual_payload(data: object) -> tuple[str, list[str]]:
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("manual payload must be a list shaped as [text, image_ids]")
    text = str(data[0])
    raw_image_ids = data[1]
    if not isinstance(raw_image_ids, list):
        raise ValueError("manual image_ids must be a list")
    return text, [str(item) for item in raw_image_ids]


def _decode_common_escapes(text: str) -> str:
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\/", "/")
    )


def normalize_manual_text(text: str) -> str:
    """Normalize manual text while preserving headings and picture markers."""

    text = _decode_common_escapes(text)
    text = _decode_unicode_escape_literals(text)
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = _normalize_latex_noise(text)
    text = _fix_common_ocr_glued_words(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*<PIC>\s*", f"\n{PIC_TOKEN}\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\n)#\s+", "\n# ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_unicode_escape_literals(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)

    return UNICODE_ESCAPE_RE.sub(repl, text)


def _normalize_latex_noise(text: str) -> str:
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = re.sub(r"\\[,;:!~% ]+", " ", text)
    text = re.sub(r"[{}$]+", " ", text)
    text = re.sub(r"[_^]\s*[A-Za-z0-9+\-*/=().]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def _fix_common_ocr_glued_words(text: str) -> str:
    fixed = text
    for old, new in OCR_WORD_FIXES:
        fixed = fixed.replace(old, new)
    fixed = re.sub(r"(?m)^([·•●-])(?=[A-Za-z])", r"\1 ", fixed)
    fixed = re.sub(r"([·•●])(?=[A-Za-z])", r"\1 ", fixed)
    fixed = re.sub(r"\blf\b", "If", fixed)
    fixed = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", fixed)
    fixed = re.sub(r"(?<=[a-z])(?=\d)", " ", fixed)
    fixed = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", fixed)
    fixed = re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", fixed)
    return fixed


def attach_image_markers(text: str, image_ids: list[str]) -> ImageAttachmentResult:
    """Replace each <PIC> with an ordered image marker used during chunking.

    When a manual has more picture placeholders than image ids, the extra
    placeholders are kept as a generic missing marker so they do not pollute
    the image index with synthetic ids.
    """

    pic_count = len(PIC_RE.findall(text))
    strategy = _choose_image_attachment_strategy(pic_count=pic_count, image_count=len(image_ids))
    if strategy == "suppress_misaligned":
        marked_text = PIC_RE.sub("\n[[PIC_MISSING]]\n", text)
        return ImageAttachmentResult(
            marked_text=marked_text,
            attached_image_ids=[],
            unmatched_pic_count=pic_count,
            strategy=strategy,
            pic_count=pic_count,
            image_count=len(image_ids),
            attached_count=0,
            suppressed_image_count=len(image_ids),
        )

    parts = PIC_RE.split(text)
    marked_parts: list[str] = []
    attached_ids: list[str] = []
    unmatched_pic_count = 0

    for index, part in enumerate(parts):
        marked_parts.append(part)
        if index >= len(parts) - 1:
            continue
        if index < len(image_ids):
            image_id = image_ids[index]
            attached_ids.append(image_id)
            marked_parts.append(f"\n[[PIC:{image_id}]]\n")
        else:
            unmatched_pic_count += 1
            marked_parts.append("\n[[PIC_MISSING]]\n")

    return ImageAttachmentResult(
        marked_text="".join(marked_parts),
        attached_image_ids=attached_ids,
        unmatched_pic_count=unmatched_pic_count,
        strategy=strategy,
        pic_count=pic_count,
        image_count=len(image_ids),
        attached_count=len(attached_ids),
        suppressed_image_count=0,
    )


def _choose_image_attachment_strategy(*, pic_count: int, image_count: int) -> str:
    """Always use sequential attachment: available images map to first N PICs, rest become PIC_MISSING."""
    return "sequential"


def _product_name_from_path(path: Path) -> str:
    name = path.stem
    return name[: -len("手册")] if name.endswith("手册") else name
