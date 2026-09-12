#!/usr/bin/env python3
"""Build the offline, file-only annotation packages (no browser, no server).

Each package contains: slide images, transcripts, two CSV workbooks with blank columns to
fill in, and a short instruction file. Blinding (deranged M1/M2/M3 and A/B/C aliases) and
opaque item IDs are reused unchanged from the frozen sealed mapping.
"""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path

HE = Path(__file__).resolve().parents[1]
OUT = HE / "offline_packages"
CH = list(csv.DictReader(open(HE / "frozen_30_slide_manifest.csv")))
CKEYS = [(r["lecture"], r["slide_id"]) for r in CH]
CROW = {(r["lecture"], r["slide_id"]): r for r in CH}

ANSWERS = "both | transcript | picture | neither | unsure"

INSTR = """WHAT IS IN THIS FOLDER
======================

slides/          30 slide pictures  (slide_01.jpg ... slide_30.jpg)
transcripts/     30 text files, what the speaker said about each slide
PART1_grouping.csv
PART2_support.csv
INSTRUCTIONS.txt   this file

Open the two CSV files in Excel, LibreOffice, or Google Sheets.
Everything you need is inside this folder. Nothing else is required.


HOW TO WORK
===========

Each row of both CSV files tells you which slide it belongs to, in the
"slide_picture" and "transcript_file" columns. Open that picture and that
transcript, and judge the row using only those two things.

Only fill in the columns whose names start with FILL_. Leave every other
column exactly as it is. Do not add, delete, sort or reorder rows.

Save the file in the same CSV format when you are done.


PART 1  -  PART1_grouping.csv
=============================

Each screenful of rows with the same "panel" value belongs to one slide.
Within a panel the rows are split into groups A, B and C. The rows are
short pieces of text pulled out of that slide.

Some rows in different groups say the same thing in different words.

FILL_label   Write a short label, anything you like: g1, g2, x, ct ...
             Rows that mean THE SAME THING get the SAME label.
             A row with no match anywhere gets its own label.
             Two rows inside the same group may share a label if they
             really do duplicate each other.

             Labels only matter inside one panel. You can reuse g1 in the
             next panel for something completely different.

FILL_unsure  Write  y  if you could not decide. Otherwise leave blank.
FILL_note    Anything you want to record. Optional.

Same thing (examples):
   magnetic resonance imaging   /   MRI
   image sharpness              /   sharpness of the image
   scanner -produces-> images   /   scanner -generates-> pictures

NOT the same thing (give different labels):
   image noise                  vs  image sharpness
   contrast agent               vs  iodinated contrast agent
   detector -measures-> signal  vs  detector -measures-> background signal

If one row says more than the other, or is a wider or narrower version of
it, they are NOT the same. Different labels.

For rows shaped  subject -predicate-> object , judge the whole statement,
not only whether the first and last words appear in both.

If you cannot decide whether two rows are the same, give them different
labels and put  y  in FILL_unsure.

A few panels have no rows at all. Nothing to do there.


PART 2  -  PART2_support.csv
============================

Each row is one piece of text and one slide. Look at the picture and the
transcript for that slide, and decide where the text is backed up.

FILL_answer  Write exactly one of:

   both         the picture AND the transcript both back it up
   transcript   the transcript backs it up, the picture alone does not
   picture      the picture backs it up, the transcript alone does not
   neither      neither one backs it up
   unsure       you cannot tell

FILL_note    Optional.

A useful way to decide: ask yourself separately "would the picture alone
be enough?" and "would the transcript alone be enough?", then combine the
two answers.

Words printed on the slide picture count as part of the picture.

For rows shaped  subject -predicate-> object , the whole statement has to
be backed up, not just the first and last words. If a slide shows the
words "detector" and "noise" but never says they are connected, then
detector -reduces-> noise is NOT backed up by the picture.

Examples:
   picture shows a labelled scanner, speaker says "this is the scanner"
      -> both
   picture is a photo with no text, speaker says "developed in 1971",
      row is "developed in 1971"                        -> transcript
   picture shows an equation the speaker never mentions -> picture
   neither the picture nor the speaker mentions cost,
      row is "scanner -costs-> two million"             -> neither
   you genuinely cannot tell                            -> unsure


NOTES
=====

The letters A, B, C and M1, M2, M3 are just labels. They have no meaning.

Use  unsure  rather than guessing. There is no penalty for it.

You are not judging whether something is important, correct in general,
or well written. Only whether the slide and transcript in front of you
back it up.

Please do not look anything up outside this folder, and please do not
compare answers with the other person until you have both finished.

You do not have to do it all in one sitting. Just save the CSV files.


WHEN YOU ARE FINISHED
=====================

Send back the two CSV files, with the same names:

   PART1_grouping.csv
   PART2_support.csv
"""


def build(annot: str):
    smap = json.loads((HE / f"sealed_mapping/annotator_{annot}_mapping.json").read_text())
    panels = json.loads((HE / f"package_data/annotator_{annot}/panels.json").read_text())["panels"]
    ev = json.loads((HE / f"package_data/annotator_{annot}/evidence.json").read_text())["items"]

    pkg = OUT / f"annotator_{annot}"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "slides").mkdir(parents=True)
    (pkg / "transcripts").mkdir(parents=True)
    for i, k in enumerate(CKEYS, 1):
        shutil.copy(CROW[k]["image_path"], pkg / "slides" / f"slide_{i:02d}.jpg")
        shutil.copy(CROW[k]["text_path"], pkg / "transcripts" / f"slide_{i:02d}.txt")

    # ---- PART 1 ----------------------------------------------------------------
    rows = []
    for p in panels:
        if p["no_annotatable_items"]:
            rows.append({"panel": p["panel_id"], "type": p["task_display"],
                         "slide_picture": f"slides/slide_{p['slide_index']:02d}.jpg",
                         "transcript_file": f"transcripts/slide_{p['slide_index']:02d}.txt",
                         "set": p["model_alias"], "group": "", "row_id": "",
                         "text": "(nothing to do for this panel)",
                         "FILL_label": "", "FILL_unsure": "", "FILL_note": ""})
            continue
        for s in p["sets"]:
            for it in s["items"]:
                rows.append({"panel": p["panel_id"], "type": p["task_display"],
                             "slide_picture": f"slides/slide_{p['slide_index']:02d}.jpg",
                             "transcript_file": f"transcripts/slide_{p['slide_index']:02d}.txt",
                             "set": p["model_alias"], "group": s["source_alias"],
                             "row_id": it["item_id"], "text": it["text"],
                             "FILL_label": "", "FILL_unsure": "", "FILL_note": ""})
    with (pkg / "PART1_grouping.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- PART 2 ----------------------------------------------------------------
    erows = []
    for n, v in enumerate(ev, 1):
        erows.append({"no": n, "type": v["task_display"],
                      "slide_picture": f"slides/slide_{v['slide_index']:02d}.jpg",
                      "transcript_file": f"transcripts/slide_{v['slide_index']:02d}.txt",
                      "set": v["model_alias"], "row_id": v["evidence_id"],
                      "text": v["item_text"],
                      "FILL_answer": "", "FILL_note": ""})
    with (pkg / "PART2_support.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(erows[0].keys()))
        w.writeheader(); w.writerows(erows)

    (pkg / "INSTRUCTIONS.txt").write_text(INSTR, encoding="utf-8")
    return len(rows), len(erows)


OUT.mkdir(exist_ok=True)
for a in ("A", "B"):
    n1, n2 = build(a)
    print(f"annotator_{a}: PART1 rows={n1}  PART2 rows={n2}")
