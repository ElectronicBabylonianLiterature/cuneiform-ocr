# What the two sign finders were trained on

This folder lists, tablet by tablet, which photographs went into the training of `sign-detector-u1.pt` and
`detr173-ft1b-e9.safetensors`, so that anyone can build a test set of tablets the models have not seen. It holds identifiers
and counts only: no photographs, no annotations, no transliterations.

| File | Rows | What it is |
|---|---|---|
| `trained-on-sign-detector-u1.tsv` | 19,938 | Every tablet with at least one tile in the training or the validation of `sign-detector-u1` |
| `trained-on-detr173-ft1b-e9.tsv` | 548 | Every fragment in our further training of the DETR (not eBL's own training, see below) |
| `test-fragments.tsv` | 24 | The 24 test fragments of the table in `README.md`, with their hand boxes |

## How names are written and matched

- `identifier` is the name under which the photograph came to us: a CDLI number (`P123456`) for a CDLI photograph, the eBL
  museum or fragment number (`BM.45659`, `K.8678`, `1880,0719.100`, `HS.2176.A`) for a photograph served by the eBL platform.
- One row is one identifier, and so one photograph. A tablet photographed by CDLI and by eBL appears as two rows (see
  "Tablets in the training under two names" below).
- `other_names_and_joined_pieces`: other names of the same object that we could find, from the CDLI catalogue (museum
  number, P-numbers in its join information; dump of November 2026) and from the eBL fragment records we hold (CDLI number,
  joins). **Joined pieces are included**: when a fragment is part of a join, the numbers of the other pieces are listed too
  (the lists are complete as far as our sources go; the longest has 42 names). 728 rows of the first file have no other
  name known to us; that does not mean the object has none.
- To check whether a tablet of yours is in a list, compare every name it has against `identifier` and
  `other_names_and_joined_pieces`, after normalising both sides: upper case; `P`/`Q`/`X` + digits written with six digits;
  otherwise spaces, dots, underscores, hyphens, commas, colons and slashes dropped as separators, and leading zeros of every
  numeric part dropped, with a dot kept between two numeric parts (so `VAT 04786` = `VAT.4786`, `1880,0719.100` =
  `1880.719.100`, and `Rm 2, 60` stays apart from `Rm 260`).
- `period` / `script_or_period`: the CDLI catalogue's period for a CDLI photograph, the eBL record's script period for an eBL
  photograph ("(CDLI)" when it was taken from the CDLI entry of the same object). Blank where we hold neither (51 rows).

## sign-detector-u1

**How the list was established.** From the training run's own record: the command line and data configuration written
when the run was started (YOLO11m from the COCO weights, 30 epochs, batch 32, 640-pixel tiles; one training folder plus four
extra folders), the count of tiles the trainer reported at its start (50,019 training images, 7,768 validation images),
and the archived copies of the six tile sets that were put into those folders. Tile file names carry the tablet (for
example `cdli_P011074_0__0_0` = P011074, first face, tile at 0,0). The run was not interrupted or resumed; the weights
published are the copy the trainer kept as best (epoch 29 of 30, chosen on the validation tiles; checksum in `SHA256SUMS`).
The starting weights (COCO) contain no cuneiform.

**Reconciliation with the run's own count (exact):**

| Tile set | Photographs | Kind of boxes | Distinct tiles | Tiles per epoch | Boxes in the distinct tiles |
|---|---|---|---|---|---|
| Alignment tiles, CDLI | CDLI | machine-made | 20,251 | 20,251 | 560,263 |
| Alignment tiles, eBL | eBL | machine-made | 8,388 | 8,388 | 367,266 |
| Hand-boxed tiles (eBL annotations, set 1) | eBL | hand-drawn | 3,486 | 10,458 (each three times) | 97,623 |
| Alignment tiles, older-period Akkadian | CDLI | machine-made | 4,609 | 4,609 | 275,859 |
| Alignment tiles, Old Babylonian | CDLI | machine-made | 1,756 | 1,756 | 79,768 |
| Hand-boxed tiles (eBL annotations, set 2) | eBL | hand-drawn | 1,444 | 4,332 (each three times) | 50,215 |
| Background tiles (no signs: charts, rulers, blank faces, margins) | eBL | none | 225 | 225 | 0 |
| **Training** | | | **40,159** | **50,019** | **1,430,994** |
| Validation (best-epoch choice only, never trained on) | CDLI 4,941 / eBL 2,422 + 405 hand-boxed | | 7,768 | | 253,585 |

The first two rows plus the three-fold first hand set are the trainer's 39,097 "main" tiles; the other four rows are the
four extra folders (4,609 + 1,756 + 4,332 + 225). 28,639 + 10,458 + 4,609 + 1,756 + 4,332 + 225 = 50,019.

**Totals by source (training rows only):**

| Source | Tablets | Tiles per epoch | Boxes |
|---|---|---|---|
| CDLI photographs (all machine-made boxes) | 12,542 | 26,616 | 915,890 |
| eBL photographs | 3,954 | 23,403 | 515,104 |
| of which with machine-made boxes | 3,076 | 8,388 | 367,266 |
| of which with hand-drawn boxes | 794 | 14,790 (4,930 distinct) | 147,838 |
| of which background tiles only or in part | 156 | 225 | 0 |

A further 3,442 tablets (2,567 CDLI, 875 eBL) are in the validation tiles only (`role` = `validation only`); 13 tablets have
tiles in both. They were not trained on, but the copy of the weights was chosen on them: for a strict held-out test, leave
them out too.

**By period (training rows; periods grouped, the file has the catalogue's own wording):**

| Period group | CDLI tablets / tiles / boxes | eBL tablets / tiles / boxes |
|---|---|---|
| Third millennium (ED, Fara, Presargonic, Old Akkadian / Sargonic, Lagash II, Ur III) | 10,460 / 20,058 / 548,403 (Ur III 10,050) | 353 / 2,589 / 20,347 |
| Old Babylonian, Old Assyrian | 1,877 / 5,263 / 292,130 | 381 / 5,144 / 53,701 |
| Middle Babylonian, Middle Assyrian (and 2 Middle Hittite, 1 Middle Elamite) | 103 / 506 / 29,720 | 90 / 2,070 / 24,474 |
| First millennium (Neo-Assyrian, Neo-Babylonian, Persian, Late Babylonian, Hellenistic, Parthian) | 53 / 513 / 31,794 | 3,123 / 13,571 / 415,870 |
| Period not known to us | 49 / 276 / 13,843 | 7 / 29 / 712 |

**Kinds of boxes (`box_kinds`).**
- *alignment* (machine-made): an earlier sign finder of ours and a sign classifier were run on the photograph and their boxes
  aligned with the published transliteration of the same tablet (CDLI ATF, or the eBL edition); only boxes the alignment
  confirmed were kept, compound signs and number signs repaired afterwards. These labels are therefore not independent of
  earlier models of ours, which were trained on overlapping material; tablets that only those earlier models saw are not
  listed here (their influence on `sign-detector-u1` is indirect, through the labels).
- *hand* (hand-drawn): sign boxes drawn on the eBL platform (types HasSign, Number, Damaged, PartiallyBroken, UnclearSign,
  CompoundGrapheme; one class). Photographs scaled so that the median sign is 64 pixels tall; only tiles densely covered by
  boxes were kept.
- *background*: tiles with no box at all, cut from parts of eBL photographs without signs.

**Columns of `trained-on-sign-detector-u1.tsv`:** `identifier`, `source`, `other_names_and_joined_pieces`, `period`, `role`
(training / validation only), `box_kinds`, `tiles_alignment`, `tiles_hand`, `tiles_background` (distinct training tiles),
`tiles_presented_per_epoch` (hand tiles counted three times), `boxes_in_training_tiles`, `validation_tiles`,
`boxes_in_validation_tiles`, `also_in_training_as` (the identifier of another training row for the same object).

## detr173-ft1b-e9

**Starting weights.** The published 173-class weights ([Zenodo 10.5281/zenodo.17395154](https://doi.org/10.5281/zenodo.17395154))
were trained by the eBL team on their own annotated fragments (their public data releases). Every fragment of those releases
therefore counts as seen by this model as well. We do not list eBL's training set; see their releases and paper.

**Our further training (all rows of `trained-on-detr173-ft1b-e9.tsv`):**

| Set | Fragments | Hand boxes | Tiles | Boxes in the tiles |
|---|---|---|---|---|
| Hand-annotated on the eBL platform, not in eBL's 2023 release (`set` = newly hand-annotated, batch 1 and batch 2) | 296 (106 + 190) | 41,682 | 2,533 | 79,680 |
| eBL's data release of November 2023 (`ebl-30-11`) | 252 | 39,880 | 2,160 | 79,750 |
| **Total** | **548** | **81,562** | **4,693** | **159,430** |

Each photograph was scaled so that its median hand box is 64 pixels tall and cut into 800-pixel tiles at a stride of 600; a
box belongs to a tile when 60% of its area lies inside; tiles with fewer than 3 boxes were dropped. `tiles` and
`boxes_in_tiles` were recomputed here from the boxes and photograph sizes with that rule: they agree with the tiling log of
the run for every one of the 548 fragments, and in total with the 4,693 tiles and 159,430 boxes of its training file.
`hand_boxes` is the number of distinct boxes on the photograph.

**Columns:** `identifier`, `source`, `other_names_and_joined_pieces`, `script_or_period`, `set`, `box_kind`, `hand_boxes`,
`tiles`, `boxes_in_tiles`.

## test-fragments.tsv

The 24 fragments of the table in `README.md`: hand-annotated on the eBL platform, in none of eBL's public data releases up to
December 2025, and, under every name known to us (checked against both lists above, the validation-only rows included), in
neither model's training. Columns: `identifier`, `cdli`, `script`, `period_group`, `hand_sign_boxes` (boxes with the mark
"hand" of the types HasSign, Number, Damaged, PartiallyBroken, UnclearSign: the boxes scored), `narrow_boxes` (under 0.6 of
the fragment's median box width), `box_types`. Totals: 8,292 boxes, 675 narrow; first millennium 12 fragments / 4,765 boxes,
second 5 / 1,700, third 7 / 1,827 (these reproduce the README's figures exactly). The copy `ft1b-e9` was chosen after pass 9
with these fragments in view (see README).

## Before you use the lists for a held-out test

1. **Tablets in the training under two names.** `sign-detector-u1` was trained before we matched tablets across their CDLI and
   eBL names, so some objects are in it twice, once per photograph. 94 objects are in it as a CDLI photograph AND as an eBL
   photograph (column `also_in_training_as`):
   P134314 = HS.1002, P134318 = HS.1006, P134336 = HS.1024, P134339 = HS.1027, P134341 = HS.1029, P134343 = HS.1031,
   P134346 = HS.1034, P134351 = HS.1039, P134352 = HS.1040, P134353 = HS.1041, P134356 = HS.1044, P134365 = HS.1053,
   P134368 = HS.1056, P134370 = HS.1058, P134372 = HS.1060, P134375 = HS.1063, P134378 = HS.1066, P134385 = HS.1073.A,
   P134387 = HS.1075, P134390 = HS.1078, P134394 = HS.1082, P134399 = HS.1087, P134400 = HS.1088, P134402 = HS.1090,
   P134405 = HS.1093, P134408 = HS.1096, P134412 = HS.1100, P134413 = HS.1101, P134414 = HS.1102, P134416 = HS.1104,
   P134420 = HS.1108, P134421 = HS.1109, P134423 = HS.1111, P134425 = HS.1113, P134426 = HS.1114, P134431 = HS.1119,
   P134433 = HS.1121, P134435 = HS.1123, P134437 = HS.1125, P134439 = HS.1127, P134450 = HS.1139, P134454 = HS.1143,
   P134456 = HS.1145, P134458 = HS.1147, P134462 = HS.1151, P134464 = HS.1153, P134465 = HS.1154, P134466 = HS.1155,
   P134472 = HS.1161, P134479 = HS.1168, P134483 = HS.1172, P134491 = HS.1180, P134492 = HS.1181, P134493 = HS.1182,
   P134494 = HS.1183, P134511 = HS.1200, P134527 = HS.1216, P134570 = HS.1259, P134572 = HS.1261, P134573 = HS.1262,
   P134591 = HS.1281, P134593 = HS.1283, P134600 = HS.1290, P134605 = HS.1295, P134610 = HS.1300, P134612 = HS.1302,
   P134620 = HS.1310, P134626 = HS.1316, P134663 = HS.1353, P134664 = HS.1354, P134666 = HS.1356, P134667 = HS.1357,
   P143228 = YBC.4, P143695 = YBC.103, P144563 = YBC.6, P145224 = YBC.11786, P200608 = NBC.7439, P212386 = HS.1352,
   P212711 = NBC.7021, P235379 = K.138, P235455 = K.139, P235487 = K.11428, P235504 = K.200, P235510 = K.230,
   P257677 = CBS.112, P258717 = CBS.1384, P263338 = CBS.8538, P265697 = CBS.10489, P331083 = U.18735, P331089 = U.18815,
   P331097 = U.18840, P346853 = UET-6.816, P394449 = K.2462, P424886 = Rm.945.
   A test that excludes a tablet by one name only can therefore still contain it.
2. **Joined tablets appear under the name their photograph had.** A photograph of a joined tablet is listed under the number
   it was served with; the other pieces are in `other_names_and_joined_pieces`. Where pieces of one join were photographed and
   trained separately, each piece is its own row. In `sign-detector-u1` 30 such groups are in the training under more than one
   name: P209424 + P209750 + P209753; K.12719 + K.12836; K.10433 + K.14469; K.4132 + K.8266; K.11213 + K.17732;
   K.12420 + K.3962; BM.37494 + BM.37616 + BM.38076; K.10125 + K.5946; K.11802 + K.17250; K.12524 + K.8307;
   K.13435 + K.4096 + K.6563; K.17902 + Sm.1563; K.14160 + K.2259; K.3240 + K.5907.A; K.12163 + K.6158; K.12494 + K.6715;
   2024,6004.279 + BM.48011; 2024,6004.13 + 2024,6004.368; K.14171 + K.17920; K.17871 + K.6760; K.10481 + K.6456;
   1881,0727.124 + K.12000; K.16011 + K.17060 + K.8806; K.16795 + Rm.593; K.7652 + Sm.1604; HS.1342 + HS.2075;
   HS.2072 + HS.2116; HS.2115 + HS.2121; NBC.3688 + NBC.3689; K.205 + Rm-II.101. In the DETR list: HS.2189 + HS.2190 and
   BM.37616 + BM.38076. A test should exclude every piece of a join if any piece was trained on.
3. **The two models overlap.** 294 of the 548 DETR fragments are also in the training of `sign-detector-u1` under the same name
   (227 of the 252 release fragments, 67 of the 296 newly hand-annotated ones, there with machine-made boxes).
4. **Tiles are counted per photograph, and tiles overlap.** Tiles are cut with a stride smaller than the tile (480 of 640 for
   `sign-detector-u1`, 600 of 800 for the DETR), so one sign can lie in two to four tiles; the box columns count it in every
   tile. Different tile sets of the same tablet (for example machine-made and hand-boxed tiles of one eBL photograph) may cover
   the same area twice. A CDLI photograph usually shows several faces; they were cut into tiles face by face.
5. **The other-name lists are as complete as our sources.** They come from the CDLI catalogue and the eBL records we hold, not
   from every museum catalogue. Check your own tablets' names as well.
6. **What is reconstructed rather than read off the run.** The tile lists and counts of `sign-detector-u1` are taken from the
   archived copies of the six tile sets and the run's tile list; the training machine itself is no longer running. Two small
   points could not be checked on it: (a) the box counts of the first hand-boxed set come from the archived build of those
   tiles; all 3,486 training tiles named in the run's list are in that build, but the copy on the training machine held a
   slightly different number of tiles in its folder (3,545 against 3,581 training images, of which only the 3,486 listed were
   used), so a few of those label files may differ from the archive; (b) the 225 background-tile label files are empty in the
   archive, as their builder writes them, while one line of our staging log counted them as non-empty. Neither changes which
   tablets are in the list. For the DETR, the training file itself is on a stopped machine; the list was rebuilt from the
   inputs of its tiling step and agrees with that step's log fragment by fragment.
