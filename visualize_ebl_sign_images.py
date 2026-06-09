#!/usr/bin/env python3
"""Fetch and visualize cropped sign images from the eBL API or MongoDB.

Example:
    python visualize_ebl_sign_images.py KI
    python visualize_ebl_sign_images.py DINGIR --include-unclustered --limit 50
    python visualize_ebl_sign_images.py KI --source mongo
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_API_BASE = "https://ebl.badw.de/api"
DEFAULT_ENV_FILE = ".env"
DEFAULT_MONGO_URI_ENV = "MONGODB_URI"
DEFAULT_MONGO_DB = "ebl"

PERIOD_ABBREVIATIONS = {
    "None": "",
    "Uncertain": "Unc",
    "Uruk IV": "Uruk4",
    "Uruk III-Jemdet Nasr": "JN",
    "ED I-II": "ED1_2",
    "Fara": "Fara",
    "Presargonic": "PSarg",
    "Sargonic": "Sarg",
    "Lagash II": "Lag2",
    "Ur III": "Ur3",
    "Old Assyrian": "OA",
    "Old Babylonian": "OB",
    "Middle Babylonian": "MB",
    "Middle Assyrian": "MA",
    "Hittite": "Hit",
    "Neo-Assyrian": "NA",
    "Neo-Babylonian": "NB",
    "Late Babylonian": "LB",
    "Persian": "Per",
    "Hellenistic": "Hel",
    "Parthian": "Par",
    "Proto-Elamite": "PElam",
    "Old Elamite": "OElam",
    "Middle Elamite": "MElam",
    "Neo-Elamite": "NElam",
    "Luwian": "Luw",
    "Aramaic": "Aram",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch eBL sign image annotations and render them as an HTML grid."
    )
    parser.add_argument(
        "sign",
        nargs="?",
        default="KI",
        help="Sign name to fetch, e.g. KI, DINGIR, LU2. Default: KI.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Base API URL. Default: {DEFAULT_API_BASE}",
    )
    parser.add_argument(
        "--source",
        choices=["api", "mongo"],
        default="api",
        help="Fetch source. Use 'mongo' to query the database directly. Default: api.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(DEFAULT_ENV_FILE),
        help=f"Path to the .env file used by --source mongo. Default: {DEFAULT_ENV_FILE}",
    )
    parser.add_argument(
        "--mongo-uri-env",
        default=DEFAULT_MONGO_URI_ENV,
        help=f"Environment variable containing the MongoDB URI. Default: {DEFAULT_MONGO_URI_ENV}",
    )
    parser.add_argument(
        "--mongo-db",
        default=DEFAULT_MONGO_DB,
        help=f"MongoDB database name. Default: {DEFAULT_MONGO_DB}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all images instead of only centroid/canonical images.",
    )
    parser.add_argument(
        "--include-unclustered",
        action="store_true",
        help="When fetching centroids, also include images without PCA clustering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of rendered images. 0 means no limit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path. Default: ebl_sign_<SIGN>_images.html",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Optional path for saving the raw API JSON response.",
    )
    return parser.parse_args()


def build_url(
    api_base: str, sign: str, centroids_only: bool, include_unclustered: bool
) -> str:
    params = {
        "centroids_only": str(centroids_only).lower(),
    }
    if include_unclustered:
        params["include_unclustered"] = "true"

    return f"{api_base.rstrip('/')}/signs/{sign}/images?{urlencode(params)}"


def fetch_json(url: str) -> list[dict[str, Any]]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "cuneiform-sign-visualizer/1.0",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as error:
        raise SystemExit(f"HTTP {error.code} while fetching {url}: {error.reason}") from error
    except URLError as error:
        raise SystemExit(f"Could not fetch {url}: {error.reason}") from error

    data = json.loads(payload)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list, got: {type(data).__name__}")
    return data


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without requiring python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def get_mongodb_uri(env_file: Path, env_name: str) -> str:
    load_env_file(env_file)
    uri = os.getenv(env_name)
    if not uri:
        raise SystemExit(
            f"{env_name} is not set. Add it to {env_file} or export it in the shell."
        )
    return uri


def script_to_abbreviation(script: Any) -> str:
    if not isinstance(script, dict):
        return "" if script is None else str(script)
    period = script.get("period")
    return PERIOD_ABBREVIATIONS.get(period, period or "")


def mongo_pipeline(
    sign: str, centroids_only: bool, include_unclustered: bool
) -> list[dict[str, Any]]:
    annotation_filter_conditions: list[dict[str, Any]] = [
        {"$eq": ["$$annotation.data.signName", sign]},
        {"$eq": ["$$annotation.data.type", "HasSign"]},
        {
            "$ne": [
                {"$ifNull": ["$$annotation.croppedSign.imageId", None]},
                None,
            ]
        },
    ]

    if centroids_only:
        centroid_condition = {
            "$eq": ["$$annotation.pcaClustering.isCentroid", True]
        }
        if include_unclustered:
            annotation_filter_conditions.append(
                {
                    "$or": [
                        centroid_condition,
                        {
                            "$eq": [
                                {"$ifNull": ["$$annotation.pcaClustering", None]},
                                None,
                            ]
                        },
                    ]
                }
            )
        else:
            annotation_filter_conditions.append(centroid_condition)

    return [
        {
            "$match": {
                "annotations.data.signName": {
                    "$regex": re.escape(sign),
                    "$options": "i",
                }
            }
        },
        {
            "$lookup": {
                "from": "fragments",
                "localField": "fragmentNumber",
                "foreignField": "_id",
                "as": "fragment",
            }
        },
        {"$unwind": "$fragment"},
        {
            "$project": {
                "fragmentNumber": 1,
                "annotations": {
                    "$filter": {
                        "input": "$annotations",
                        "as": "annotation",
                        "cond": {"$and": annotation_filter_conditions},
                    }
                },
                "date": "$fragment.date",
                "provenance": "$fragment.archaeology.site",
                "scriptRaw": "$fragment.script",
            }
        },
        {"$unwind": "$annotations"},
        {
            "$lookup": {
                "from": "cropped_sign_images",
                "localField": "annotations.croppedSign.imageId",
                "foreignField": "_id",
                "as": "imageDoc",
            }
        },
        {"$unwind": "$imageDoc"},
        {
            "$project": {
                "_id": 0,
                "fragmentNumber": 1,
                "image": "$imageDoc.image",
                "scriptRaw": 1,
                "label": "$annotations.croppedSign.label",
                "date": 1,
                "provenance": 1,
                "annotationId": "$annotations.data.id",
                "pcaClustering": "$annotations.pcaClustering",
            }
        },
    ]


def normalize_mongo_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    normalized["script"] = script_to_abbreviation(normalized.pop("scriptRaw", None))
    if "pcaClustering" in normalized and normalized["pcaClustering"] is None:
        normalized.pop("pcaClustering")
    return normalized


def fetch_mongo_sign_images(
    sign: str,
    centroids_only: bool,
    include_unclustered: bool,
    env_file: Path,
    mongo_uri_env: str,
    mongo_db: str,
) -> list[dict[str, Any]]:
    try:
        from pymongo import MongoClient
    except ImportError as error:
        raise SystemExit(
            "pymongo is required for --source mongo. Install it with: pip install pymongo"
        ) from error

    uri = get_mongodb_uri(env_file, mongo_uri_env)
    client = MongoClient(uri)
    try:
        collection = client[mongo_db]["annotations"]
        cursor = collection.aggregate(
            mongo_pipeline(sign, centroids_only, include_unclustered),
            allowDiskUse=True,
        )
        return [normalize_mongo_item(item) for item in cursor]
    finally:
        client.close()


def image_data_uri(image: str) -> str:
    if image.startswith("data:image/"):
        return image

    raw = base64.b64decode(image[:128] + "===")
    if raw.startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif raw.startswith(b"GIF"):
        mime_type = "image/gif"
    elif raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
        mime_type = "image/webp"
    else:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{image}"


def display_value(value: Any) -> str:
    if value in (None, "", [], {}):
        return "-"
    if isinstance(value, (dict, list)):
        return html.escape(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        )
    return html.escape(str(value))


def render_card(item: dict[str, Any], index: int) -> str:
    clustering = item.get("pcaClustering") or {}
    fields = [
        ("fragment", item.get("fragmentNumber")),
        ("label", item.get("label")),
        ("script", item.get("script")),
        ("date", item.get("date")),
        ("form", clustering.get("form")),
        ("cluster", clustering.get("clusterId")),
        ("rank", clustering.get("clusterRank")),
        ("size", clustering.get("clusterSize")),
        ("main", clustering.get("isMain")),
        ("annotation", item.get("annotationId")),
        ("provenance", item.get("provenance")),
    ]
    details = "\n".join(
        f"<dt>{html.escape(name)}</dt><dd>{display_value(value)}</dd>"
        for name, value in fields
    )
    src = image_data_uri(str(item["image"]))
    alt = f"{item.get('label') or 'sign image'} {index + 1}"
    return f"""
      <article class="card">
        <div class="image-wrap">
          <img src="{src}" alt="{html.escape(alt)}" loading="lazy">
        </div>
        <dl>{details}</dl>
      </article>
    """


def render_html(sign: str, source_url: str, items: list[dict[str, Any]]) -> str:
    forms = Counter(
        (item.get("pcaClustering") or {}).get("form") or "unclustered" for item in items
    )
    form_summary = ", ".join(
        f"{html.escape(str(form))}: {count}" for form, count in sorted(forms.items())
    )
    cards = "\n".join(render_card(item, index) for index, item in enumerate(items))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>eBL sign images: {html.escape(sign)}</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #202124;
      --muted: #5f6368;
      --line: #d8d2c6;
      --paper: #fbfaf7;
      --panel: #ffffff;
      --accent: #1f6f78;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--text);
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 16px;
      border-bottom: 1px solid var(--line);
      background: #fffdf8;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 44px);
      font-weight: 720;
      letter-spacing: 0;
    }}
    .meta {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }}
    .meta a {{ color: var(--accent); word-break: break-all; }}
    main {{
      padding: 22px clamp(18px, 4vw, 48px) 48px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .image-wrap {{
      min-height: 150px;
      display: grid;
      place-items: center;
      padding: 14px;
      background:
        linear-gradient(45deg, #eee 25%, transparent 25%),
        linear-gradient(-45deg, #eee 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #eee 75%),
        linear-gradient(-45deg, transparent 75%, #eee 75%);
      background-size: 18px 18px;
      background-position: 0 0, 0 9px, 9px -9px, -9px 0;
    }}
    img {{
      max-width: 100%;
      max-height: 180px;
      object-fit: contain;
      image-rendering: auto;
    }}
    dl {{
      margin: 0;
      padding: 12px;
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr);
      gap: 5px 10px;
      font-size: 12px;
      line-height: 1.35;
    }}
    dt {{
      color: var(--muted);
      font-weight: 650;
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .empty {{
      max-width: 760px;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(sign)}</h1>
    <div class="meta">
      <div>{len(items)} images rendered</div>
      <div>{html.escape(form_summary) if form_summary else "No PCA form data"}</div>
      <div>{html.escape(source_url)}</div>
    </div>
  </header>
  <main>
    {f'<section class="grid">{cards}</section>' if cards else '<div class="empty">No images returned by the selected source.</div>'}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    centroids_only = not args.all
    output = args.output or Path(f"ebl_sign_{args.sign}_images.html")

    if args.source == "api":
        source = build_url(
            args.api_base,
            args.sign,
            centroids_only=centroids_only,
            include_unclustered=args.include_unclustered,
        )
        items = fetch_json(source)
    else:
        source = (
            f"mongodb:{args.mongo_db}.annotations -> fragments -> "
            "cropped_sign_images"
        )
        items = fetch_mongo_sign_images(
            args.sign,
            centroids_only=centroids_only,
            include_unclustered=args.include_unclustered,
            env_file=args.env_file,
            mongo_uri_env=args.mongo_uri_env,
            mongo_db=args.mongo_db,
        )

    if args.save_json:
        args.save_json.write_text(
            json.dumps(items, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    rendered_items = items[: args.limit] if args.limit > 0 else items
    output.write_text(render_html(args.sign, source, rendered_items), encoding="utf-8")

    print(f"Fetched {len(items)} records from {source}")
    print(f"Rendered {len(rendered_items)} records to {output}")


if __name__ == "__main__":
    main()
