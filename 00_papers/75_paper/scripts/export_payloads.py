#!/usr/bin/env python3
"""Emit sample SCADA payload artifacts for documentation."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_DIR = ROOT / "payloads"
PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLE_JSON = {
    "site_id": "PV-PLANT-08",
    "uav": "M300 RTK",
    "ts_utc": "2025-09-30T10:12:33Z",
    "detections": [
        {
            "id": "clu_012a",
            "class": "hotspot_single",
            "conf": 0.91,
            "temp_C": 82.4,
            "centroid_wgs84": [49.407251, 26.984173],
            "polygon_wgs84": [
                [49.407249, 26.984170],
                [49.407252, 26.984175],
                [49.407254, 26.984172],
            ],
            "media": {
                "rgb": "gs://bucket/vid123_03456.jpg",
                "tiff": "gs://bucket/vid123_03456.tif",
            },
        }
    ],
}

KML_TEMPLATE = dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
      <Document>
        <name>UAV PV Detections</name>
        <Placemark>
          <name>clu_012a: hotspot_single</name>
          <Point>
            <coordinates>26.984173,49.407251,0</coordinates>
          </Point>
        </Placemark>
      </Document>
    </kml>
    """
)


def write_detection_payload() -> Path:
    """Write the example detection JSON payload."""
    output_path = PAYLOAD_DIR / "example_detection.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(EXAMPLE_JSON, file, indent=2)
    return output_path


def write_cluster_kml() -> Path:
    """Write the companion KML for geospatial context."""
    output_path = PAYLOAD_DIR / "example_clusters.kml"
    output_path.write_text(KML_TEMPLATE, encoding="utf-8")
    return output_path


def main() -> None:
    """Emit both JSON and KML payload examples."""
    json_path = write_detection_payload()
    kml_path = write_cluster_kml()
    print(f"Wrote payloads to {json_path.parent}:\n- {json_path.name}\n- {kml_path.name}")


if __name__ == "__main__":
    main()
