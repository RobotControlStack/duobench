#!/usr/bin/env python3
"""Scale MuJoCo size values and local positional offsets in gray_microwave.xml."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_INPUT = Path("assets/objects/spring_door/gray_microwave_original.xml")
DEFAULT_OUTPUT = Path("assets/objects/spring_door/gray_microwave.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse gray_microwave.xml, scale local geometry values, and write "
            "the output XML."
        )
    )
    parser.add_argument(
        "scale_float",
        type=float,
        help="Scale factor applied to every number in each `size` attribute.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input XML path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output XML path. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def scale_vector_value(vector_value: str, scale_float: float) -> str:
    scaled = [f"{float(part) * scale_float:.9g}" for part in vector_value.split()]
    return " ".join(scaled)


def main() -> None:
    args = parse_args()

    tree = ET.parse(args.input)
    root = tree.getroot()

    pos_tags = {"body", "geom", "joint", "inertial", "site", "camera", "light"}

    for element in root.iter():
        size_value = element.get("size")
        if size_value is not None:
            element.set("size", scale_vector_value(size_value, args.scale_float))

        pos_value = element.get("pos")
        if pos_value is not None and element.tag in pos_tags:
            element.set("pos", scale_vector_value(pos_value, args.scale_float))

    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=False)


if __name__ == "__main__":
    main()
