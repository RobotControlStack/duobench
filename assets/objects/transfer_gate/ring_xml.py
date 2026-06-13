#!/usr/bin/env python3
"""Print MuJoCo box geom tags arranged in a circular ring."""

from __future__ import annotations

import argparse
import math


def fmt(value: float) -> str:
    """Format floats compactly while keeping enough precision for XML use."""
    if abs(value) < 0.5e-6:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_geom_tag(
    index: int,
    angle: float,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_height: float,
    z: float,
    name_prefix: str,
    geom_class: str,
    name_suffix: str,
    rgba: str,
    friction: str,
    density: float | None,
) -> str:
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    attrs = [
        'type="box"',
        f'class="{geom_class}"',
        f'pos="{fmt(x)} {fmt(y)} {fmt(z)}"',
        f'euler="0 0 {fmt(angle)}"',
        f'size="{fmt(half_thickness)} {fmt(half_length)} {fmt(half_height)}"',
    ]

    if rgba:
        attrs.append(f'rgba="{rgba}"')
    if friction:
        attrs.append(f'friction="{friction}"')
    if density is not None:
        attrs.append(f'density="{fmt(density)}"')

    return f"<geom {' '.join(attrs)} />"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate box geom XML tags for a ring made of tangent boxes."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.15,
        help="Centerline radius of the ring.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=8,
        help="Number of box segments around the ring.",
    )
    parser.add_argument(
        "--half-thickness",
        type=float,
        default=0.01,
        help="Radial half-thickness of each box segment.",
    )
    parser.add_argument(
        "--half-height",
        type=float,
        default=0.01,
        help="Vertical half-height of each box segment.",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=0.0,
        help="Z position for all generated box geoms.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=1.02,
        help="Tangential length multiplier to avoid tiny gaps between segments.",
    )
    parser.add_argument(
        "--name-prefix",
        default="ring_segment",
        help="Prefix used for generated geom names.",
    )
    parser.add_argument(
        "--visual-class",
        default="visual",
        help="Class assigned to visual geom copies.",
    )
    parser.add_argument(
        "--collision-class",
        default="collision",
        help="Class assigned to collision geom copies.",
    )
    parser.add_argument(
        "--rgba",
        default="0 0 1 1",
        help='RGBA value copied into each visual geom, for example "0 0 1 1".',
    )
    parser.add_argument(
        "--friction",
        default="1 0.3 0.1",
        help='Friction value copied into each collision geom, for example "1 0.3 0.1".',
    )
    parser.add_argument(
        "--density",
        type=float,
        default=50.0,
        help="Density value copied into each collision geom.",
    )
    parser.add_argument(
        "--indent",
        default="        ",
        help="Indent placed before each generated tag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.radius <= 0:
        raise SystemExit("--radius must be greater than 0")
    if args.segments < 3:
        raise SystemExit("--segments must be at least 3")
    if args.half_thickness <= 0:
        raise SystemExit("--half-thickness must be greater than 0")
    if args.half_height <= 0:
        raise SystemExit("--half-height must be greater than 0")
    if args.overlap <= 0:
        raise SystemExit("--overlap must be greater than 0")

    angle_step = math.tau / args.segments
    outer_radius = args.radius + args.half_thickness
    half_length = outer_radius * math.tan(angle_step / 2.0) * args.overlap

    for index in range(args.segments):
        angle = index * angle_step
        visual_tag = build_geom_tag(
            index=index,
            angle=angle,
            radius=args.radius,
            half_thickness=args.half_thickness,
            half_length=half_length,
            half_height=args.half_height,
            z=args.z,
            name_prefix=args.name_prefix,
            geom_class=args.visual_class,
            name_suffix="visual",
            rgba=args.rgba,
            friction="",
            density=None,
        )
        collision_tag = build_geom_tag(
            index=index,
            angle=angle,
            radius=args.radius,
            half_thickness=args.half_thickness,
            half_length=half_length,
            half_height=args.half_height,
            z=args.z,
            name_prefix=args.name_prefix,
            geom_class=args.collision_class,
            name_suffix="collision",
            rgba="",
            friction=args.friction,
            density=args.density,
        )
        print(f"{args.indent}{visual_tag}")
        print(f"{args.indent}{collision_tag}")


if __name__ == "__main__":
    main()
