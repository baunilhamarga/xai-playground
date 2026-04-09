from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MembershipSpec:
    label: str
    shape: str
    params: tuple[float, ...]


@dataclass(frozen=True)
class AntecedentTerm:
    variable: str
    label: str


@dataclass(frozen=True)
class RuleSpec:
    antecedents: tuple[AntecedentTerm, ...]
    consequent: str
    operator: str = "and"


@dataclass(frozen=True)
class RulebaseConfig:
    name: str
    universes: dict[str, tuple[float, float]]
    input_memberships: dict[str, tuple[MembershipSpec, ...]]
    output_memberships: tuple[MembershipSpec, ...]
    rules: tuple[RuleSpec, ...]


RULEBASES: dict[str, RulebaseConfig] = {
    "heitor1": RulebaseConfig(
        name="heitor1",
        universes={"food": (0.0, 10.0), "service": (0.0, 10.0), "tip": (0.0, 30.0)},
        input_memberships={
            "food": (
                MembershipSpec("rancid", "trap", (0.0, 0.0, 1.0, 3.0)),
                MembershipSpec("okay", "tri", (2.0, 5.0, 8.0)),
                MembershipSpec("delicious", "trap", (7.0, 9.0, 10.0, 10.0)),
            ),
            "service": (
                MembershipSpec("poor", "trap", (0.0, 0.0, 2.0, 4.0)),
                MembershipSpec("good", "tri", (2.0, 5.0, 8.0)),
                MembershipSpec("excellent", "trap", (6.0, 8.0, 10.0, 10.0)),
            ),
        },
        output_memberships=(
            MembershipSpec("cheap", "tri", (0.0, 5.0, 10.0)),
            MembershipSpec("average", "tri", (10.0, 15.0, 20.0)),
            MembershipSpec("generous", "tri", (20.0, 25.0, 30.0)),
        ),
        rules=(
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "poor"),
                    AntecedentTerm("food", "rancid"),
                ),
                consequent="cheap",
                operator="or",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "good"),
                    AntecedentTerm("food", "okay"),
                ),
                consequent="average",
                operator="or",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "excellent"),
                    AntecedentTerm("food", "delicious"),
                ),
                consequent="generous",
                operator="or",
            ),
        ),
    ),
    "heitor2": RulebaseConfig(
        name="heitor2",
        universes={"food": (0.0, 10.0), "service": (0.0, 10.0), "tip": (0.0, 30.0)},
        input_memberships={
            "food": (
                MembershipSpec("rancid", "trap", (0.0, 0.0, 1.0, 3.0)),
                MembershipSpec("okay", "tri", (2.0, 5.0, 8.0)),
                MembershipSpec("delicious", "trap", (7.0, 9.0, 10.0, 10.0)),
            ),
            "service": (
                MembershipSpec("poor", "trap", (0.0, 0.0, 2.0, 4.0)),
                MembershipSpec("good", "tri", (2.0, 5.0, 8.0)),
                MembershipSpec("excellent", "trap", (6.0, 8.0, 10.0, 10.0)),
            ),
        },
        output_memberships=(
            MembershipSpec("cheap", "tri", (0.0, 5.0, 10.0)),
            MembershipSpec("average", "tri", (10.0, 15.0, 20.0)),
            MembershipSpec("generous", "tri", (20.0, 25.0, 30.0)),
        ),
        rules=(
            RuleSpec(
                antecedents=(AntecedentTerm("service", "excellent"),),
                consequent="generous",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "poor"),
                    AntecedentTerm("food", "rancid"),
                ),
                consequent="cheap",
                operator="or",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "good"),
                    AntecedentTerm("food", "delicious"),
                ),
                consequent="generous",
                operator="and",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "good"),
                    AntecedentTerm("food", "okay"),
                ),
                consequent="average",
                operator="and",
            ),
        ),
    ),
    "heitor3": RulebaseConfig(
        name="heitor3",
        universes={"food": (0.0, 10.0), "service": (0.0, 10.0), "tip": (0.0, 30.0)},
        input_memberships={
            "food": (
                MembershipSpec("rancid", "trap", (0.0, 0.0, 4.0, 6.0)),
                MembershipSpec("delicious", "trap", (4.0, 8.0, 10.0, 10.0)),
            ),
            "service": (
                MembershipSpec("poor", "trap", (0.0, 0.0, 3.0, 4.5)),
                MembershipSpec("good", "trap", (2.5, 4.0, 6.0, 7.5)),
                MembershipSpec("excellent", "trap", (5.5, 7.0, 10.0, 10.0)),
            ),
        },
        output_memberships=(
            MembershipSpec("cheap", "tri", (0.0, 10.0, 20.0)),
            MembershipSpec("average", "tri", (5.0, 15.0, 25.0)),
            MembershipSpec("generous", "tri", (10.0, 20.0, 30.0)),
        ),
        rules=(
            RuleSpec(
                antecedents=(AntecedentTerm("service", "excellent"),),
                consequent="generous",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "poor"),
                    AntecedentTerm("food", "delicious"),
                ),
                consequent="average",
                operator="and",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "good"),
                    AntecedentTerm("food", "delicious"),
                ),
                consequent="average",
                operator="and",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "good"),
                    AntecedentTerm("food", "rancid"),
                ),
                consequent="cheap",
                operator="and",
            ),
            RuleSpec(
                antecedents=(
                    AntecedentTerm("service", "poor"),
                    AntecedentTerm("food", "rancid"),
                ),
                consequent="cheap",
                operator="and",
            ),
        ),
    ),
}


def trimf(x: np.ndarray | Iterable[float], a: float, b: float, c: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)

    if b != a:
        idx = (a < x) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)

    y[x == b] = 1.0

    if c != b:
        idx = (b < x) & (x < c)
        y[idx] = (c - x[idx]) / (c - b)

    return np.clip(y, 0.0, 1.0)


def trapmf(
    x: np.ndarray | Iterable[float],
    a: float,
    b: float,
    c: float,
    d: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)

    if b != a:
        idx = (a < x) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)

    idx = (b <= x) & (x <= c)
    y[idx] = 1.0

    if d != c:
        idx = (c < x) & (x < d)
        y[idx] = (d - x[idx]) / (d - c)

    return np.clip(y, 0.0, 1.0)


def evaluate_membership(spec: MembershipSpec, x: np.ndarray | Iterable[float]) -> np.ndarray:
    if spec.shape == "tri":
        return trimf(x, *spec.params)
    if spec.shape == "trap":
        return trapmf(x, *spec.params)
    raise ValueError(f"Unsupported membership shape: {spec.shape}")


def centroid(x: np.ndarray, mu: np.ndarray) -> float:
    area = np.trapezoid(mu, x)
    if area <= 1e-12:
        return float(np.mean(x))
    return float(np.trapezoid(x * mu, x) / area)


def scalar_membership(spec: MembershipSpec, value: float) -> float:
    return float(evaluate_membership(spec, [value])[0])


def combine_strengths(values: list[float], operator: str) -> float:
    if not values:
        raise ValueError("At least one antecedent is required.")
    if len(values) == 1:
        return values[0]
    if operator == "and":
        return min(values)
    if operator == "or":
        return max(values)
    raise ValueError(f"Unsupported rule operator: {operator}")


def spec_lookup(config: RulebaseConfig) -> dict[str, dict[str, MembershipSpec]]:
    return {
        variable: {spec.label: spec for spec in specs}
        for variable, specs in config.input_memberships.items()
    }


def output_lookup(config: RulebaseConfig) -> dict[str, MembershipSpec]:
    return {spec.label: spec for spec in config.output_memberships}


def format_params(params: tuple[float, ...]) -> str:
    values = []
    for value in params:
        if float(value).is_integer():
            values.append(str(int(value)))
        else:
            values.append(f"{value:g}")
    return "[" + ", ".join(values) + "]"


def membership_label(spec: MembershipSpec) -> str:
    return f"{spec.label} ({spec.shape} {format_params(spec.params)})"


def evaluate_tip(
    config: RulebaseConfig,
    service_value: float,
    food_value: float,
    output_x: np.ndarray,
    input_specs: dict[str, dict[str, MembershipSpec]],
    output_specs: dict[str, MembershipSpec],
) -> float:
    aggregated_outputs = []
    inputs = {"service": service_value, "food": food_value}

    for rule in config.rules:
        strengths = [
            scalar_membership(input_specs[term.variable][term.label], inputs[term.variable])
            for term in rule.antecedents
        ]
        firing = combine_strengths(strengths, rule.operator)
        output_shape = evaluate_membership(output_specs[rule.consequent], output_x)
        aggregated_outputs.append(np.minimum(firing, output_shape))

    combined = np.maximum.reduce(aggregated_outputs)
    return centroid(output_x, combined)


def build_surface(
    config: RulebaseConfig,
    surface_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    service_min, service_max = config.universes["service"]
    food_min, food_max = config.universes["food"]
    tip_min, tip_max = config.universes["tip"]

    service_axis = np.linspace(service_min, service_max, surface_points)
    food_axis = np.linspace(food_min, food_max, surface_points)
    tip_axis = np.linspace(tip_min, tip_max, max(surface_points * 15, 301))

    input_specs = spec_lookup(config)
    output_specs = output_lookup(config)

    z = np.zeros((len(food_axis), len(service_axis)))
    for i, food_value in enumerate(food_axis):
        for j, service_value in enumerate(service_axis):
            z[i, j] = evaluate_tip(
                config,
                service_value,
                food_value,
                tip_axis,
                input_specs,
                output_specs,
            )
    return service_axis, food_axis, z


def save_membership_plot(
    x: np.ndarray,
    specs: tuple[MembershipSpec, ...],
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for spec in specs:
        ax.plot(x, evaluate_membership(spec, x), label=membership_label(spec), linewidth=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Membership degree")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, format="jpg")
    plt.close(fig)


def save_surface_plot(
    service_axis: np.ndarray,
    food_axis: np.ndarray,
    z: np.ndarray,
    title: str,
    path: Path,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    service_grid, food_grid = np.meshgrid(service_axis, food_axis)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        food_grid,
        service_grid,
        z,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("food")
    ax.set_ylabel("service")
    ax.set_zlabel("tip")
    ax.invert_xaxis()
    fig.colorbar(surface, ax=ax, shrink=0.6, aspect=10, pad=0.1, label="tip")
    fig.tight_layout()
    fig.savefig(path, dpi=180, format="jpg")
    plt.close(fig)


def save_heatmap(
    service_axis: np.ndarray,
    food_axis: np.ndarray,
    z: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[service_axis.min(), service_axis.max(), food_axis.min(), food_axis.max()],
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("service")
    ax.set_ylabel("food")
    fig.colorbar(image, ax=ax, label="tip")
    fig.tight_layout()
    fig.savefig(path, dpi=180, format="jpg")
    plt.close(fig)


def save_slice_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("tip")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, format="jpg")
    plt.close(fig)


def generate_rulebase_plots(
    config: RulebaseConfig,
    output_root: Path,
    membership_points: int,
    surface_points: int,
    line_points: int,
) -> list[Path]:
    plots_dir = output_root / config.name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    food_x = np.linspace(*config.universes["food"], membership_points)
    service_x = np.linspace(*config.universes["service"], membership_points)
    tip_x = np.linspace(*config.universes["tip"], membership_points * 3)

    written: list[Path] = []

    service_path = plots_dir / "service_memberships.jpg"
    save_membership_plot(
        service_x,
        config.input_memberships["service"],
        f"{config.name}: service membership functions",
        "service score",
        service_path,
    )
    written.append(service_path)

    food_path = plots_dir / "food_memberships.jpg"
    save_membership_plot(
        food_x,
        config.input_memberships["food"],
        f"{config.name}: food membership functions",
        "food score",
        food_path,
    )
    written.append(food_path)

    tip_path = plots_dir / "tip_memberships.jpg"
    save_membership_plot(
        tip_x,
        config.output_memberships,
        f"{config.name}: tip membership functions",
        "tip",
        tip_path,
    )
    written.append(tip_path)

    service_axis, food_axis, z = build_surface(config, surface_points)

    surface_path = plots_dir / "tip_surface.jpg"
    save_surface_plot(service_axis, food_axis, z, f"{config.name}: tip output surface", surface_path)
    written.append(surface_path)

    heatmap_path = plots_dir / "tip_heatmap.jpg"
    save_heatmap(service_axis, food_axis, z, f"{config.name}: tip output heatmap", heatmap_path)
    written.append(heatmap_path)

    line_service = np.linspace(*config.universes["service"], line_points)
    line_food = np.linspace(*config.universes["food"], line_points)
    tip_axis = np.linspace(*config.universes["tip"], max(line_points * 15, 301))
    input_specs = spec_lookup(config)
    output_specs = output_lookup(config)

    food0 = config.universes["food"][0]
    tip_vs_service = np.array(
        [
            evaluate_tip(config, service_value, food0, tip_axis, input_specs, output_specs)
            for service_value in line_service
        ]
    )
    slice_service_path = plots_dir / "slice_food0_tip_vs_service.jpg"
    save_slice_plot(
        line_service,
        tip_vs_service,
        f"{config.name}: food = {food0:g}, tip vs service",
        "service",
        slice_service_path,
    )
    written.append(slice_service_path)

    service0 = config.universes["service"][0]
    tip_vs_food = np.array(
        [
            evaluate_tip(config, service0, food_value, tip_axis, input_specs, output_specs)
            for food_value in line_food
        ]
    )
    slice_food_path = plots_dir / "slice_service0_tip_vs_food.jpg"
    save_slice_plot(
        line_food,
        tip_vs_food,
        f"{config.name}: service = {service0:g}, tip vs food",
        "food",
        slice_food_path,
    )
    written.append(slice_food_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate JPG plots for one or more tip-problem rulebases. "
            "To add a new rulebase, register one new RulebaseConfig in RULEBASES."
        )
    )
    parser.add_argument(
        "--rulebase",
        choices=[*RULEBASES.keys(), "all"],
        default="all",
        help="Rulebase to plot. Default: all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory where <rulebase>/plots will be created.",
    )
    parser.add_argument(
        "--membership-points",
        type=int,
        default=1001,
        help="Number of samples for membership-function plots.",
    )
    parser.add_argument(
        "--surface-points",
        type=int,
        default=81,
        help="Number of samples per axis for the output surface and heatmap.",
    )
    parser.add_argument(
        "--line-points",
        type=int,
        default=201,
        help="Number of samples for the 1D slice plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = (
        RULEBASES.values()
        if args.rulebase == "all"
        else [RULEBASES[args.rulebase]]
    )

    for config in selected:
        written = generate_rulebase_plots(
            config=config,
            output_root=args.output_root,
            membership_points=args.membership_points,
            surface_points=args.surface_points,
            line_points=args.line_points,
        )
        print(f"{config.name}:")
        for path in written:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
