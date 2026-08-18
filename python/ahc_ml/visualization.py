from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from torch import nn
from torchview import draw_graph


def render_model_graph(
    model: nn.Module,
    *,
    input_size: tuple[int, ...],
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Render a model architecture graph as SVG and PNG files."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    graph = draw_graph(
        deepcopy(model),
        input_size=input_size,
        device="meta",
        depth=5,
        expand_nested=True,
    ).visual_graph

    svg_path = Path(
        graph.render(
            filename=output_stem.name,
            directory=output_stem.parent,
            format="svg",
            cleanup=True,
        )
    )
    png_path = Path(
        graph.render(
            filename=output_stem.name,
            directory=output_stem.parent,
            format="png",
            cleanup=True,
        )
    )
    return svg_path, png_path
