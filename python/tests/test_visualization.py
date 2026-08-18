from __future__ import annotations

import shutil

import pytest
from ahc_ml.visualization import render_model_graph
from torch import nn


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz is not installed")
def test_render_model_graph_does_not_move_the_training_model_to_meta(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(4, 2), nn.ReLU())

    svg_path, png_path = render_model_graph(
        model,
        input_size=(1, 4),
        output_stem=tmp_path / "model-graph",
    )

    assert svg_path.is_file()
    assert png_path.is_file()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
