use std::collections::HashMap;

use ndarray::{Array1, Array2, Array4, Axis, Ix1, Ix2, Ix4};

use crate::Tensor;

const SIDE: usize = 10;
const CELLS: usize = SIDE * SIDE;
const FEATURE_CHANNELS: usize = 18;
const SPATIAL_CHANNELS: usize = 15;
const SEQUENCE_CHANNELS: usize = 3;
const CHANNELS: usize = 144;
const BLOCKS: usize = 9;
const FUSION_UNITS: usize = CHANNELS * 2;

struct Linear {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

struct ResidualBlock {
    depthwise_weight: Array2<f32>,
    depthwise_bias: Array1<f32>,
    pointwise: Linear,
}

/// Fixed AHC015 afterstate residual-value network.
///
/// Input is contiguous NCHW `(N, 18, 10, 10)`. The returned value is the
/// learned residual `G(W)`; callers add the analytic connectedness potential.
pub struct Ahc015ValueNet {
    stem_weight: Array4<f32>,
    stem_bias: Array1<f32>,
    blocks: Vec<ResidualBlock>,
    future_fc1: Linear,
    future_fc2: Linear,
    film: Option<Linear>,
    fusion_fc: Linear,
    output: Linear,
}

fn take_tensor(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    expected_rank: usize,
) -> Result<Tensor, String> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| format!("missing tensor: {name}"))?;
    if tensor.values.ndim() != expected_rank {
        return Err(format!(
            "invalid rank for tensor {name}: {:?}",
            tensor.values.shape()
        ));
    }
    Ok(tensor)
}

fn take_bias(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    size: usize,
) -> Result<Array1<f32>, String> {
    let bias = take_tensor(tensors, name, 1)?
        .values
        .into_dimensionality::<Ix1>()
        .map_err(|error| format!("invalid bias {name}: {error}"))?;
    if bias.len() != size {
        return Err(format!("invalid shape for {name}: {:?}", bias.shape()));
    }
    Ok(bias)
}

fn take_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    output_size: usize,
    input_size: usize,
) -> Result<Linear, String> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = take_tensor(tensors, &weight_name, 2)?
        .values
        .into_dimensionality::<Ix2>()
        .map_err(|error| format!("invalid linear weight {weight_name}: {error}"))?;
    if weight.dim() != (output_size, input_size) {
        return Err(format!(
            "invalid shape for {weight_name}: {:?}, expected ({output_size}, {input_size})",
            weight.dim()
        ));
    }
    let bias = take_bias(tensors, &bias_name, output_size)?;
    Ok(Linear { weight, bias })
}

impl Ahc015ValueNet {
    pub const PARAMETER_COUNT: usize = 409_969;
    pub const INPUT_LENGTH: usize = FEATURE_CHANNELS * CELLS;

    pub fn from_tensors(mut tensors: HashMap<String, Tensor>) -> Result<Self, String> {
        let stem_weight = take_tensor(&mut tensors, "board_stem.weight", 4)?
            .values
            .into_dimensionality::<Ix4>()
            .map_err(|error| format!("invalid board_stem.weight: {error}"))?;
        if stem_weight.dim() != (CHANNELS, SPATIAL_CHANNELS, 3, 3) {
            return Err(format!(
                "invalid shape for board_stem.weight: {:?}",
                stem_weight.dim()
            ));
        }
        let stem_bias = take_bias(&mut tensors, "board_stem.bias", CHANNELS)?;

        let mut blocks = Vec::with_capacity(BLOCKS);
        for block in 0..BLOCKS {
            let depthwise_name = format!("blocks.{block}.depthwise.weight");
            let depthwise = take_tensor(&mut tensors, &depthwise_name, 4)?
                .values
                .into_dimensionality::<Ix4>()
                .map_err(|error| format!("invalid {depthwise_name}: {error}"))?;
            if depthwise.dim() != (CHANNELS, 1, 3, 3) {
                return Err(format!(
                    "invalid shape for {depthwise_name}: {:?}",
                    depthwise.dim()
                ));
            }
            let depthwise_weight = depthwise
                .into_shape_with_order((CHANNELS, 9))
                .map_err(|error| format!("invalid {depthwise_name}: {error}"))?;
            let depthwise_bias = take_bias(
                &mut tensors,
                &format!("blocks.{block}.depthwise.bias"),
                CHANNELS,
            )?;

            let pointwise_name = format!("blocks.{block}.pointwise.weight");
            let pointwise_weight = take_tensor(&mut tensors, &pointwise_name, 4)?
                .values
                .into_dimensionality::<Ix4>()
                .map_err(|error| format!("invalid {pointwise_name}: {error}"))?;
            if pointwise_weight.dim() != (CHANNELS, CHANNELS, 1, 1) {
                return Err(format!(
                    "invalid shape for {pointwise_name}: {:?}",
                    pointwise_weight.dim()
                ));
            }
            let pointwise = Linear {
                weight: pointwise_weight
                    .into_shape_with_order((CHANNELS, CHANNELS))
                    .map_err(|error| format!("invalid {pointwise_name}: {error}"))?,
                bias: take_bias(
                    &mut tensors,
                    &format!("blocks.{block}.pointwise.bias"),
                    CHANNELS,
                )?,
            };
            blocks.push(ResidualBlock {
                depthwise_weight,
                depthwise_bias,
                pointwise,
            });
        }

        let future_fc1 = take_linear(
            &mut tensors,
            "future_fc1",
            CHANNELS,
            SEQUENCE_CHANNELS * CELLS,
        )?;
        let future_fc2 = take_linear(&mut tensors, "future_fc2", CHANNELS, CHANNELS)?;
        let film = if tensors.contains_key("film.weight") {
            Some(take_linear(&mut tensors, "film", CHANNELS * 2, CHANNELS)?)
        } else {
            None
        };
        let fusion_fc = take_linear(&mut tensors, "fusion_fc", FUSION_UNITS, CHANNELS * 2)?;
        let output = take_linear(&mut tensors, "output", 1, FUSION_UNITS)?;

        if !tensors.is_empty() {
            let mut names = tensors.keys().cloned().collect::<Vec<_>>();
            names.sort();
            return Err(format!("unexpected tensors: {}", names.join(", ")));
        }

        Ok(Self {
            stem_weight,
            stem_bias,
            blocks,
            future_fc1,
            future_fc2,
            film,
            fusion_fc,
            output,
        })
    }

    /// Runs a batch. `input` must be contiguous NCHW data.
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.is_empty() || !input.len().is_multiple_of(Self::INPUT_LENGTH) {
            return Err(format!(
                "AHC015 input length must be a positive multiple of {}, got {}",
                Self::INPUT_LENGTH,
                input.len()
            ));
        }
        let batch = input.len() / Self::INPUT_LENGTH;
        let rows = batch * CELLS;

        // im2col gives the stem one matrix multiplication for the whole batch.
        let mut columns = Array2::<f32>::zeros((rows, SPATIAL_CHANNELS * 9));
        for sample in 0..batch {
            let sample_base = sample * Self::INPUT_LENGTH;
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let row = sample * CELLS + y * SIDE + x;
                    for channel in 0..SPATIAL_CHANNELS {
                        let channel_base = sample_base + channel * CELLS;
                        for kernel_y in 0..3 {
                            let input_y = y as isize + kernel_y as isize - 1;
                            if !(0..SIDE as isize).contains(&input_y) {
                                continue;
                            }
                            for kernel_x in 0..3 {
                                let input_x = x as isize + kernel_x as isize - 1;
                                if (0..SIDE as isize).contains(&input_x) {
                                    let column = channel * 9 + kernel_y * 3 + kernel_x;
                                    columns[(row, column)] = input
                                        [channel_base + input_y as usize * SIDE + input_x as usize];
                                }
                            }
                        }
                    }
                }
            }
        }
        let stem_matrix = self
            .stem_weight
            .view()
            .into_shape_with_order((CHANNELS, SPATIAL_CHANNELS * 9))
            .expect("validated stem shape");
        let mut activation = columns.dot(&stem_matrix.t());
        add_bias_relu(&mut activation, &self.stem_bias);

        let mut future = Array2::<f32>::zeros((batch, SEQUENCE_CHANNELS * CELLS));
        for sample in 0..batch {
            let sample_base = sample * Self::INPUT_LENGTH;
            for channel in 0..SEQUENCE_CHANNELS {
                let source = sample_base + (SPATIAL_CHANNELS + channel) * CELLS;
                let destination = channel * CELLS;
                for cell in 0..CELLS {
                    future[(sample, destination + cell)] = input[source + cell];
                }
            }
        }
        let future = linear_relu(future, &self.future_fc1);
        let future = linear_relu(future, &self.future_fc2);
        if let Some(film_layer) = &self.film {
            let mut film = future.dot(&film_layer.weight.t());
            add_bias(&mut film, &film_layer.bias);
            for sample in 0..batch {
                for cell in 0..CELLS {
                    let row = sample * CELLS + cell;
                    for channel in 0..CHANNELS {
                        let gamma = film[(sample, channel)];
                        let beta = film[(sample, CHANNELS + channel)];
                        activation[(row, channel)] =
                            activation[(row, channel)] * (1.0 + gamma) + beta;
                    }
                }
            }
        }

        for block in &self.blocks {
            let mut depthwise = Array2::<f32>::zeros((rows, CHANNELS));
            for sample in 0..batch {
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let row = sample * CELLS + y * SIDE + x;
                        for channel in 0..CHANNELS {
                            let mut value = block.depthwise_bias[channel];
                            for kernel_y in 0..3 {
                                let input_y = y as isize + kernel_y as isize - 1;
                                if !(0..SIDE as isize).contains(&input_y) {
                                    continue;
                                }
                                for kernel_x in 0..3 {
                                    let input_x = x as isize + kernel_x as isize - 1;
                                    if (0..SIDE as isize).contains(&input_x) {
                                        let input_row = sample * CELLS
                                            + input_y as usize * SIDE
                                            + input_x as usize;
                                        value += activation[(input_row, channel)]
                                            * block.depthwise_weight
                                                [(channel, kernel_y * 3 + kernel_x)];
                                    }
                                }
                            }
                            depthwise[(row, channel)] = value.max(0.0);
                        }
                    }
                }
            }
            let mut projected = depthwise.dot(&block.pointwise.weight.t());
            for (mut output_row, input_row) in projected
                .axis_iter_mut(Axis(0))
                .zip(activation.axis_iter(Axis(0)))
            {
                for channel in 0..CHANNELS {
                    output_row[channel] =
                        (output_row[channel] + block.pointwise.bias[channel] + input_row[channel])
                            .max(0.0);
                }
            }
            activation = projected;
        }

        let mut board_embedding = Array2::<f32>::zeros((batch, CHANNELS));
        for sample in 0..batch {
            for cell in 0..CELLS {
                let row = sample * CELLS + cell;
                for channel in 0..CHANNELS {
                    board_embedding[(sample, channel)] += activation[(row, channel)];
                }
            }
        }
        board_embedding.mapv_inplace(|value| value / CELLS as f32);

        let mut fusion = Array2::<f32>::zeros((batch, CHANNELS * 2));
        for sample in 0..batch {
            for channel in 0..CHANNELS {
                fusion[(sample, channel)] = board_embedding[(sample, channel)];
                fusion[(sample, CHANNELS + channel)] = future[(sample, channel)];
            }
        }
        let fusion = linear_relu(fusion, &self.fusion_fc);
        let mut output = fusion.dot(&self.output.weight.t());
        for mut row in output.axis_iter_mut(Axis(0)) {
            row[0] += self.output.bias[0];
        }
        Ok(output.column(0).to_vec())
    }
}

fn add_bias_relu(values: &mut Array2<f32>, bias: &Array1<f32>) {
    add_bias(values, bias);
    values.mapv_inplace(|value| value.max(0.0));
}

fn add_bias(values: &mut Array2<f32>, bias: &Array1<f32>) {
    for mut row in values.axis_iter_mut(Axis(0)) {
        for index in 0..bias.len() {
            row[index] += bias[index];
        }
    }
}

fn linear_relu(input: Array2<f32>, layer: &Linear) -> Array2<f32> {
    let mut output = input.dot(&layer.weight.t());
    add_bias_relu(&mut output, &layer.bias);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn tensor(shape: &[usize], value: f32) -> Tensor {
        Tensor {
            values: ArrayD::from_elem(IxDyn(shape), value),
        }
    }

    fn zero_tensors() -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "board_stem.weight".to_string(),
            tensor(&[CHANNELS, SPATIAL_CHANNELS, 3, 3], 0.0),
        );
        tensors.insert("board_stem.bias".to_string(), tensor(&[CHANNELS], 0.0));
        for block in 0..BLOCKS {
            tensors.insert(
                format!("blocks.{block}.depthwise.weight"),
                tensor(&[CHANNELS, 1, 3, 3], 0.0),
            );
            tensors.insert(
                format!("blocks.{block}.depthwise.bias"),
                tensor(&[CHANNELS], 0.0),
            );
            tensors.insert(
                format!("blocks.{block}.pointwise.weight"),
                tensor(&[CHANNELS, CHANNELS, 1, 1], 0.0),
            );
            tensors.insert(
                format!("blocks.{block}.pointwise.bias"),
                tensor(&[CHANNELS], 0.0),
            );
        }
        tensors.insert(
            "future_fc1.weight".to_string(),
            tensor(&[CHANNELS, SEQUENCE_CHANNELS * CELLS], 0.0),
        );
        tensors.insert("future_fc1.bias".to_string(), tensor(&[CHANNELS], 0.0));
        tensors.insert(
            "future_fc2.weight".to_string(),
            tensor(&[CHANNELS, CHANNELS], 0.0),
        );
        tensors.insert("future_fc2.bias".to_string(), tensor(&[CHANNELS], 0.0));
        tensors.insert(
            "fusion_fc.weight".to_string(),
            tensor(&[FUSION_UNITS, CHANNELS * 2], 0.0),
        );
        tensors.insert("fusion_fc.bias".to_string(), tensor(&[FUSION_UNITS], 0.0));
        tensors.insert("output.weight".to_string(), tensor(&[1, FUSION_UNITS], 0.0));
        tensors.insert("output.bias".to_string(), tensor(&[1], 0.0));
        tensors
    }

    #[test]
    fn standard_parameter_count_is_stable() {
        let count = CHANNELS * SPATIAL_CHANNELS * 9
            + CHANNELS
            + BLOCKS * (CHANNELS * 9 + CHANNELS + CHANNELS * CHANNELS + CHANNELS)
            + CHANNELS * SEQUENCE_CHANNELS * CELLS
            + CHANNELS
            + CHANNELS * CHANNELS
            + CHANNELS
            + CHANNELS * 2 * CHANNELS
            + CHANNELS * 2
            + FUSION_UNITS * CHANNELS * 2
            + FUSION_UNITS
            + FUSION_UNITS
            + 1;
        assert_eq!(count, Ahc015ValueNet::PARAMETER_COUNT);
    }

    #[test]
    fn zero_model_returns_zero_for_a_batch() {
        let model = Ahc015ValueNet::from_tensors(zero_tensors()).unwrap();
        let output = model
            .predict(&vec![0.0; 2 * Ahc015ValueNet::INPUT_LENGTH])
            .unwrap();
        assert_eq!(output, vec![0.0, 0.0]);
    }
}
