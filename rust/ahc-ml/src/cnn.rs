use std::collections::HashMap;

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView3, Ix1, Ix2, Ix4};

use crate::Tensor;

struct Conv2d {
    weight: Array4<f32>,
    bias: Array1<f32>,
}

struct Linear {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

pub struct MnistCnn {
    conv1: Conv2d,
    conv2: Conv2d,
    linear1: Linear,
    linear2: Linear,
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

fn make_conv(
    tensors: &mut HashMap<String, Tensor>,
    weight_name: &str,
    bias_name: &str,
) -> Result<Conv2d, String> {
    let weight = take_tensor(tensors, weight_name, 4)?
        .values
        .into_dimensionality::<Ix4>()
        .map_err(|error| format!("invalid convolution weight {weight_name}: {error}"))?;
    let bias = take_tensor(tensors, bias_name, 1)?
        .values
        .into_dimensionality::<Ix1>()
        .map_err(|error| format!("invalid convolution bias {bias_name}: {error}"))?;
    let (output_channels, _, kernel_height, kernel_width) = weight.dim();
    if (kernel_height, kernel_width) != (3, 3) || bias.len() != output_channels {
        return Err(format!("invalid convolution shape: {weight_name}"));
    }
    Ok(Conv2d { weight, bias })
}

fn make_linear(
    tensors: &mut HashMap<String, Tensor>,
    weight_name: &str,
    bias_name: &str,
) -> Result<Linear, String> {
    let weight = take_tensor(tensors, weight_name, 2)?
        .values
        .into_dimensionality::<Ix2>()
        .map_err(|error| format!("invalid linear weight {weight_name}: {error}"))?;
    let bias = take_tensor(tensors, bias_name, 1)?
        .values
        .into_dimensionality::<Ix1>()
        .map_err(|error| format!("invalid linear bias {bias_name}: {error}"))?;
    if bias.len() != weight.nrows() {
        return Err(format!("invalid linear shape: {weight_name}"));
    }
    Ok(Linear { weight, bias })
}

impl MnistCnn {
    pub fn from_tensors(mut tensors: HashMap<String, Tensor>) -> Result<Self, String> {
        let conv1 = make_conv(&mut tensors, "features.0.weight", "features.0.bias")?;
        let conv2 = make_conv(&mut tensors, "features.3.weight", "features.3.bias")?;
        let linear1 = make_linear(&mut tensors, "classifier.1.weight", "classifier.1.bias")?;
        let linear2 = make_linear(&mut tensors, "classifier.3.weight", "classifier.3.bias")?;
        if conv1.weight.dim().1 != 1
            || conv2.weight.dim().1 != conv1.weight.dim().0
            || linear1.weight.ncols() != conv2.weight.dim().0 * 7 * 7
            || linear2.weight.ncols() != linear1.weight.nrows()
            || linear2.weight.nrows() != 10
        {
            return Err("incompatible mnist-cnn-v1 tensor shapes".to_string());
        }
        if !tensors.is_empty() {
            return Err(format!(
                "unexpected tensors: {}",
                tensors.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        Ok(Self {
            conv1,
            conv2,
            linear1,
            linear2,
        })
    }

    pub fn predict_u8(&self, pixels: &[u8]) -> Result<[f32; 10], String> {
        if pixels.len() != 28 * 28 {
            return Err(format!("expected 784 pixels, got {}", pixels.len()));
        }
        let input =
            Array3::from_shape_fn((1, 28, 28), |(_, y, x)| pixels[y * 28 + x] as f32 / 255.0);
        Ok(self.forward_array(input.view()))
    }

    pub fn forward(&self, input: &[f32]) -> [f32; 10] {
        let input = ArrayView3::from_shape((1, 28, 28), input)
            .expect("MNIST input must contain exactly 784 values");
        self.forward_array(input)
    }

    fn forward_array(&self, input: ArrayView3<'_, f32>) -> [f32; 10] {
        let mut features = conv3x3_pad1(input, &self.conv1);
        relu(&mut features);
        features = max_pool2x2(features.view());
        features = conv3x3_pad1(features.view(), &self.conv2);
        relu(&mut features);
        features = max_pool2x2(features.view());
        let flattened = features
            .into_shape_with_order((self.linear1.weight.ncols(),))
            .expect("pooled feature shape must match the first linear layer");
        let mut hidden = linear(flattened.view(), &self.linear1);
        relu(&mut hidden);
        linear(hidden.view(), &self.linear2)
            .to_vec()
            .try_into()
            .unwrap()
    }
}

fn conv3x3_pad1(input: ArrayView3<'_, f32>, layer: &Conv2d) -> Array3<f32> {
    let (input_channels, height, width) = input.dim();
    let (output_channels, weight_input_channels, _, _) = layer.weight.dim();
    assert_eq!(input_channels, weight_input_channels);
    let input_values = input.as_slice().expect("CNN input must be contiguous");
    let weight_values = layer
        .weight
        .as_slice()
        .expect("convolution weights must be contiguous");
    let mut output = Array3::zeros((output_channels, height, width));
    let output_values = output.as_slice_mut().unwrap();

    for output_channel in 0..output_channels {
        let output_plane = output_channel * height * width;
        output_values[output_plane..output_plane + height * width].fill(layer.bias[output_channel]);
        for input_channel in 0..input_channels {
            let input_plane = input_channel * height * width;
            let weight_plane = (output_channel * input_channels + input_channel) * 9;
            for y in 0..height {
                let kernel_y_start = usize::from(y == 0);
                let kernel_y_end = if y + 1 == height { 2 } else { 3 };
                for x in 0..width {
                    let kernel_x_start = usize::from(x == 0);
                    let kernel_x_end = if x + 1 == width { 2 } else { 3 };
                    let mut sum = 0.0;
                    for kernel_y in kernel_y_start..kernel_y_end {
                        let input_y = y + kernel_y - 1;
                        for kernel_x in kernel_x_start..kernel_x_end {
                            let input_x = x + kernel_x - 1;
                            sum += input_values[input_plane + input_y * width + input_x]
                                * weight_values[weight_plane + kernel_y * 3 + kernel_x];
                        }
                    }
                    output_values[output_plane + y * width + x] += sum;
                }
            }
        }
    }
    output
}

fn relu<D: ndarray::Dimension>(values: &mut ndarray::Array<f32, D>) {
    values.mapv_inplace(|value| value.max(0.0));
}

fn max_pool2x2(input: ArrayView3<'_, f32>) -> Array3<f32> {
    let (channels, height, width) = input.dim();
    assert_eq!(height % 2, 0);
    assert_eq!(width % 2, 0);
    let output_height = height / 2;
    let output_width = width / 2;
    let mut output = Array3::zeros((channels, output_height, output_width));
    for channel in 0..channels {
        for y in 0..output_height {
            for x in 0..output_width {
                output[[channel, y, x]] = input[[channel, y * 2, x * 2]]
                    .max(input[[channel, y * 2, x * 2 + 1]])
                    .max(input[[channel, y * 2 + 1, x * 2]])
                    .max(input[[channel, y * 2 + 1, x * 2 + 1]]);
            }
        }
    }
    output
}

fn linear(input: ArrayView1<'_, f32>, layer: &Linear) -> Array1<f32> {
    assert_eq!(input.len(), layer.weight.ncols());
    layer.weight.dot(&input) + &layer.bias
}
