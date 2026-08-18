use std::collections::HashMap;
use std::fs;
use std::path::Path;

use ndarray::{ArrayD, IxDyn};

const FLOAT_MAGIC: &[u8; 8] = b"AHCRLWT1";
const QUANTIZED_MAGIC: &[u8; 8] = b"AHCRLQ81";
const COMPRESSED_MAGIC: &[u8; 8] = b"AHCRLZ01";

#[derive(Debug, Clone)]
pub struct Tensor {
    pub values: ArrayD<f32>,
}

impl Tensor {
    fn from_shape_vec(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, String> {
        let values = ArrayD::from_shape_vec(IxDyn(&shape), values)
            .map_err(|error| format!("invalid tensor shape {shape:?}: {error}"))?;
        Ok(Self { values })
    }
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn take(&mut self, size: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(size)
            .ok_or_else(|| "model offset overflow".to_string())?;
        if end > self.data.len() {
            return Err("truncated model data".to_string());
        }
        let value = &self.data[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8, String> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn finish(self) -> Result<(), String> {
        if self.offset == self.data.len() {
            Ok(())
        } else {
            Err("unexpected trailing model data".to_string())
        }
    }
}

fn read_name(reader: &mut Reader<'_>) -> Result<String, String> {
    let length = reader.u16()? as usize;
    std::str::from_utf8(reader.take(length)?)
        .map(str::to_owned)
        .map_err(|error| format!("invalid tensor name: {error}"))
}

fn read_shape(reader: &mut Reader<'_>, rank: usize) -> Result<(Vec<usize>, usize), String> {
    let mut shape = Vec::with_capacity(rank);
    let mut length = 1usize;
    for _ in 0..rank {
        let dimension = reader.u32()? as usize;
        length = length
            .checked_mul(dimension)
            .ok_or_else(|| "tensor size overflow".to_string())?;
        shape.push(dimension);
    }
    Ok((shape, length))
}

fn insert_tensor(
    tensors: &mut HashMap<String, Tensor>,
    name: String,
    tensor: Tensor,
) -> Result<(), String> {
    if tensors.insert(name.clone(), tensor).is_some() {
        Err(format!("duplicate tensor name: {name}"))
    } else {
        Ok(())
    }
}

fn read_float_model(data: &[u8]) -> Result<HashMap<String, Tensor>, String> {
    let mut reader = Reader::new(data);
    if reader.take(8)? != FLOAT_MAGIC {
        return Err("invalid float model magic".to_string());
    }
    let version = reader.u32()?;
    if version != 1 {
        return Err(format!("unsupported float model version: {version}"));
    }
    let count = reader.u32()? as usize;
    let mut tensors = HashMap::with_capacity(count);
    for _ in 0..count {
        let name = read_name(&mut reader)?;
        let dtype = reader.u8()?;
        if dtype != 1 {
            return Err(format!("unsupported dtype code for {name}: {dtype}"));
        }
        let rank = reader.u8()? as usize;
        let (shape, length) = read_shape(&mut reader, rank)?;
        let byte_length =
            usize::try_from(reader.u64()?).map_err(|_| format!("tensor is too large: {name}"))?;
        if byte_length != length * 4 {
            return Err(format!("invalid byte length for tensor: {name}"));
        }
        let raw = reader.take(byte_length)?;
        let values = raw
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        insert_tensor(&mut tensors, name, Tensor::from_shape_vec(shape, values)?)?;
    }
    reader.finish()?;
    Ok(tensors)
}

fn decompress_huffman(
    data: &[u8],
    lengths: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>, String> {
    if lengths.len() != 256 || lengths.iter().copied().max().unwrap_or(0) > 32 {
        return Err("invalid Huffman code lengths".to_string());
    }
    let mut entries: Vec<(u8, u8)> = lengths
        .iter()
        .enumerate()
        .filter_map(|(symbol, &length)| (length != 0).then_some((length, symbol as u8)))
        .collect();
    entries.sort_unstable();
    let mut codes = HashMap::with_capacity(entries.len());
    let mut code = 0u32;
    let mut previous_length = 0u8;
    for (length, symbol) in entries {
        code <<= length - previous_length;
        codes.insert((length, code), symbol);
        code += 1;
        previous_length = length;
    }

    let mut output = Vec::with_capacity(expected_size);
    let mut current_code = 0u32;
    let mut current_length = 0u8;
    for (byte_index, &byte) in data.iter().enumerate() {
        for shift in (0..8).rev() {
            current_code = (current_code << 1) | ((byte >> shift) & 1) as u32;
            current_length += 1;
            if let Some(&symbol) = codes.get(&(current_length, current_code)) {
                output.push(symbol);
                current_code = 0;
                current_length = 0;
                if output.len() == expected_size {
                    if byte_index + 1 != data.len() {
                        return Err("unexpected trailing Huffman data".to_string());
                    }
                    return Ok(output);
                }
            } else if current_length >= 32 {
                return Err("invalid Huffman bitstream".to_string());
            }
        }
    }
    Err("truncated Huffman bitstream".to_string())
}

fn read_quantized_model(data: &[u8]) -> Result<HashMap<String, Tensor>, String> {
    let mut reader = Reader::new(data);
    if reader.take(8)? != QUANTIZED_MAGIC {
        return Err("invalid quantized model magic".to_string());
    }
    let version = reader.u32()?;
    if version != 1 {
        return Err(format!("unsupported quantized model version: {version}"));
    }
    let count = reader.u32()? as usize;
    let mut tensors = HashMap::with_capacity(count);
    for _ in 0..count {
        let name = read_name(&mut reader)?;
        let rank = reader.u8()? as usize;
        let axis = reader.i8()?;
        let (shape, length) = read_shape(&mut reader, rank)?;
        let scale_count = reader.u32()? as usize;
        let mut scales = Vec::with_capacity(scale_count);
        for _ in 0..scale_count {
            scales.push(reader.f32()?);
        }
        let data_length =
            usize::try_from(reader.u64()?).map_err(|_| format!("tensor is too large: {name}"))?;
        if data_length != length {
            return Err(format!("invalid byte length for tensor: {name}"));
        }
        let quantized = reader.take(data_length)?;
        let mut values = Vec::with_capacity(length);
        match axis {
            -1 if scales.len() == 1 => {
                values.extend(
                    quantized
                        .iter()
                        .map(|&value| (value as i8) as f32 * scales[0]),
                );
            }
            0 if !shape.is_empty() && scales.len() == shape[0] => {
                let channel_size = length / shape[0];
                for (index, &value) in quantized.iter().enumerate() {
                    values.push((value as i8) as f32 * scales[index / channel_size]);
                }
            }
            _ => {
                return Err(format!(
                    "invalid quantization parameters for tensor: {name}"
                ));
            }
        }
        insert_tensor(&mut tensors, name, Tensor::from_shape_vec(shape, values)?)?;
    }
    reader.finish()?;
    Ok(tensors)
}

pub fn read_model(data: &[u8]) -> Result<HashMap<String, Tensor>, String> {
    if data.starts_with(FLOAT_MAGIC) {
        return read_float_model(data);
    }
    if !data.starts_with(COMPRESSED_MAGIC) {
        return Err("unknown model format".to_string());
    }
    let mut reader = Reader::new(data);
    reader.take(8)?;
    let version = reader.u32()?;
    if version != 1 {
        return Err(format!("unsupported compression version: {version}"));
    }
    let raw_size = usize::try_from(reader.u64()?)
        .map_err(|_| "uncompressed model is too large".to_string())?;
    let codec = reader.u8()?;
    let payload = reader.take(data.len() - reader.offset)?;
    let raw = match codec {
        0 if payload.len() == raw_size => payload.to_vec(),
        0 => return Err("invalid stored model length".to_string()),
        1 if payload.len() >= 256 => {
            decompress_huffman(&payload[256..], &payload[..256], raw_size)?
        }
        1 => return Err("truncated Huffman header".to_string()),
        _ => return Err(format!("unsupported compression codec: {codec}")),
    };
    read_quantized_model(&raw)
}

pub fn read_model_file(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>, String> {
    let path = path.as_ref();
    let data =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    read_model(&data)
}

pub fn decode_base64(text: &str) -> Result<Vec<u8>, String> {
    let mut output = Vec::with_capacity(text.len() / 4 * 3);
    let mut block = [0u8; 4];
    let mut count = 0usize;
    for byte in text.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
        block[count] = byte;
        count += 1;
        if count == 4 {
            let mut values = [0u8; 4];
            let mut padding = 0;
            for index in 0..4 {
                values[index] = match block[index] {
                    b'A'..=b'Z' => block[index] - b'A',
                    b'a'..=b'z' => block[index] - b'a' + 26,
                    b'0'..=b'9' => block[index] - b'0' + 52,
                    b'+' => 62,
                    b'/' => 63,
                    b'=' if index >= 2 => {
                        padding += 1;
                        0
                    }
                    _ => return Err("invalid base64 model data".to_string()),
                };
            }
            let bits = ((values[0] as u32) << 18)
                | ((values[1] as u32) << 12)
                | ((values[2] as u32) << 6)
                | values[3] as u32;
            output.push((bits >> 16) as u8);
            if padding < 2 {
                output.push((bits >> 8) as u8);
            }
            if padding == 0 {
                output.push(bits as u8);
            }
            count = 0;
        }
    }
    if count != 0 {
        return Err("invalid base64 model data length".to_string());
    }
    Ok(output)
}

pub fn decode_base93(text: &str) -> Result<Vec<u8>, String> {
    const BASE: u32 = 93;
    const MASK_13: u32 = (1 << 13) - 1;
    const THRESHOLD: u32 = BASE * BASE - (1 << 13) - 1;

    fn digit(byte: u8) -> Option<u32> {
        match byte {
            b' '..=b'!' => Some((byte - b' ') as u32),
            b'#'..=b'[' => Some((byte - b'#') as u32 + 2),
            b']'..=b'~' => Some((byte - b']') as u32 + 59),
            _ => None,
        }
    }

    let mut output = Vec::with_capacity(text.len());
    let mut buffer = 0u32;
    let mut bit_count = 0u32;
    let mut first = None;
    for (index, byte) in text.bytes().enumerate() {
        let value = digit(byte)
            .ok_or_else(|| format!("invalid Base93 character at byte {index}: {byte:#04x}"))?;
        if let Some(low) = first.take() {
            let value = low + value * BASE;
            buffer |= value << bit_count;
            bit_count += if value & MASK_13 > THRESHOLD { 13 } else { 14 };
            while bit_count >= 8 {
                output.push(buffer as u8);
                buffer >>= 8;
                bit_count -= 8;
            }
        } else {
            first = Some(value);
        }
    }

    if let Some(value) = first {
        if bit_count == 0 || value >= 1 << (8 - bit_count) {
            return Err("invalid Base93 trailing data".to_string());
        }
        output.push((buffer | value << bit_count) as u8);
    } else if buffer != 0 {
        return Err("invalid Base93 trailing data".to_string());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{decode_base64, decode_base93};

    const BASE93_ALPHABET: &[u8; 93] =
        b" !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~";

    fn encode_base93(data: &[u8]) -> String {
        const THRESHOLD: u32 = 93 * 93 - (1 << 13) - 1;
        let mut output = Vec::new();
        let mut buffer = 0u32;
        let mut bit_count = 0u32;
        for &byte in data {
            buffer |= (byte as u32) << bit_count;
            bit_count += 8;
            if bit_count > 13 {
                let mut value = buffer & ((1 << 13) - 1);
                if value > THRESHOLD {
                    buffer >>= 13;
                    bit_count -= 13;
                } else {
                    value = buffer & ((1 << 14) - 1);
                    buffer >>= 14;
                    bit_count -= 14;
                }
                output.push(BASE93_ALPHABET[(value % 93) as usize]);
                output.push(BASE93_ALPHABET[(value / 93) as usize]);
            }
        }
        if bit_count > 0 {
            output.push(BASE93_ALPHABET[(buffer % 93) as usize]);
            if bit_count > 7 || buffer >= 93 {
                output.push(BASE93_ALPHABET[(buffer / 93) as usize]);
            }
        }
        String::from_utf8(output).unwrap()
    }

    #[test]
    fn decodes_base64() {
        assert_eq!(decode_base64("YWJjAA==").unwrap(), b"abc\0");
    }

    #[test]
    fn decodes_base93() {
        let cases: &[(&str, &[u8])] = &[
            ("", b""),
            ("  ", &[0]),
            ("! ", &[1]),
            ("~ ", &[92]),
            (" !", &[93]),
            ("   ", &[0, 0]),
            ("T}!^g4T ", b"Base93"),
        ];
        for &(encoded, expected) in cases {
            assert_eq!(decode_base93(encoded).unwrap(), expected);
        }
    }

    #[test]
    fn base93_rejects_escaped_and_noncanonical_data() {
        assert!(decode_base93("\\").is_err());
        assert!(decode_base93("\"").is_err());
        assert!(decode_base93(" ").is_err());
    }

    #[test]
    fn base93_round_trips_every_one_and_two_byte_input() {
        for first in 0..=u8::MAX {
            let one = [first];
            assert_eq!(decode_base93(&encode_base93(&one)).unwrap(), one);
            for second in 0..=u8::MAX {
                let two = [first, second];
                assert_eq!(decode_base93(&encode_base93(&two)).unwrap(), two);
            }
        }
        let long: Vec<u8> = (0..4096).map(|index| (index * 73 % 256) as u8).collect();
        assert_eq!(decode_base93(&encode_base93(&long)).unwrap(), long);
    }
}
