use std::fs;
use std::path::Path;

fn read_u32(data: &[u8], offset: usize) -> Result<usize, String> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| "truncated IDX header".to_string())?;
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()) as usize)
}

pub struct IdxImages {
    data: Vec<u8>,
    count: usize,
    rows: usize,
    columns: usize,
}

impl IdxImages {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let data = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if read_u32(&data, 0)? != 2051 {
            return Err("invalid IDX image magic".to_string());
        }
        let count = read_u32(&data, 4)?;
        let rows = read_u32(&data, 8)?;
        let columns = read_u32(&data, 12)?;
        let expected = 16usize
            .checked_add(
                count
                    .checked_mul(rows * columns)
                    .ok_or_else(|| "IDX image size overflow".to_string())?,
            )
            .ok_or_else(|| "IDX image size overflow".to_string())?;
        if data.len() != expected {
            return Err("invalid IDX image length".to_string());
        }
        Ok(Self {
            data,
            count,
            rows,
            columns,
        })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn image(&self, index: usize) -> Result<&[u8], String> {
        if index >= self.count {
            return Err(format!("image index out of range: {index}"));
        }
        let size = self.rows * self.columns;
        let start = 16 + index * size;
        Ok(&self.data[start..start + size])
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
}

pub struct IdxLabels {
    data: Vec<u8>,
    count: usize,
}

impl IdxLabels {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let data = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if read_u32(&data, 0)? != 2049 {
            return Err("invalid IDX label magic".to_string());
        }
        let count = read_u32(&data, 4)?;
        if data.len() != count + 8 {
            return Err("invalid IDX label length".to_string());
        }
        Ok(Self { data, count })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn label(&self, index: usize) -> Result<u8, String> {
        if index >= self.count {
            return Err(format!("label index out of range: {index}"));
        }
        Ok(self.data[8 + index])
    }
}
