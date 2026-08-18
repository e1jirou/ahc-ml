mod generated_model;

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use ahc_ml::{IdxImages, IdxLabels, MnistCnn, decode_base93, read_model, read_model_file};

struct Arguments {
    model: Option<String>,
    embedded: bool,
    images: String,
    labels: Option<String>,
    limit: Option<usize>,
    index: Option<usize>,
}

fn usage() -> &'static str {
    "usage: mnist-inference (--model MODEL | --embedded) --images IDX [--labels IDX] \
     [--limit N | --index N]"
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut model = None;
    let mut embedded = false;
    let mut images = None;
    let mut labels = None;
    let mut limit = None;
    let mut index = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = |arguments: &mut std::iter::Skip<std::env::Args>| {
            arguments
                .next()
                .ok_or_else(|| format!("missing value after {argument}"))
        };
        match argument.as_str() {
            "--model" => model = Some(value(&mut arguments)?),
            "--embedded" => embedded = true,
            "--images" => images = Some(value(&mut arguments)?),
            "--labels" => labels = Some(value(&mut arguments)?),
            "--limit" => {
                limit = Some(
                    value(&mut arguments)?
                        .parse()
                        .map_err(|_| "--limit must be an integer".to_string())?,
                )
            }
            "--index" => {
                index = Some(
                    value(&mut arguments)?
                        .parse()
                        .map_err(|_| "--index must be an integer".to_string())?,
                )
            }
            "-h" | "--help" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument: {argument}\n{}", usage())),
        }
    }
    if model.is_some() == embedded {
        return Err(format!(
            "choose exactly one of --model and --embedded\n{}",
            usage()
        ));
    }
    if limit.is_some() && index.is_some() {
        return Err("--limit and --index cannot be used together".to_string());
    }
    Ok(Arguments {
        model,
        embedded,
        images: images.ok_or_else(|| format!("--images is required\n{}", usage()))?,
        labels,
        limit,
        index,
    })
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap()
        .0
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments()?;
    let tensors = if arguments.embedded {
        if generated_model::MODEL_DATA_BASE93.is_empty() {
            return Err("embedded model is empty; run export.py --rust-output first".to_string());
        }
        let bytes = decode_base93(generated_model::MODEL_DATA_BASE93)?;
        read_model(&bytes)?
    } else {
        read_model_file(arguments.model.unwrap())?
    };
    let model = MnistCnn::from_tensors(tensors)?;
    let images = IdxImages::read(arguments.images)?;
    if images.dimensions() != (28, 28) {
        return Err(format!(
            "expected 28x28 images, got {:?}",
            images.dimensions()
        ));
    }
    let labels = arguments.labels.map(IdxLabels::read).transpose()?;
    if let Some(labels) = &labels
        && labels.len() != images.len()
    {
        return Err("image and label counts differ".to_string());
    }

    if let Some(index) = arguments.index {
        let logits = model.predict_u8(images.image(index)?)?;
        println!("index={index}");
        println!("prediction={}", argmax(&logits));
        if let Some(labels) = &labels {
            println!("label={}", labels.label(index)?);
        }
        print!("logits=");
        for (position, logit) in logits.iter().enumerate() {
            if position > 0 {
                print!(",");
            }
            print!("{logit:.9}");
        }
        println!();
        return Ok(());
    }

    let count = arguments.limit.unwrap_or(images.len()).min(images.len());
    if count == 0 {
        return Err("no images selected".to_string());
    }
    let start = Instant::now();
    let mut correct = 0usize;
    for index in 0..count {
        let logits = model.predict_u8(images.image(index)?)?;
        if let Some(labels) = &labels
            && argmax(&logits) == labels.label(index)? as usize
        {
            correct += 1;
        }
    }
    let elapsed = start.elapsed();
    println!("images={count}");
    if labels.is_some() {
        println!("correct={correct}");
        println!("accuracy={:.6}", correct as f64 / count as f64);
    }
    println!("elapsed_seconds={:.6}", elapsed.as_secs_f64());
    println!(
        "images_per_second={:.3}",
        count as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
