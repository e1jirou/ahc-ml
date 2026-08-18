use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Token {
    kind: TokenKind,
    text: String,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Ident,
    Literal,
    Other,
}

#[derive(Debug)]
struct Args {
    entry_path: PathBuf,
    lib_path: PathBuf,
    crate_name: Option<String>,
    output_path: String,
    strip_tests: bool,
    max_bytes: Option<usize>,
}

#[derive(Debug)]
struct Replacement {
    start: usize,
    end: usize,
    text: String,
}

#[derive(Debug)]
struct Bundler {
    strip_tests: bool,
    active_files: Vec<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IncludeKind {
    Source,
    String,
    Bytes,
}

fn main() {
    let args = match parse_args(env::args().skip(1).collect()) {
        Ok(args) => args,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("{message}");
            }
            print_usage();
            process::exit(2);
        }
    };

    let entry_path = match canonical_file(&args.entry_path, "entry source") {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{err}");
            process::exit(1);
        }
    };
    let lib_path = match canonical_file(&args.lib_path, "library source") {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{err}");
            process::exit(1);
        }
    };

    let crate_name = match args.crate_name {
        Some(name) => validate_identifier(&name).map(|_| name),
        None => infer_crate_name(&lib_path),
    };
    let crate_name = match crate_name {
        Ok(name) => name,
        Err(err) => {
            eprintln!("{err}");
            process::exit(1);
        }
    };
    let module_name = format!("__atcoder_bundle_{crate_name}");

    let output = match bundle_submission(
        &entry_path,
        &lib_path,
        &crate_name,
        &module_name,
        args.strip_tests,
    ) {
        Ok(output) => output,
        Err(err) => {
            eprintln!("Bundling failed: {err}");
            process::exit(1);
        }
    };

    if let Err(err) = check_output_size(&output, args.max_bytes) {
        eprintln!("Bundling failed: {err}");
        process::exit(1);
    }

    if let Err(err) = write_output(&args.output_path, &output) {
        eprintln!("Could not write {}: {err}", args.output_path);
        process::exit(1);
    }

    if args.output_path != "-" {
        eprintln!(
            "Bundled {} and {} into {} ({} bytes)",
            entry_path.display(),
            lib_path.display(),
            args.output_path,
            output.len()
        );
    }
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    let mut lib_path = None;
    let mut crate_name = None;
    let mut bin_name = None;
    let mut strip_tests = true;
    let mut max_bytes = None;
    let mut paths = Vec::new();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--lib" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--lib requires a path.".to_string())?;
                lib_path = Some(PathBuf::from(value));
            }
            "--crate-name" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--crate-name requires an identifier.".to_string())?;
                crate_name = Some(value.clone());
            }
            "--bin" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--bin requires a target name.".to_string())?;
                bin_name = Some(value.clone());
            }
            "--keep-tests" => strip_tests = false,
            "--max-bytes" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--max-bytes requires a positive integer.".to_string())?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("--max-bytes must be a positive integer, got: {value}"))?;
                if parsed == 0 {
                    return Err("--max-bytes must be a positive integer.".to_string());
                }
                max_bytes = Some(parsed);
            }
            "-h" | "--help" => return Err(String::new()),
            value if value.starts_with('-') && value != "-" => {
                return Err(format!("Unknown option: {value}"));
            }
            value => paths.push(value.to_string()),
        }
        i += 1;
    }

    let (entry_path, output_path) = if let Some(bin_name) = bin_name {
        if paths.len() != 1 {
            return Err("--bin requires exactly one output path.".to_string());
        }
        let lib_path_for_root = lib_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("src/lib.rs"));
        let package_root = find_package_root(&lib_path_for_root).ok_or_else(|| {
            format!(
                "Could not find Cargo.toml above {} while resolving --bin.",
                lib_path_for_root.display()
            )
        })?;
        (
            package_root.join("src/bin").join(format!("{bin_name}.rs")),
            paths.remove(0),
        )
    } else {
        if paths.len() != 2 {
            return Err(
                "Specify <entry.rs> <output.rs>, or use --bin <name> <output.rs>.".to_string(),
            );
        }
        (PathBuf::from(paths.remove(0)), paths.remove(0))
    };

    if entry_path == Path::new("-") {
        return Err(
            "The entry source must be a file path so relative modules can be resolved.".to_string(),
        );
    }

    let lib_path = match lib_path {
        Some(path) => path,
        None => find_default_lib(&entry_path).ok_or_else(|| {
            format!(
                "Could not find src/lib.rs above {}. Pass --lib <path> explicitly.",
                entry_path.display()
            )
        })?,
    };

    Ok(Args {
        entry_path,
        lib_path,
        crate_name,
        output_path,
        strip_tests,
        max_bytes,
    })
}

fn print_usage() {
    eprintln!("Usage: rustc --edition=2024 -O scripts/bundle_submit.rs -o target/bundle_submit");
    eprintln!("       target/bundle_submit [options] <entry.rs> <output.rs|->");
    eprintln!("       target/bundle_submit [options] --bin <name> <output.rs|->");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --lib <path>         Library root (default: discover src/lib.rs)");
    eprintln!("  --crate-name <name>  Library crate identifier (default: Cargo.toml)");
    eprintln!("  --bin <name>         Use src/bin/<name>.rs as the entry source");
    eprintln!("  --keep-tests          Keep #[cfg(test)] items instead of removing them");
    eprintln!("  --max-bytes <n>      Reject output larger than n bytes");
    eprintln!();
    eprintln!(
        "The bundler expands first-party mod and include macros with one string-literal path. Registry crates remain imports."
    );
}

fn canonical_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_file() {
        return Err(format!("The {label} is not a file: {}", path.display()));
    }
    fs::canonicalize(path)
        .map_err(|err| format!("Could not canonicalize {}: {err}", path.display()))
}

fn find_default_lib(entry_path: &Path) -> Option<PathBuf> {
    let mut current = entry_path.parent()?;
    loop {
        let candidate = current.join("src/lib.rs");
        if candidate.is_file() {
            return Some(candidate);
        }
        current = current.parent()?;
    }
}

fn find_package_root(path: &Path) -> Option<PathBuf> {
    let mut current = if path.is_dir() { path } else { path.parent()? };
    loop {
        if current.join("Cargo.toml").is_file() {
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

fn infer_crate_name(lib_path: &Path) -> Result<String, String> {
    let package_root = find_package_root(lib_path).ok_or_else(|| {
        format!(
            "Could not find Cargo.toml above {}. Pass --crate-name explicitly.",
            lib_path.display()
        )
    })?;
    let manifest_path = package_root.join("Cargo.toml");
    let manifest = fs::read_to_string(&manifest_path)
        .map_err(|err| format!("Could not read {}: {err}", manifest_path.display()))?;

    let mut section = "";
    let mut package_name = None;
    let mut lib_name = None;
    for line in manifest.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') && line.ends_with(']') {
            section = &line[1..line.len() - 1];
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != "name" {
            continue;
        }
        let Some(value) = parse_toml_string(value.trim()) else {
            continue;
        };
        match section {
            "package" => package_name = Some(value),
            "lib" => lib_name = Some(value),
            _ => {}
        }
    }

    let name = lib_name.or(package_name).ok_or_else(|| {
        format!(
            "Could not infer a crate name from {}. Pass --crate-name explicitly.",
            manifest_path.display()
        )
    })?;
    let name = name.replace('-', "_");
    validate_identifier(&name)?;
    Ok(name)
}

fn parse_toml_string(value: &str) -> Option<String> {
    let value = value.trim();
    if !value.starts_with('"') {
        return None;
    }
    let end = value[1..].find('"')? + 1;
    Some(value[1..end].to_string())
}

fn validate_identifier(value: &str) -> Result<(), String> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err("An identifier must not be empty.".to_string());
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return Err(format!("Not a Rust identifier: {value}"));
    }
    Ok(())
}

fn bundle_submission(
    entry_path: &Path,
    lib_path: &Path,
    crate_name: &str,
    module_name: &str,
    strip_tests: bool,
) -> Result<String, String> {
    let entry_source = fs::read_to_string(entry_path)
        .map_err(|err| format!("Could not read {}: {err}", entry_path.display()))?;
    reject_root_inner_doc_comments(&entry_source, entry_path, "entry source")?;

    let mut bundler = Bundler {
        strip_tests,
        active_files: Vec::new(),
    };
    let library = bundler.bundle_file(lib_path)?;
    reject_root_inner_attributes(&library, lib_path, "library")?;
    reject_macro_exports(&library, lib_path)?;
    let library = rewrite_library_crate_references(&library, module_name)?;

    let entry = bundler.bundle_file(entry_path)?;
    let entry = rewrite_entry_crate_references(&entry, crate_name, module_name)?;
    let (inner_attributes, entry) = extract_leading_inner_attributes(&entry)?;

    let output = format!(
        "{inner_attributes}// Generated by scripts/bundle_submit.rs. Do not edit.\n\n#[allow(dead_code, unused_imports)]\npub mod {module_name} {{\n{library}\n}}\n\n{entry}\n"
    );
    validate_output(&output)?;
    Ok(output)
}

impl Bundler {
    fn bundle_file(&mut self, path: &Path) -> Result<String, String> {
        let path = canonical_file(path, "source")?;
        let module_dir = module_directory(&path, false)?;
        self.bundle_canonical_file(path, module_dir)
    }

    fn bundle_included_file(&mut self, path: &Path) -> Result<String, String> {
        let path = canonical_file(path, "included source")?;
        let module_dir = module_directory(&path, true)?;
        self.bundle_canonical_file(path, module_dir)
    }

    fn bundle_canonical_file(
        &mut self,
        path: PathBuf,
        module_dir: PathBuf,
    ) -> Result<String, String> {
        if let Some(position) = self.active_files.iter().position(|active| active == &path) {
            let mut chain: Vec<String> = self.active_files[position..]
                .iter()
                .map(|item| item.display().to_string())
                .collect();
            chain.push(path.display().to_string());
            return Err(format!("Cyclic source inclusion: {}", chain.join(" -> ")));
        }

        let source = fs::read_to_string(&path)
            .map_err(|err| format!("Could not read {}: {err}", path.display()))?;
        if source.starts_with("#!") && !source.starts_with("#![") {
            return Err(format!(
                "Shebangs are not supported while bundling {}.",
                path.display()
            ));
        }
        self.active_files.push(path.clone());
        let result = self.bundle_source(&path, &module_dir, &source);
        self.active_files.pop();
        result
    }

    fn bundle_source(
        &mut self,
        path: &Path,
        module_dir: &Path,
        source: &str,
    ) -> Result<String, String> {
        let source = if self.strip_tests {
            strip_cfg_test_items(source)?
        } else {
            source.to_string()
        };
        let tokens = lex(&source)?;
        reject_path_attributes(&tokens, path)?;
        let scopes = module_scopes(&tokens)?;
        let mut replacements = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            if scopes[i] && tokens[i].text == "mod" {
                if let Some((source_name, file_name, semicolon)) =
                    external_module_declaration(&tokens, i)
                {
                    let child_path = resolve_module(path, module_dir, &file_name)?;
                    let child = self.bundle_file(&child_path)?;
                    replacements.push(Replacement {
                        start: tokens[i].start,
                        end: tokens[semicolon].end,
                        text: inline_module(&source_name, &child),
                    });
                    i = semicolon + 1;
                    continue;
                }
            }

            if let Some(kind) = include_kind(&tokens[i].text) {
                if let Some((close, include_path)) = direct_include_path(&tokens, i)? {
                    let include_path = resolve_include(path, &include_path)?;
                    let replacement = match kind {
                        IncludeKind::Source => self.bundle_included_file(&include_path)?,
                        IncludeKind::String => {
                            let content = fs::read_to_string(&include_path).map_err(|err| {
                                format!("Could not read {}: {err}", include_path.display())
                            })?;
                            format!("{content:?}")
                        }
                        IncludeKind::Bytes => {
                            let content = fs::read(&include_path).map_err(|err| {
                                format!("Could not read {}: {err}", include_path.display())
                            })?;
                            rust_byte_string(&content)
                        }
                    };
                    let mut end = tokens[close].end;
                    if is_standalone_module_macro_item(&tokens, i)
                        && tokens.get(close + 1).is_some_and(|token| token.text == ";")
                    {
                        end = tokens[close + 1].end;
                    }
                    replacements.push(Replacement {
                        start: tokens[i].start,
                        end,
                        text: replacement,
                    });
                    i = close + 1;
                    continue;
                }
            }

            i += 1;
        }

        apply_replacements(&source, replacements)
    }
}

fn is_standalone_module_macro_item(tokens: &[Token], i: usize) -> bool {
    let Some(module_open) = enclosing_module_opening_brace(tokens, i) else {
        return false;
    };
    let mut start = module_open.map_or(0, |open| open + 1);
    let mut braces = 0usize;
    let mut parens = 0usize;
    let mut brackets = 0usize;

    for (j, token) in tokens.iter().enumerate().take(i).skip(start) {
        let at_item_level = braces == 0 && parens == 0 && brackets == 0;
        match token.text.as_str() {
            ";" if at_item_level => start = j + 1,
            "{" => braces += 1,
            "}" => {
                braces -= 1;
                if braces == 0 && parens == 0 && brackets == 0 {
                    start = j + 1;
                }
            }
            "(" => parens += 1,
            ")" => parens -= 1,
            "[" => brackets += 1,
            "]" => brackets -= 1,
            _ => {}
        }
    }

    loop {
        if is_outer_attribute(tokens, start) {
            let Some(close) = matching_delimiter(tokens, start + 1) else {
                return false;
            };
            start = close + 1;
        } else if tokens.get(start).is_some_and(|token| token.text == "#")
            && tokens.get(start + 1).is_some_and(|token| token.text == "!")
            && tokens.get(start + 2).is_some_and(|token| token.text == "[")
        {
            let Some(close) = matching_delimiter(tokens, start + 2) else {
                return false;
            };
            start = close + 1;
        } else {
            break;
        }
    }

    start == i
}

fn enclosing_module_opening_brace(tokens: &[Token], i: usize) -> Option<Option<usize>> {
    let mut braces = Vec::new();
    for (j, token) in tokens.iter().enumerate().take(i) {
        match token.text.as_str() {
            "{" => braces.push(j),
            "}" => {
                braces.pop()?;
            }
            _ => {}
        }
    }

    let Some(open) = braces.last().copied() else {
        return Some(None);
    };
    if open >= 2 && tokens[open - 2].text == "mod" && tokens[open - 1].kind == TokenKind::Ident
    {
        Some(Some(open))
    } else {
        None
    }
}

fn reject_path_attributes(tokens: &[Token], path: &Path) -> Result<(), String> {
    for window in tokens.windows(3) {
        if window[0].text == "#" && window[1].text == "[" && window[2].text == "path" {
            return Err(format!(
                "#[path = ...] is not supported while bundling {}.",
                path.display()
            ));
        }
    }
    Ok(())
}

fn strip_cfg_test_items(source: &str) -> Result<String, String> {
    let tokens = lex(source)?;
    let mut replacements = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        if is_cfg_test_attribute(&tokens, i) {
            let start = first_outer_attribute_of_item(&tokens, i)?;
            let end = skip_attributes_and_item(&tokens, i + 7)?;
            if end <= i + 7 {
                return Err("#[cfg(test)] did not have a following item.".to_string());
            }
            replacements.push(Replacement {
                start: if start == 0 { 0 } else { tokens[start - 1].end },
                end: tokens[end - 1].end,
                text: String::new(),
            });
            i = end;
        } else {
            i += 1;
        }
    }

    apply_replacements(source, replacements)
}

fn is_cfg_test_attribute(tokens: &[Token], i: usize) -> bool {
    matches!(
        tokens.get(i..i + 7),
        Some([hash, open, cfg, paren_open, test, paren_close, close])
            if hash.text == "#"
                && open.text == "["
                && cfg.text == "cfg"
                && paren_open.text == "("
                && test.text == "test"
                && paren_close.text == ")"
                && close.text == "]"
    )
}

fn first_outer_attribute_of_item(tokens: &[Token], mut start: usize) -> Result<usize, String> {
    while start > 1 && tokens[start - 1].text == "]" {
        let open = matching_opening_delimiter(tokens, start - 1)
            .ok_or_else(|| "An attribute bracket was not closed.".to_string())?;
        if open == 0 || tokens[open - 1].text != "#" {
            break;
        }
        start = open - 1;
    }
    Ok(start)
}

fn skip_attributes_and_item(tokens: &[Token], mut i: usize) -> Result<usize, String> {
    while is_outer_attribute(tokens, i) {
        let close = matching_delimiter(tokens, i + 1)
            .ok_or_else(|| "An attribute bracket was not closed.".to_string())?;
        i = close + 1;
    }
    if i >= tokens.len() {
        return Err("Expected an item after #[cfg(test)].".to_string());
    }

    let mut j = i;
    while j < tokens.len() {
        match tokens[j].text.as_str() {
            ";" => return Ok(j + 1),
            "{" => {
                let close = matching_delimiter(tokens, j)
                    .ok_or_else(|| "A block was not closed.".to_string())?;
                let mut end = close + 1;
                if tokens.get(end).is_some_and(|token| token.text == ";") {
                    end += 1;
                }
                return Ok(end);
            }
            "(" | "[" => {
                j = matching_delimiter(tokens, j)
                    .ok_or_else(|| "A delimiter was not closed.".to_string())?
                    + 1;
            }
            _ => j += 1,
        }
    }
    Err("Could not find the end of a #[cfg(test)] item.".to_string())
}

fn is_outer_attribute(tokens: &[Token], i: usize) -> bool {
    tokens.get(i).is_some_and(|token| token.text == "#")
        && tokens.get(i + 1).is_some_and(|token| token.text == "[")
}

fn external_module_declaration(tokens: &[Token], i: usize) -> Option<(String, String, usize)> {
    if tokens.get(i)?.text != "mod" {
        return None;
    }
    let name = tokens.get(i + 1)?;
    let semicolon = tokens.get(i + 2)?;
    if name.kind != TokenKind::Ident || semicolon.text != ";" {
        return None;
    }
    Some((
        name.text.clone(),
        module_file_name(&name.text).to_string(),
        i + 2,
    ))
}

fn module_file_name(name: &str) -> &str {
    name.strip_prefix("r#").unwrap_or(name)
}

fn module_directory(path: &Path, included: bool) -> Result<PathBuf, String> {
    let parent_dir = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory.", path.display()))?;
    if included {
        return Ok(parent_dir.to_path_buf());
    }
    let parent_file = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} has no UTF-8 file name.", path.display()))?;
    match parent_file {
        "lib.rs" | "main.rs" | "mod.rs" => Ok(parent_dir.to_path_buf()),
        _ => {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| format!("{} has no UTF-8 file stem.", path.display()))?;
            Ok(parent_dir.join(module_file_name(stem)))
        }
    }
}

fn resolve_module(source_path: &Path, module_dir: &Path, name: &str) -> Result<PathBuf, String> {
    let direct = module_dir.join(format!("{name}.rs"));
    let directory = module_dir.join(name).join("mod.rs");
    let direct_exists = direct.is_file();
    let directory_exists = directory.is_file();

    match (direct_exists, directory_exists) {
        (true, false) => Ok(direct),
        (false, true) => Ok(directory),
        (false, false) => Err(format!(
            "Could not resolve mod {name}; from {}. Looked for {} and {}.",
            source_path.display(),
            direct.display(),
            directory.display()
        )),
        (true, true) => Err(format!(
            "Ambiguous mod {name}; from {}: both {} and {} exist.",
            source_path.display(),
            direct.display(),
            directory.display()
        )),
    }
}

fn inline_module(name: &str, content: &str) -> String {
    let mut output = format!("mod {name} {{\n");
    output.push_str(content);
    if !content.ends_with('\n') {
        output.push('\n');
    }
    output.push('}');
    output
}

fn include_kind(name: &str) -> Option<IncludeKind> {
    match name {
        "include" => Some(IncludeKind::Source),
        "include_str" => Some(IncludeKind::String),
        "include_bytes" => Some(IncludeKind::Bytes),
        _ => None,
    }
}

fn direct_include_path(tokens: &[Token], i: usize) -> Result<Option<(usize, String)>, String> {
    if tokens.get(i + 1).is_none_or(|token| token.text != "!")
        || tokens.get(i + 2).is_none_or(|token| token.text != "(")
    {
        return Ok(None);
    }
    let close = matching_delimiter(tokens, i + 2)
        .ok_or_else(|| "An include macro delimiter was not closed.".to_string())?;
    if close != i + 4 || tokens[i + 3].kind != TokenKind::Literal {
        return Err("include macros must use exactly one string literal path.".to_string());
    }
    let path = parse_rust_string_literal(&tokens[i + 3].text)?;
    Ok(Some((close, path)))
}

fn resolve_include(parent: &Path, include_path: &str) -> Result<PathBuf, String> {
    let include_path = Path::new(include_path);
    let path = if include_path.is_absolute() {
        include_path.to_path_buf()
    } else {
        parent
            .parent()
            .ok_or_else(|| format!("{} has no parent directory.", parent.display()))?
            .join(include_path)
    };
    canonical_file(&path, "included source")
}

fn parse_rust_string_literal(literal: &str) -> Result<String, String> {
    if literal.starts_with('b') || literal.starts_with('c') {
        return Err(format!(
            "include path must be a string literal, not {literal}"
        ));
    }
    if literal.starts_with('r') {
        return parse_raw_string_literal(literal);
    }
    if !(literal.starts_with('"') && literal.ends_with('"')) {
        return Err(format!("Invalid include string literal: {literal}"));
    }

    let mut output = String::new();
    let mut chars = literal[1..literal.len() - 1].chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        let escaped = chars
            .next()
            .ok_or_else(|| "A string literal ended after a backslash.".to_string())?;
        match escaped {
            '\\' => output.push('\\'),
            '"' => output.push('"'),
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            '0' => output.push('\0'),
            'x' => {
                let high = chars
                    .next()
                    .ok_or_else(|| "Incomplete \\x escape in include path.".to_string())?;
                let low = chars
                    .next()
                    .ok_or_else(|| "Incomplete \\x escape in include path.".to_string())?;
                let byte = (hex_value(high)? << 4) | hex_value(low)?;
                output.push(byte as char);
            }
            'u' => {
                if chars.next() != Some('{') {
                    return Err("Expected { after \\u in include path.".to_string());
                }
                let mut hex = String::new();
                loop {
                    let value = chars
                        .next()
                        .ok_or_else(|| "Unclosed \\u escape in include path.".to_string())?;
                    if value == '}' {
                        break;
                    }
                    hex.push(value);
                }
                let value = u32::from_str_radix(&hex, 16)
                    .map_err(|_| "Invalid \\u escape in include path.".to_string())?;
                let value = char::from_u32(value)
                    .ok_or_else(|| "Invalid Unicode scalar in include path.".to_string())?;
                output.push(value);
            }
            _ => return Err(format!("Unsupported escape \\{escaped} in include path.")),
        }
    }
    Ok(output)
}

fn parse_raw_string_literal(literal: &str) -> Result<String, String> {
    let bytes = literal.as_bytes();
    let mut i = 1;
    while bytes.get(i) == Some(&b'#') {
        i += 1;
    }
    if bytes.get(i) != Some(&b'"') {
        return Err(format!("Invalid raw include string literal: {literal}"));
    }
    let hashes = i - 1;
    let suffix_length = 1 + hashes;
    if literal.len() < i + 1 + suffix_length {
        return Err(format!("Invalid raw include string literal: {literal}"));
    }
    let content_start = i + 1;
    let content_end = literal.len() - suffix_length;
    if !literal[content_end..].starts_with('"')
        || !literal[content_end + 1..].chars().all(|ch| ch == '#')
    {
        return Err(format!("Invalid raw include string literal: {literal}"));
    }
    Ok(literal[content_start..content_end].to_string())
}

fn hex_value(ch: char) -> Result<u8, String> {
    ch.to_digit(16)
        .map(|value| value as u8)
        .ok_or_else(|| format!("Invalid hexadecimal digit: {ch}"))
}

fn rust_byte_string(bytes: &[u8]) -> String {
    let mut output = String::from("b\"");
    for &byte in bytes {
        match byte {
            b'\\' => output.push_str("\\\\"),
            b'"' => output.push_str("\\\""),
            b'\n' => output.push_str("\\n"),
            b'\r' => output.push_str("\\r"),
            b'\t' => output.push_str("\\t"),
            0x20..=0x7e => output.push(byte as char),
            _ => output.push_str(&format!("\\x{byte:02x}")),
        }
    }
    output.push('"');
    output
}

fn rewrite_path_prefixes(source: &str, from: &str, to: &str) -> Result<String, String> {
    let tokens = lex(source)?;
    let mut replacements = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        if token.text == from && tokens.get(i + 1).is_some_and(|next| next.text == "::") {
            let start =
                if i > 0 && tokens[i - 1].text == "::" && is_leading_path_separator(&tokens, i - 1)
                {
                    tokens[i - 1].start
                } else {
                    token.start
                };
            replacements.push(Replacement {
                start,
                end: token.end,
                text: to.to_string(),
            });
        }
    }
    apply_replacements(source, replacements)
}

fn is_leading_path_separator(tokens: &[Token], separator: usize) -> bool {
    if separator == 0 {
        return true;
    }
    let previous = &tokens[separator - 1];
    if previous.kind != TokenKind::Ident {
        return true;
    }
    matches!(
        previous.text.as_str(),
        "as" | "async"
            | "await"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "match"
            | "mod"
            | "move"
            | "pub"
            | "return"
            | "self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "type"
            | "union"
            | "unsafe"
            | "use"
            | "where"
            | "while"
    )
}

fn rewrite_library_crate_references(source: &str, module_name: &str) -> Result<String, String> {
    let replacement = format!("crate::{module_name}");
    let tokens = lex(source)?;
    let mut replacements = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        if tokens[i].text == "crate" {
            let is_path = tokens.get(i + 1).is_some_and(|next| next.text == "::");
            let is_use_alias = tokens
                .get(i.wrapping_sub(1))
                .is_some_and(|previous| previous.text == "use")
                && tokens.get(i + 1).is_some_and(|next| next.text == "as");
            let is_visibility_root = i >= 3
                && tokens[i - 3].text == "pub"
                && tokens[i - 2].text == "("
                && tokens[i - 1].text == "in"
                && tokens.get(i + 1).is_some_and(|next| next.text == ")");
            if is_path || is_use_alias || is_visibility_root {
                replacements.push(Replacement {
                    start: tokens[i].start,
                    end: tokens[i].end,
                    text: replacement.clone(),
                });
            }
        }

        if tokens[i].text == "extern"
            && tokens.get(i + 1).is_some_and(|token| token.text == "crate")
            && tokens.get(i + 2).is_some_and(|token| token.text == "self")
        {
            if tokens.get(i + 3).is_none_or(|token| token.text != "as") {
                return Err(
                    "extern crate self without an alias is not supported in a bundled library."
                        .to_string(),
                );
            }
            let alias = tokens
                .get(i + 4)
                .ok_or_else(|| "Expected an alias after extern crate self as.".to_string())?;
            if alias.kind != TokenKind::Ident {
                return Err("Expected an identifier after extern crate self as.".to_string());
            }
            let semicolon = tokens
                .get(i + 5)
                .ok_or_else(|| "Expected a semicolon after extern crate self.".to_string())?;
            if semicolon.text != ";" {
                return Err("Expected a semicolon after extern crate self.".to_string());
            }
            replacements.push(Replacement {
                start: tokens[i].start,
                end: semicolon.end,
                text: format!("use {replacement} as {};", alias.text),
            });
            i += 6;
            continue;
        }

        i += 1;
    }
    apply_replacements(source, replacements)
}

fn rewrite_entry_crate_references(
    source: &str,
    crate_name: &str,
    module_name: &str,
) -> Result<String, String> {
    let replacement = format!("crate::{module_name}");
    let source = rewrite_path_prefixes(source, crate_name, &replacement)?;
    let tokens = lex(&source)?;
    let mut replacements = Vec::new();
    let mut i = 0;

    while i + 2 < tokens.len() {
        if tokens[i].text == "use" && tokens[i + 1].text == crate_name {
            match tokens[i + 2].text.as_str() {
                ";" => replacements.push(Replacement {
                    start: tokens[i].start,
                    end: tokens[i + 2].end,
                    text: format!("use {replacement} as {crate_name};"),
                }),
                "as" => replacements.push(Replacement {
                    start: tokens[i + 1].start,
                    end: tokens[i + 1].end,
                    text: replacement.clone(),
                }),
                _ => {}
            }
        }
        if tokens[i].text == "extern"
            && tokens[i + 1].text == "crate"
            && tokens[i + 2].text == crate_name
        {
            let mut end = i + 3;
            let mut alias = crate_name.to_string();
            if tokens.get(end).is_some_and(|token| token.text == "as") {
                let alias_token = tokens
                    .get(end + 1)
                    .ok_or_else(|| "Expected an alias after extern crate ... as.".to_string())?;
                if alias_token.kind != TokenKind::Ident {
                    return Err("Expected an identifier after extern crate ... as.".to_string());
                }
                alias = alias_token.text.clone();
                end += 2;
            }
            if tokens.get(end).is_none_or(|token| token.text != ";") {
                return Err("Expected a semicolon after extern crate.".to_string());
            }
            replacements.push(Replacement {
                start: tokens[i].start,
                end: tokens[end].end,
                text: format!("use {replacement} as {alias};"),
            });
            i = end;
        }
        i += 1;
    }
    apply_replacements(&source, replacements)
}

fn reject_root_inner_attributes(source: &str, path: &Path, label: &str) -> Result<(), String> {
    let tokens = lex(source)?;
    let scopes = module_scopes(&tokens)?;
    for i in 0..tokens.len() {
        if scopes[i]
            && tokens[i].text == "#"
            && tokens.get(i + 1).is_some_and(|token| token.text == "!")
            && tokens.get(i + 2).is_some_and(|token| token.text == "[")
        {
            return Err(format!(
                "The {label} {} contains a root inner attribute. Root inner attributes in bundled libraries are not supported.",
                path.display()
            ));
        }
    }
    Ok(())
}

fn reject_root_inner_doc_comments(source: &str, path: &Path, label: &str) -> Result<(), String> {
    if let Some(position) = find_root_inner_doc_comment(source)? {
        return Err(format!(
            "The {label} {} contains an inner documentation comment near byte {position}. Inner documentation comments are not supported in bundled output.",
            path.display()
        ));
    }
    Ok(())
}

fn find_root_inner_doc_comment(source: &str) -> Result<Option<usize>, String> {
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut braces = 0usize;
    while i < bytes.len() {
        if starts_with(bytes, i, b"//") {
            if braces == 0 && starts_with(bytes, i, b"//!") {
                return Ok(Some(i));
            }
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if starts_with(bytes, i, b"/*") {
            if braces == 0 && starts_with(bytes, i, b"/*!") {
                return Ok(Some(i));
            }
            i = skip_block_comment(bytes, i)?;
            continue;
        }
        if let Some(end) = raw_string_end(bytes, i)? {
            i = end;
            continue;
        }
        if bytes[i] == b'"' {
            i = quoted_end(bytes, i, b'"')?;
            continue;
        }
        if (bytes[i] == b'b' || bytes[i] == b'c') && bytes.get(i + 1) == Some(&b'"') {
            i = quoted_end(bytes, i + 1, b'"')?;
            continue;
        }
        if (bytes[i] == b'b' || bytes[i] == b'c') && bytes.get(i + 1) == Some(&b'\'') {
            i = char_literal_end(bytes, i + 1)?;
            continue;
        }
        if bytes[i] == b'\'' {
            i = lifetime_or_char_end(bytes, i)?;
            continue;
        }
        if bytes[i] == b'{' {
            braces += 1;
        } else if bytes[i] == b'}' {
            braces = braces.saturating_sub(1);
        }
        i += punct_len(bytes, i);
    }
    Ok(None)
}

fn reject_macro_exports(source: &str, path: &Path) -> Result<(), String> {
    let tokens = lex(source)?;
    if tokens.windows(3).any(|window| {
        window[0].text == "#" && window[1].text == "[" && window[2].text == "macro_export"
    }) {
        return Err(format!(
            "The library {} uses #[macro_export]. It is not supported because bundled libraries are nested in an internal module.",
            path.display()
        ));
    }
    Ok(())
}

fn extract_leading_inner_attributes(source: &str) -> Result<(String, String), String> {
    let tokens = lex(source)?;
    let scopes = module_scopes(&tokens)?;
    let mut replacements = Vec::new();
    let mut attributes = String::new();
    let mut i = 0;

    while scopes.get(i).copied().unwrap_or(false)
        && tokens.get(i).is_some_and(|token| token.text == "#")
        && tokens.get(i + 1).is_some_and(|token| token.text == "!")
        && tokens.get(i + 2).is_some_and(|token| token.text == "[")
    {
        let close = matching_delimiter(&tokens, i + 2)
            .ok_or_else(|| "An inner attribute bracket was not closed.".to_string())?;
        attributes.push_str(&source[tokens[i].start..tokens[close].end]);
        attributes.push('\n');
        replacements.push(Replacement {
            start: tokens[i].start,
            end: tokens[close].end,
            text: String::new(),
        });
        i = close + 1;
    }

    for j in i..tokens.len() {
        if scopes[j]
            && tokens[j].text == "#"
            && tokens.get(j + 1).is_some_and(|token| token.text == "!")
            && tokens.get(j + 2).is_some_and(|token| token.text == "[")
        {
            return Err(
                "Inner attributes must appear at the beginning of the entry source.".to_string(),
            );
        }
    }
    Ok((attributes, apply_replacements(source, replacements)?))
}

fn validate_output(source: &str) -> Result<(), String> {
    let tokens = lex(source)?;
    for i in 0..tokens.len() {
        if external_module_declaration(&tokens, i).is_some() {
            return Err(format!(
                "The generated output still contains an external mod declaration near byte {}.",
                tokens[i].start
            ));
        }
        if include_kind(&tokens[i].text).is_some()
            && tokens.get(i + 1).is_some_and(|token| token.text == "!")
        {
            return Err(format!(
                "The generated output still contains an include macro near byte {}.",
                tokens[i].start
            ));
        }
    }
    Ok(())
}

fn check_output_size(source: &str, max_bytes: Option<usize>) -> Result<(), String> {
    let Some(max_bytes) = max_bytes else {
        return Ok(());
    };
    if source.len() > max_bytes {
        return Err(format!(
            "The generated output is {} bytes, exceeding the {}-byte limit.",
            source.len(),
            max_bytes
        ));
    }
    Ok(())
}

fn apply_replacements(source: &str, mut replacements: Vec<Replacement>) -> Result<String, String> {
    replacements.sort_by_key(|replacement| replacement.start);
    let mut output = String::with_capacity(source.len());
    let mut cursor = 0;
    for replacement in replacements {
        if replacement.start < cursor || replacement.end < replacement.start {
            return Err("Overlapping source transformations were requested.".to_string());
        }
        output.push_str(&source[cursor..replacement.start]);
        output.push_str(&replacement.text);
        cursor = replacement.end;
    }
    output.push_str(&source[cursor..]);
    Ok(output)
}

fn module_scopes(tokens: &[Token]) -> Result<Vec<bool>, String> {
    let mut scopes = Vec::with_capacity(tokens.len());
    let mut braces = 0usize;
    let mut parens = 0usize;
    let mut brackets = 0usize;

    for token in tokens {
        scopes.push(braces == 0 && parens == 0 && brackets == 0);
        match token.text.as_str() {
            "{" => braces += 1,
            "}" => {
                braces = braces
                    .checked_sub(1)
                    .ok_or_else(|| "An unmatched } was found.".to_string())?
            }
            "(" => parens += 1,
            ")" => {
                parens = parens
                    .checked_sub(1)
                    .ok_or_else(|| "An unmatched ) was found.".to_string())?
            }
            "[" => brackets += 1,
            "]" => {
                brackets = brackets
                    .checked_sub(1)
                    .ok_or_else(|| "An unmatched ] was found.".to_string())?
            }
            _ => {}
        }
    }
    if braces != 0 || parens != 0 || brackets != 0 {
        return Err("An opening delimiter was not closed.".to_string());
    }
    Ok(scopes)
}

fn matching_delimiter(tokens: &[Token], start: usize) -> Option<usize> {
    let open = tokens.get(start)?.text.as_str();
    let close = match open {
        "(" => ")",
        "[" => "]",
        "{" => "}",
        _ => return None,
    };
    let mut depth = 0usize;
    for (i, token) in tokens.iter().enumerate().skip(start) {
        if token.text == open {
            depth += 1;
        } else if token.text == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

fn matching_opening_delimiter(tokens: &[Token], end: usize) -> Option<usize> {
    let close = tokens.get(end)?.text.as_str();
    let open = match close {
        ")" => "(",
        "]" => "[",
        "}" => "{",
        _ => return None,
    };
    let mut depth = 0usize;
    for i in (0..=end).rev() {
        if tokens[i].text == close {
            depth += 1;
        } else if tokens[i].text == open {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

fn lex(source: &str) -> Result<Vec<Token>, String> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;

    if bytes.starts_with(b"#!") && !bytes.starts_with(b"#![") {
        while i < bytes.len() && bytes[i] != b'\n' {
            i += 1;
        }
    }

    while i < bytes.len() {
        let byte = bytes[i];
        if byte.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if starts_with(bytes, i, b"//") {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if starts_with(bytes, i, b"/*") {
            i = skip_block_comment(bytes, i)?;
            continue;
        }
        if let Some(end) = raw_string_end(bytes, i)? {
            tokens.push(token(TokenKind::Literal, source, i, end));
            i = end;
            continue;
        }
        if byte == b'"' {
            let end = quoted_end(bytes, i, b'"')?;
            tokens.push(token(TokenKind::Literal, source, i, end));
            i = end;
            continue;
        }
        if (byte == b'b' || byte == b'c') && bytes.get(i + 1) == Some(&b'"') {
            let end = quoted_end(bytes, i + 1, b'"')?;
            tokens.push(token(TokenKind::Literal, source, i, end));
            i = end;
            continue;
        }
        if (byte == b'b' || byte == b'c') && bytes.get(i + 1) == Some(&b'\'') {
            let end = char_literal_end(bytes, i + 1)?;
            tokens.push(token(TokenKind::Literal, source, i, end));
            i = end;
            continue;
        }
        if byte == b'\'' {
            let end = lifetime_or_char_end(bytes, i)?;
            let kind = if bytes.get(i + 1).is_some_and(|next| is_ident_start(*next))
                && bytes.get(end) != Some(&b'\'')
            {
                TokenKind::Ident
            } else {
                TokenKind::Literal
            };
            tokens.push(token(kind, source, i, end));
            i = end;
            continue;
        }
        if is_ident_start(byte) {
            let end = ident_end(bytes, i);
            tokens.push(token(TokenKind::Ident, source, i, end));
            i = end;
            continue;
        }
        let length = punct_len(bytes, i);
        tokens.push(token(TokenKind::Other, source, i, i + length));
        i += length;
    }
    Ok(tokens)
}

fn token(kind: TokenKind, source: &str, start: usize, end: usize) -> Token {
    Token {
        kind,
        text: source[start..end].to_string(),
        start,
        end,
    }
}

fn starts_with(bytes: &[u8], i: usize, needle: &[u8]) -> bool {
    bytes.get(i..i + needle.len()) == Some(needle)
}

fn skip_block_comment(bytes: &[u8], mut i: usize) -> Result<usize, String> {
    i += 2;
    let mut depth = 1usize;
    while i < bytes.len() {
        if starts_with(bytes, i, b"/*") {
            depth += 1;
            i += 2;
        } else if starts_with(bytes, i, b"*/") {
            depth -= 1;
            i += 2;
            if depth == 0 {
                return Ok(i);
            }
        } else {
            i += 1;
        }
    }
    Err("An unterminated block comment was found.".to_string())
}

fn raw_string_end(bytes: &[u8], i: usize) -> Result<Option<usize>, String> {
    let mut j = i;
    if starts_with(bytes, j, b"br") || starts_with(bytes, j, b"cr") {
        j += 2;
    } else if bytes.get(j) == Some(&b'r') {
        j += 1;
    } else {
        return Ok(None);
    }
    let hash_start = j;
    while bytes.get(j) == Some(&b'#') {
        j += 1;
    }
    let hashes = j - hash_start;
    if bytes.get(j) != Some(&b'"') {
        return Ok(None);
    }
    j += 1;
    while j < bytes.len() {
        if bytes[j] == b'"' && has_hashes(bytes, j + 1, hashes) {
            return Ok(Some(j + 1 + hashes));
        }
        j += 1;
    }
    Err("An unterminated raw string literal was found.".to_string())
}

fn has_hashes(bytes: &[u8], mut i: usize, hashes: usize) -> bool {
    for _ in 0..hashes {
        if bytes.get(i) != Some(&b'#') {
            return false;
        }
        i += 1;
    }
    true
}

fn quoted_end(bytes: &[u8], mut i: usize, quote: u8) -> Result<usize, String> {
    i += 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            value if value == quote => return Ok(i + 1),
            _ => i += 1,
        }
    }
    Err("An unterminated string or character literal was found.".to_string())
}

fn char_literal_end(bytes: &[u8], i: usize) -> Result<usize, String> {
    quoted_end(bytes, i, b'\'')
}

fn lifetime_or_char_end(bytes: &[u8], i: usize) -> Result<usize, String> {
    if bytes.get(i + 1).is_some_and(|next| is_ident_start(*next)) {
        let end = ident_end(bytes, i + 1);
        if bytes.get(end) != Some(&b'\'') {
            return Ok(end);
        }
    }
    char_literal_end(bytes, i)
}

fn ident_end(bytes: &[u8], i: usize) -> usize {
    if starts_with(bytes, i, b"r#") && bytes.get(i + 2).is_some_and(|next| is_ident_start(*next)) {
        let mut j = i + 3;
        while bytes.get(j).is_some_and(|next| is_ident_continue(*next)) {
            j += 1;
        }
        return j;
    }
    let mut j = i + 1;
    while bytes.get(j).is_some_and(|next| is_ident_continue(*next)) {
        j += 1;
    }
    j
}

fn is_ident_start(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic()
}

fn is_ident_continue(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

fn punct_len(bytes: &[u8], i: usize) -> usize {
    const THREE: [&[u8]; 4] = [b"<<=", b">>=", b"...", b"..="];
    const TWO: [&[u8]; 21] = [
        b"::", b"->", b"=>", b"==", b"!=", b"<=", b">=", b"&&", b"||", b"+=", b"-=", b"*=", b"/=",
        b"%=", b"^=", b"&=", b"|=", b"<<", b">>", b"..", b"##",
    ];
    if THREE
        .iter()
        .any(|punctuation| starts_with(bytes, i, punctuation))
    {
        3
    } else if TWO
        .iter()
        .any(|punctuation| starts_with(bytes, i, punctuation))
    {
        2
    } else {
        1
    }
}

fn write_output(path: &str, output: &str) -> io::Result<()> {
    if path == "-" {
        io::stdout().write_all(output.as_bytes())
    } else {
        fs::write(path, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = env::temp_dir().join(format!(
                "neural-network-rust-bundle-submit-{name}-{}-{nonce}",
                process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn write(&self, relative: &str, content: &str) {
            let path = self.path.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, content).unwrap();
        }

        fn path(&self, relative: &str) -> PathBuf {
            self.path.join(relative)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn bundle_fixture(temp: &TempDir) -> String {
        bundle_submission(
            &temp.path("src/bin/solve.rs"),
            &temp.path("src/lib.rs"),
            "demo_lib",
            "__atcoder_bundle_demo_lib",
            true,
        )
        .unwrap()
    }

    fn compile_and_run(temp: &TempDir, source: &str) -> String {
        let source_path = temp.path("submission.rs");
        let executable_path = temp.path("submission");
        fs::write(&source_path, source).unwrap();
        let status = process::Command::new("rustc")
            .arg("--edition=2024")
            .arg(&source_path)
            .arg("-o")
            .arg(&executable_path)
            .status()
            .unwrap();
        assert!(status.success());
        let output = process::Command::new(executable_path).output().unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap()
    }

    #[test]
    fn bundles_nested_modules_and_rewrites_crate_paths() {
        let temp = TempDir::new("nested");
        temp.write("src/lib.rs", "pub const ROOT: i32 = 40;\npub mod outer;\n");
        temp.write(
            "src/outer.rs",
            "pub mod inner;\npub fn value() -> i32 { crate::ROOT + inner::value() }\n",
        );
        temp.write("src/outer/inner.rs", "pub fn value() -> i32 { 2 }\n");
        temp.write(
            "src/bin/solve.rs",
            "use demo_lib::outer;\nfn main() { println!(\"{}\", outer::value()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(output.contains("crate::__atcoder_bundle_demo_lib::ROOT"));
        assert!(output.contains("use crate::__atcoder_bundle_demo_lib::outer;"));
        assert_eq!(compile_and_run(&temp, &output), "42\n");
    }

    #[test]
    fn expands_include_macros_and_omits_test_items() {
        let temp = TempDir::new("include");
        temp.write(
            "src/lib.rs",
            "pub fn weights() -> Vec<u8> { include!(\"weights.rs\") }\n#[cfg(test)]\nmod tests { const TEST_ONLY: &[u8] = include_bytes!(\"large.bin\"); }\n",
        );
        temp.write("src/weights.rs", "vec![1, 2, 3]");
        temp.write("src/large.bin", "must not be bundled");
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{:?}\", demo_lib::weights()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(output.contains("vec![1, 2, 3]"));
        assert!(!output.contains("TEST_ONLY"));
        assert!(!output.contains("include!"));
        assert_eq!(compile_and_run(&temp, &output), "[1, 2, 3]\n");
    }

    #[test]
    fn resolves_modules_inside_included_sources() {
        let temp = TempDir::new("included-module");
        temp.write("src/lib.rs", "include!(\"generated/root.rs\");\n");
        temp.write("src/generated/root.rs", "pub mod child;\n");
        temp.write("src/generated/child.rs", "pub fn value() -> i32 { 12 }\n");
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{}\", demo_lib::child::value()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert_eq!(compile_and_run(&temp, &output), "12\n");
    }

    #[test]
    fn keeps_semicolons_for_expression_includes() {
        let temp = TempDir::new("expression-include");
        temp.write(
            "src/lib.rs",
            "pub const VALUE: i32 = include!(\"value.rs\");\n",
        );
        temp.write("src/value.rs", "7");
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{}\", demo_lib::VALUE); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(output.contains("pub const VALUE: i32 = 7;"));
        assert_eq!(compile_and_run(&temp, &output), "7\n");
    }

    #[test]
    fn removes_semicolon_after_include_item_in_inline_module() {
        let temp = TempDir::new("inline-module-include");
        temp.write("src/lib.rs", "pub fn value() -> i32 { 7 }\n");
        temp.write(
            "src/bin/solve.rs",
            "mod local { include!(\"local.rs\"); }\nfn main() { println!(\"{}\", local::value()); }\n",
        );
        temp.write("src/bin/local.rs", "pub fn value() -> i32 { 9 }\n");

        let output = bundle_fixture(&temp);
        assert!(!output.contains("pub fn value() -> i32 { 9 }\n;"));
        assert_eq!(compile_and_run(&temp, &output), "9\n");
    }

    #[test]
    fn preserves_semicolon_after_include_statement_in_function() {
        let temp = TempDir::new("function-statement-include");
        temp.write(
            "src/lib.rs",
            "pub fn value() -> i32 { include!(\"value.rs\"); 8 }\n",
        );
        temp.write("src/value.rs", "7");
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{}\", demo_lib::value()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(output.contains("{ 7; 8 }"));
        assert_eq!(compile_and_run(&temp, &output), "8\n");
    }

    #[test]
    fn strips_preceding_attributes_and_docs_with_test_items() {
        let temp = TempDir::new("test-attributes");
        temp.write(
            "src/lib.rs",
            "/// Only used by tests.\n#[repr(C)]\n#[cfg(test)]\nstruct TestOnly;\npub fn value() -> i32 { 3 }\n",
        );
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{}\", demo_lib::value()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(!output.contains("TestOnly"));
        assert!(!output.contains("Only used by tests"));
        assert_eq!(compile_and_run(&temp, &output), "3\n");
    }

    #[test]
    fn leaves_mod_text_in_literals_and_comments_alone() {
        let temp = TempDir::new("literals");
        temp.write(
            "src/lib.rs",
            "// mod imaginary;\npub fn message() -> &'static str { \"mod imaginary;\" }\n",
        );
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{}\", demo_lib::message()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert_eq!(compile_and_run(&temp, &output), "mod imaginary;\n");
    }

    #[test]
    fn preserves_raw_module_identifiers() {
        let temp = TempDir::new("raw-module-name");
        temp.write("src/lib.rs", "pub mod r#type;\n");
        temp.write("src/type.rs", "pub fn value() -> i32 { 9 }\n");
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{}\", demo_lib::r#type::value()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(output.contains("mod r#type {"));
        assert_eq!(compile_and_run(&temp, &output), "9\n");
    }

    #[test]
    fn rewrites_library_root_aliases() {
        let temp = TempDir::new("library-root-aliases");
        temp.write(
            "src/lib.rs",
            "pub const VALUE: i32 = 5;\nuse crate as module_root;\nextern crate self as self_root;\npub fn value() -> i32 { module_root::VALUE + self_root::VALUE }\n",
        );
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{}\", demo_lib::value()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(output.contains("use crate::__atcoder_bundle_demo_lib as module_root;"));
        assert!(output.contains("use crate::__atcoder_bundle_demo_lib as self_root;"));
        assert_eq!(compile_and_run(&temp, &output), "10\n");
    }

    #[test]
    fn rewrites_leading_absolute_entry_paths() {
        let temp = TempDir::new("absolute-entry-path");
        temp.write("src/lib.rs", "pub fn value() -> i32 { 8 }\n");
        temp.write(
            "src/bin/solve.rs",
            "use ::demo_lib::value;\nfn main() { println!(\"{}\", value()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(output.contains("use crate::__atcoder_bundle_demo_lib::value;"));
        assert_eq!(compile_and_run(&temp, &output), "8\n");
    }

    #[test]
    fn supports_include_str_and_include_bytes() {
        let temp = TempDir::new("include-literals");
        temp.write(
            "src/lib.rs",
            "pub fn text() -> &'static str { include_str!(\"message.txt\") }\npub fn bytes() -> &'static [u8] { include_bytes!(\"bytes.bin\") }\n",
        );
        temp.write("src/message.txt", "hello\nworld");
        temp.write("src/bytes.bin", "\u{0}\nA");
        temp.write(
            "src/bin/solve.rs",
            "fn main() { println!(\"{}:{:?}\", demo_lib::text(), demo_lib::bytes()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(!output.contains("include_str!"));
        assert!(!output.contains("include_bytes!"));
        assert_eq!(
            compile_and_run(&temp, &output),
            "hello\nworld:[0, 10, 65]\n"
        );
    }

    #[test]
    fn reports_missing_modules() {
        let temp = TempDir::new("missing-module");
        temp.write("src/lib.rs", "pub mod missing;\n");
        temp.write("src/bin/solve.rs", "fn main() {}\n");
        let error = bundle_submission(
            &temp.path("src/bin/solve.rs"),
            &temp.path("src/lib.rs"),
            "demo_lib",
            "__atcoder_bundle_demo_lib",
            true,
        )
        .unwrap_err();
        assert!(error.contains("Could not resolve mod missing"));
    }

    #[test]
    fn rejects_path_modules() {
        let temp = TempDir::new("path-module");
        temp.write("src/lib.rs", "#[path = \"other.rs\"]\npub mod other;\n");
        temp.write("src/other.rs", "pub fn value() -> i32 { 1 }\n");
        temp.write("src/bin/solve.rs", "fn main() {}\n");
        let error = bundle_submission(
            &temp.path("src/bin/solve.rs"),
            &temp.path("src/lib.rs"),
            "demo_lib",
            "__atcoder_bundle_demo_lib",
            true,
        )
        .unwrap_err();
        assert!(error.contains("#[path = ...]"));
    }

    #[test]
    fn rejects_macro_exports() {
        let temp = TempDir::new("macro-export");
        temp.write(
            "src/lib.rs",
            "#[macro_export]\nmacro_rules! exported { () => { 1 }; }\n",
        );
        temp.write("src/bin/solve.rs", "fn main() {}\n");
        let error = bundle_submission(
            &temp.path("src/bin/solve.rs"),
            &temp.path("src/lib.rs"),
            "demo_lib",
            "__atcoder_bundle_demo_lib",
            true,
        )
        .unwrap_err();
        assert!(error.contains("#[macro_export]"));
    }

    #[test]
    fn rejects_inner_docs_and_shebangs() {
        let docs = TempDir::new("inner-doc");
        docs.write("src/lib.rs", "pub fn value() -> i32 { 1 }\n");
        docs.write(
            "src/bin/solve.rs",
            "//! Submission documentation.\nfn main() { println!(\"{}\", demo_lib::value()); }\n",
        );
        let error = bundle_submission(
            &docs.path("src/bin/solve.rs"),
            &docs.path("src/lib.rs"),
            "demo_lib",
            "__atcoder_bundle_demo_lib",
            true,
        )
        .unwrap_err();
        assert!(error.contains("inner documentation"));

        let shebang = TempDir::new("shebang");
        shebang.write(
            "src/lib.rs",
            "#!/usr/bin/env rustx\npub fn value() -> i32 { 1 }\n",
        );
        shebang.write("src/bin/solve.rs", "fn main() {}\n");
        let error = bundle_submission(
            &shebang.path("src/bin/solve.rs"),
            &shebang.path("src/lib.rs"),
            "demo_lib",
            "__atcoder_bundle_demo_lib",
            true,
        )
        .unwrap_err();
        assert!(error.contains("Shebangs are not supported"));
    }

    #[test]
    fn allows_inner_docs_inside_entry_modules() {
        let temp = TempDir::new("entry-module-inner-doc");
        temp.write("src/lib.rs", "pub fn value() -> i32 { 1 }\n");
        temp.write(
            "src/bin/solve.rs",
            "mod local;\nfn main() { println!(\"{}\", local::value()); }\n",
        );
        temp.write(
            "src/bin/solve/local.rs",
            "//! Documentation for the local module.\npub fn value() -> i32 { 11 }\n",
        );

        let output = bundle_fixture(&temp);
        assert_eq!(compile_and_run(&temp, &output), "11\n");
    }

    #[test]
    fn allows_inner_attributes_inside_nested_modules() {
        let temp = TempDir::new("nested-inner-attribute");
        temp.write("src/lib.rs", "pub mod library_local;\n");
        temp.write(
            "src/library_local.rs",
            "#![allow(dead_code)]\npub fn value() -> i32 { 13 }\n",
        );
        temp.write(
            "src/bin/solve.rs",
            "mod local;\nfn main() { println!(\"{}:{}\", local::value(), demo_lib::library_local::value()); }\n",
        );
        temp.write(
            "src/bin/solve/local.rs",
            "#![allow(dead_code)]\npub fn value() -> i32 { 12 }\n",
        );

        let output = bundle_fixture(&temp);
        assert_eq!(compile_and_run(&temp, &output), "12:13\n");
    }

    #[test]
    fn reports_cyclic_includes() {
        let temp = TempDir::new("cyclic-include");
        temp.write("src/lib.rs", "include!(\"part.rs\");\n");
        temp.write("src/part.rs", "include!(\"lib.rs\");\n");
        temp.write("src/bin/solve.rs", "fn main() {}\n");
        let error = bundle_submission(
            &temp.path("src/bin/solve.rs"),
            &temp.path("src/lib.rs"),
            "demo_lib",
            "__atcoder_bundle_demo_lib",
            true,
        )
        .unwrap_err();
        assert!(error.contains("Cyclic source inclusion"));
    }

    #[test]
    fn extracts_leading_entry_inner_attributes() {
        let temp = TempDir::new("inner-attributes");
        temp.write("src/lib.rs", "pub fn value() -> i32 { 7 }\n");
        temp.write(
            "src/bin/solve.rs",
            "#![allow(unused_imports)]\nuse demo_lib::value;\nfn main() { println!(\"{}\", value()); }\n",
        );

        let output = bundle_fixture(&temp);
        assert!(output.starts_with("#![allow(unused_imports)]"));
        assert_eq!(compile_and_run(&temp, &output), "7\n");
    }

    #[test]
    fn enforces_configured_size_limit() {
        assert!(check_output_size("1234", Some(4)).is_ok());
        let error = check_output_size("12345", Some(4)).unwrap_err();
        assert!(error.contains("5 bytes"));
        assert!(check_output_size("12345", None).is_ok());
    }
}
