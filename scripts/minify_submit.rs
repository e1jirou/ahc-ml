use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::process;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Token {
    kind: TokenKind,
    text: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Ident,
    Number,
    Lifetime,
    Literal,
    Punct,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Config {
    strip_debug: bool,
    submit_cfg: bool,
    shorten_names: bool,
}

fn main() {
    let (config, input_path, output_path) = match parse_args(env::args().skip(1).collect()) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            process::exit(2);
        }
    };

    let input = match read_all(&input_path) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("入力を読めませんでした: {input_path}: {err}");
            process::exit(1);
        }
    };

    let output = match minify_source(&input, config) {
        Ok(output) => output,
        Err(err) => {
            eprintln!("圧縮に失敗しました: {err}");
            process::exit(1);
        }
    };

    if let Err(err) = write_all(&output_path, output.as_bytes()) {
        eprintln!("出力を書けませんでした: {output_path}: {err}");
        process::exit(1);
    }

    if output_path != "-" {
        eprintln!(
            "圧縮しました: {} bytes -> {} bytes",
            input.len(),
            output.len()
        );
    }
}

fn parse_args(args: Vec<String>) -> Result<(Config, String, String), String> {
    let mut strip_debug = true;
    let mut submit_cfg = true;
    // Token-level renaming cannot reliably distinguish a local name from a
    // segment in an external path (for example, `std::error::Error`). Keep
    // names intact so the generated submission remains valid Rust.
    let shorten_names = false;
    let mut paths = Vec::new();

    for arg in args {
        match arg.as_str() {
            "--keep-debug" => strip_debug = false,
            "--keep-cfg" => submit_cfg = false,
            "--keep-names" => {}
            "-h" | "--help" => return Err(String::new()),
            _ if arg.starts_with('-') && arg != "-" => {
                return Err(format!("不明なオプションです: {arg}"));
            }
            _ => paths.push(arg),
        }
    }

    if paths.len() != 2 {
        return Err("入力ファイルと出力ファイルを指定してください。".to_string());
    }

    Ok((
        Config {
            strip_debug,
            submit_cfg,
            shorten_names,
        },
        paths.remove(0),
        paths.remove(0),
    ))
}

fn print_usage() {
    eprintln!("使い方: rustc scripts/minify_submit.rs -O -o scripts/minify_submit");
    eprintln!(
        "        scripts/minify_submit [--keep-debug] [--keep-cfg] [--keep-names] <input.rs|-> <output.rs|->"
    );
    eprintln!(
        "既定では submit 用 cfg と debug 系マクロを削除し、空白・コメントを除去します。識別子名は保持します。"
    );
}

fn read_all(path: &str) -> io::Result<String> {
    if path == "-" {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        Ok(input)
    } else {
        fs::read_to_string(path)
    }
}

fn write_all(path: &str, bytes: &[u8]) -> io::Result<()> {
    if path == "-" {
        io::stdout().write_all(bytes)
    } else {
        fs::write(path, bytes)
    }
}

fn minify_source(input: &str, config: Config) -> Result<String, String> {
    let mut tokens = lex(input)?;
    if config.submit_cfg {
        tokens = apply_submit_cfg(&tokens)?;
    }
    if config.strip_debug {
        tokens = strip_debug_macros(&tokens);
    }
    if config.shorten_names {
        tokens = strip_debug_derives(&tokens)?;
        tokens = shorten_submit_idents(&tokens);
        tokens = prepend_allow_warnings(tokens);
    }
    Ok(render_minified(&tokens))
}

fn lex(input: &str) -> Result<Vec<Token>, String> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;

    if bytes.starts_with(b"#!") && !bytes.starts_with(b"#![") {
        while i < bytes.len() && bytes[i] != b'\n' {
            i += 1;
        }
    }

    while i < bytes.len() {
        let b = bytes[i];

        if b.is_ascii_whitespace() {
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
            tokens.push(Token {
                kind: TokenKind::Literal,
                text: input[i..end].to_string(),
            });
            i = end;
            continue;
        }

        if b == b'"' {
            let end = quoted_end(bytes, i, b'"')?;
            tokens.push(Token {
                kind: TokenKind::Literal,
                text: input[i..end].to_string(),
            });
            i = end;
            continue;
        }

        if (b == b'b' || b == b'c') && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
            let end = quoted_end(bytes, i + 1, b'"')?;
            tokens.push(Token {
                kind: TokenKind::Literal,
                text: input[i..end].to_string(),
            });
            i = end;
            continue;
        }

        if (b == b'b' || b == b'c') && i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
            let end = char_or_lifetime_end(bytes, i + 1)?.0;
            tokens.push(Token {
                kind: TokenKind::Literal,
                text: input[i..end].to_string(),
            });
            i = end;
            continue;
        }

        if b == b'\'' {
            let (end, kind) = char_or_lifetime_end(bytes, i)?;
            tokens.push(Token {
                kind,
                text: input[i..end].to_string(),
            });
            i = end;
            continue;
        }

        if b.is_ascii_digit() {
            let end = number_end(bytes, i);
            tokens.push(Token {
                kind: TokenKind::Number,
                text: input[i..end].to_string(),
            });
            i = end;
            continue;
        }

        if is_ident_start(b) {
            let end = ident_end(bytes, i);
            tokens.push(Token {
                kind: TokenKind::Ident,
                text: input[i..end].to_string(),
            });
            i = end;
            continue;
        }

        let len = punct_len(bytes, i);
        tokens.push(Token {
            kind: TokenKind::Punct,
            text: input[i..i + len].to_string(),
        });
        i += len;
    }

    Ok(tokens)
}

fn starts_with(bytes: &[u8], i: usize, needle: &[u8]) -> bool {
    bytes.get(i..i + needle.len()) == Some(needle)
}

fn skip_block_comment(bytes: &[u8], mut i: usize) -> Result<usize, String> {
    debug_assert!(starts_with(bytes, i, b"/*"));
    i += 2;
    let mut depth = 1;

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

    Err("ブロックコメントが閉じていません。".to_string())
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

    Err("raw string literal が閉じていません。".to_string())
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
    debug_assert_eq!(bytes.get(i), Some(&quote));
    i += 1;

    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                i += 2;
            }
            b if b == quote => {
                return Ok(i + 1);
            }
            _ => {
                i += 1;
            }
        }
    }

    Err("文字列/文字リテラルが閉じていません。".to_string())
}

fn char_or_lifetime_end(bytes: &[u8], i: usize) -> Result<(usize, TokenKind), String> {
    debug_assert_eq!(bytes.get(i), Some(&b'\''));

    if i + 1 < bytes.len() && is_ident_start(bytes[i + 1]) {
        let ident_end = ident_end(bytes, i + 1);
        if bytes.get(ident_end) != Some(&b'\'') {
            return Ok((ident_end, TokenKind::Lifetime));
        }
    }

    quoted_end(bytes, i, b'\'').map(|end| (end, TokenKind::Literal))
}

fn number_end(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    i
}

fn ident_end(bytes: &[u8], i: usize) -> usize {
    if starts_with(bytes, i, b"r#") && i + 2 < bytes.len() && is_ident_start(bytes[i + 2]) {
        let mut j = i + 3;
        while j < bytes.len() && is_ident_continue(bytes[j]) {
            j += 1;
        }
        return j;
    }

    let mut j = i + 1;
    while j < bytes.len() && is_ident_continue(bytes[j]) {
        j += 1;
    }
    j
}

fn is_ident_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic()
}

fn is_ident_continue(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

fn punct_len(bytes: &[u8], i: usize) -> usize {
    const THREE: [&[u8]; 4] = [b"<<=", b">>=", b"...", b"..="];
    const TWO: [&[u8]; 21] = [
        b"::", b"->", b"=>", b"==", b"!=", b"<=", b">=", b"&&", b"||", b"+=", b"-=", b"*=", b"/=",
        b"%=", b"^=", b"&=", b"|=", b"<<", b">>", b"..", b"##",
    ];

    if THREE.iter().any(|punct| starts_with(bytes, i, punct)) {
        3
    } else if TWO.iter().any(|punct| starts_with(bytes, i, punct)) {
        2
    } else {
        1
    }
}

fn apply_submit_cfg(tokens: &[Token]) -> Result<Vec<Token>, String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if let Some((attr_end, keep)) = submit_cfg_attr(tokens, i)? {
            if keep {
                i = attr_end + 1;
            } else {
                i = skip_cfg_target(tokens, attr_end + 1)?;
            }
            continue;
        }

        out.push(tokens[i].clone());
        i += 1;
    }

    Ok(out)
}

fn submit_cfg_attr(tokens: &[Token], i: usize) -> Result<Option<(usize, bool)>, String> {
    if tokens.get(i).is_none_or(|token| token.text != "#")
        || tokens.get(i + 1).is_none_or(|token| token.text != "[")
        || tokens.get(i + 2).is_none_or(|token| token.text != "cfg")
        || tokens.get(i + 3).is_none_or(|token| token.text != "(")
    {
        return Ok(None);
    }

    let Some(close_paren) = matching_delimiter(tokens, i + 3) else {
        return Err("cfg 属性の括弧が閉じていません。".to_string());
    };
    if tokens
        .get(close_paren + 1)
        .is_none_or(|token| token.text != "]")
    {
        return Ok(None);
    }

    let expr: String = tokens[i + 4..close_paren]
        .iter()
        .map(|token| token.text.as_str())
        .collect();
    let keep = match expr.as_str() {
        r#"feature="submit""# | r#"not(feature="local")"# => true,
        r#"feature="local""# | r#"not(feature="submit")"# => false,
        _ => return Ok(None),
    };

    Ok(Some((close_paren + 1, keep)))
}

fn skip_cfg_target(tokens: &[Token], mut i: usize) -> Result<usize, String> {
    while let Some(attr_end) = outer_attr_end(tokens, i)? {
        i = attr_end + 1;
    }
    skip_item_or_statement(tokens, i)
}

fn outer_attr_end(tokens: &[Token], i: usize) -> Result<Option<usize>, String> {
    if tokens.get(i).is_none_or(|token| token.text != "#")
        || tokens.get(i + 1).is_none_or(|token| token.text != "[")
    {
        return Ok(None);
    }
    matching_delimiter(tokens, i + 1)
        .map(Some)
        .ok_or_else(|| "属性の角括弧が閉じていません。".to_string())
}

fn skip_item_or_statement(tokens: &[Token], i: usize) -> Result<usize, String> {
    if tokens.get(i).is_some_and(|token| token.text == "use") {
        return skip_until_semicolon(tokens, i);
    }

    let stop_at_comma = tokens
        .get(i)
        .is_none_or(|token| !is_item_start(&token.text));
    let return_after_block = tokens
        .get(i)
        .is_some_and(|token| is_block_target_start(&token.text));
    skip_item_or_statement_with_mode(tokens, i, stop_at_comma, return_after_block)
}

fn is_item_start(text: &str) -> bool {
    matches!(
        text,
        "const"
            | "enum"
            | "extern"
            | "fn"
            | "impl"
            | "macro_rules"
            | "mod"
            | "static"
            | "struct"
            | "trait"
            | "type"
            | "union"
            | "unsafe"
    )
}

fn is_block_target_start(text: &str) -> bool {
    text == "{" || is_item_start(text) || matches!(text, "for" | "if" | "loop" | "match" | "while")
}

fn skip_item_or_statement_with_mode(
    tokens: &[Token],
    i: usize,
    stop_at_comma: bool,
    return_after_block: bool,
) -> Result<usize, String> {
    let mut j = i;
    while j < tokens.len() {
        match tokens[j].text.as_str() {
            "(" | "[" => {
                j = matching_delimiter(tokens, j)
                    .ok_or_else(|| "区切り文字が閉じていません。".to_string())?
                    + 1;
            }
            "{" => {
                j = matching_delimiter(tokens, j)
                    .ok_or_else(|| "ブロックが閉じていません。".to_string())?
                    + 1;
                while j < tokens.len() && tokens[j].text == "else" {
                    j += 1;
                    if j < tokens.len() && tokens[j].text == "if" {
                        j = skip_item_or_statement_with_mode(
                            tokens,
                            j,
                            stop_at_comma,
                            return_after_block,
                        )?;
                        continue;
                    }
                    if j < tokens.len() && tokens[j].text == "{" {
                        j = matching_delimiter(tokens, j)
                            .ok_or_else(|| "else ブロックが閉じていません。".to_string())?
                            + 1;
                    }
                    break;
                }
                if return_after_block {
                    if j < tokens.len() && tokens[j].text == ";" {
                        j += 1;
                    }
                    return Ok(j);
                }
            }
            "," if stop_at_comma => return Ok(j + 1),
            ";" => return Ok(j + 1),
            _ => j += 1,
        }
    }
    Ok(j)
}

fn skip_until_semicolon(tokens: &[Token], mut i: usize) -> Result<usize, String> {
    while i < tokens.len() {
        match tokens[i].text.as_str() {
            "(" | "[" | "{" => {
                i = matching_delimiter(tokens, i)
                    .ok_or_else(|| "区切り文字が閉じていません。".to_string())?
                    + 1;
            }
            ";" => return Ok(i + 1),
            _ => i += 1,
        }
    }
    Ok(i)
}

fn strip_debug_macros(tokens: &[Token]) -> Vec<Token> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if let Some(debug_kind) = debug_macro_kind(tokens, i) {
            if let Some(close) = matching_delimiter(tokens, i + 2) {
                let next = close + 1;
                let previous = out.last();

                match debug_kind {
                    DebugMacroKind::Unit => {
                        if next < tokens.len()
                            && tokens[next].text == ";"
                            && is_statement_boundary(previous)
                        {
                            i = next + 1;
                            continue;
                        }

                        if next == tokens.len()
                            || tokens.get(next).is_some_and(|token| token.text == "}")
                        {
                            if is_statement_boundary(previous) {
                                i = next;
                                continue;
                            }
                        }

                        out.push(punct("("));
                        out.push(punct(")"));
                        i = next;
                        continue;
                    }
                    DebugMacroKind::Dbg => {
                        if next < tokens.len()
                            && tokens[next].text == ";"
                            && is_statement_boundary(previous)
                        {
                            i = next + 1;
                            continue;
                        }
                    }
                }
            }
        }

        out.push(tokens[i].clone());
        i += 1;
    }

    out
}

fn strip_debug_derives(tokens: &[Token]) -> Result<Vec<Token>, String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if is_derive_attr(tokens, i) {
            let close_paren = matching_delimiter(tokens, i + 3)
                .ok_or_else(|| "derive 属性の括弧が閉じていません。".to_string())?;
            let attr_end = close_paren + 1;
            let mut inner = Vec::new();
            let mut j = i + 4;
            while j < close_paren {
                if tokens[j].kind == TokenKind::Ident && tokens[j].text == "Debug" {
                    if j + 1 < close_paren && tokens[j + 1].text == "," {
                        j += 2;
                    } else if inner.last().is_some_and(|token: &Token| token.text == ",") {
                        inner.pop();
                        j += 1;
                    } else {
                        j += 1;
                    }
                    continue;
                }
                inner.push(tokens[j].clone());
                j += 1;
            }

            if !inner.is_empty() {
                out.push(punct("#"));
                out.push(punct("["));
                out.push(ident("derive"));
                out.push(punct("("));
                out.extend(inner);
                out.push(punct(")"));
                out.push(punct("]"));
            }
            i = attr_end + 1;
            continue;
        }

        out.push(tokens[i].clone());
        i += 1;
    }

    Ok(out)
}

fn is_derive_attr(tokens: &[Token], i: usize) -> bool {
    if tokens.get(i).is_none_or(|token| token.text != "#")
        || tokens.get(i + 1).is_none_or(|token| token.text != "[")
        || tokens.get(i + 2).is_none_or(|token| token.text != "derive")
        || tokens.get(i + 3).is_none_or(|token| token.text != "(")
    {
        return false;
    }

    let Some(close_paren) = matching_delimiter(tokens, i + 3) else {
        return false;
    };
    tokens
        .get(close_paren + 1)
        .is_some_and(|token| token.text == "]")
}

fn shorten_submit_idents(tokens: &[Token]) -> Vec<Token> {
    let declared = collect_declared_idents(tokens);
    let counts = identifier_counts(tokens);
    let reserved = reserved_idents();
    let mut used = all_ident_texts(tokens);

    let mut candidates: Vec<_> = declared
        .into_iter()
        .filter_map(|name| {
            let count = *counts.get(name.as_str())?;
            if is_rename_candidate(&name, count, &reserved) {
                Some((name, count))
            } else {
                None
            }
        })
        .collect();
    candidates.sort_unstable_by(|(a_name, a_count), (b_name, b_count)| {
        let a_score = a_name.len() * *a_count;
        let b_score = b_name.len() * *b_count;
        b_score.cmp(&a_score).then_with(|| a_name.cmp(b_name))
    });

    let mut map = HashMap::new();
    let mut next_name_index = 0;
    for (name, count) in candidates {
        let short = next_short_name(&mut next_name_index, &mut used, &reserved);
        if name.len() <= short.len() || (name.len() - short.len()) * count == 0 {
            continue;
        }
        used.insert(short.clone());
        map.insert(name, short);
    }

    tokens
        .iter()
        .map(|token| {
            if token.kind == TokenKind::Ident {
                if let Some(short) = map.get(token.text.as_str()) {
                    return Token {
                        kind: token.kind,
                        text: short.clone(),
                    };
                }
            }
            token.clone()
        })
        .collect()
}

fn prepend_allow_warnings(mut tokens: Vec<Token>) -> Vec<Token> {
    let mut out = vec![
        punct("#"),
        punct("!"),
        punct("["),
        ident("allow"),
        punct("("),
        ident("warnings"),
        punct(")"),
        punct("]"),
    ];
    out.append(&mut tokens);
    out
}

fn collect_declared_idents(tokens: &[Token]) -> HashSet<String> {
    let mut ids = HashSet::new();
    let mut i = 0;

    while i < tokens.len() {
        match tokens[i].text.as_str() {
            "const" | "static" | "type" | "trait" => {
                if let Some(name_index) = next_ident_index(tokens, i + 1) {
                    add_declared_ident(&mut ids, &tokens[name_index].text);
                }
            }
            "fn" => {
                if let Some(name_index) = next_ident_index(tokens, i + 1) {
                    add_declared_ident(&mut ids, &tokens[name_index].text);
                    if let Some(open) = find_next_text(tokens, name_index + 1, "(") {
                        if let Some(close) = matching_delimiter(tokens, open) {
                            collect_function_params(tokens, open + 1, close, &mut ids);
                            i = close;
                        }
                    }
                }
            }
            "struct" => {
                if let Some(name_index) = next_ident_index(tokens, i + 1) {
                    add_declared_ident(&mut ids, &tokens[name_index].text);
                    if let Some(open) = find_item_body(tokens, name_index + 1) {
                        if let Some(close) = matching_delimiter(tokens, open) {
                            collect_struct_fields(tokens, open + 1, close, &mut ids);
                            i = close;
                        }
                    }
                }
            }
            "enum" => {
                if let Some(name_index) = next_ident_index(tokens, i + 1) {
                    add_declared_ident(&mut ids, &tokens[name_index].text);
                    if let Some(open) = find_item_body(tokens, name_index + 1) {
                        if let Some(close) = matching_delimiter(tokens, open) {
                            collect_enum_variants(tokens, open + 1, close, &mut ids);
                            i = close;
                        }
                    }
                }
            }
            "let" => {
                let end = find_pattern_end(tokens, i + 1, &["=", ":", ";"]);
                collect_pattern_idents(tokens, i + 1, end, &mut ids);
            }
            "for" => {
                let end = find_pattern_end(tokens, i + 1, &["in"]);
                collect_pattern_idents(tokens, i + 1, end, &mut ids);
            }
            _ => {}
        }
        i += 1;
    }

    ids
}

fn next_ident_index(tokens: &[Token], mut i: usize) -> Option<usize> {
    while i < tokens.len() {
        if tokens[i].kind == TokenKind::Ident {
            return Some(i);
        }
        if matches!(tokens[i].text.as_str(), ";" | "{" | "(") {
            return None;
        }
        i += 1;
    }
    None
}

fn find_next_text(tokens: &[Token], mut i: usize, text: &str) -> Option<usize> {
    while i < tokens.len() {
        if tokens[i].text == text {
            return Some(i);
        }
        if matches!(tokens[i].text.as_str(), ";" | "{") {
            return None;
        }
        i += 1;
    }
    None
}

fn find_item_body(tokens: &[Token], mut i: usize) -> Option<usize> {
    while i < tokens.len() {
        match tokens[i].text.as_str() {
            "{" => return Some(i),
            ";" => return None,
            "(" | "[" => {
                i = matching_delimiter(tokens, i)? + 1;
            }
            _ => i += 1,
        }
    }
    None
}

fn collect_function_params(
    tokens: &[Token],
    mut start: usize,
    end: usize,
    ids: &mut HashSet<String>,
) {
    while start < end {
        let comma = find_top_level_token(tokens, start, end, &[","]).unwrap_or(end);
        let pattern_end = find_top_level_token(tokens, start, comma, &[":"]).unwrap_or(comma);
        collect_pattern_idents(tokens, start, pattern_end, ids);
        start = comma + 1;
    }
}

fn collect_struct_fields(tokens: &[Token], mut i: usize, end: usize, ids: &mut HashSet<String>) {
    while i < end {
        if let Some(attr_end) = outer_attr_end(tokens, i).ok().flatten() {
            i = attr_end + 1;
            continue;
        }
        if tokens[i].text == "," {
            i += 1;
            continue;
        }
        if tokens[i].text == "pub" {
            i += 1;
            if i < end && tokens[i].text == "(" {
                if let Some(close) = matching_delimiter(tokens, i) {
                    i = close + 1;
                }
            }
            continue;
        }
        if tokens[i].kind == TokenKind::Ident
            && tokens.get(i + 1).is_some_and(|token| token.text == ":")
        {
            add_declared_ident(ids, &tokens[i].text);
        }
        i = skip_to_next_top_level_comma(tokens, i, end);
    }
}

fn collect_enum_variants(tokens: &[Token], mut i: usize, end: usize, ids: &mut HashSet<String>) {
    while i < end {
        if let Some(attr_end) = outer_attr_end(tokens, i).ok().flatten() {
            i = attr_end + 1;
            continue;
        }
        if tokens[i].text == "," {
            i += 1;
            continue;
        }
        if tokens[i].kind == TokenKind::Ident {
            add_declared_ident(ids, &tokens[i].text);
        }
        i = skip_to_next_top_level_comma(tokens, i, end);
    }
}

fn skip_to_next_top_level_comma(tokens: &[Token], mut i: usize, end: usize) -> usize {
    while i < end {
        match tokens[i].text.as_str() {
            "(" | "[" | "{" => {
                if let Some(close) = matching_delimiter(tokens, i) {
                    i = close + 1;
                } else {
                    return end;
                }
            }
            "," => return i + 1,
            _ => i += 1,
        }
    }
    end
}

fn find_pattern_end(tokens: &[Token], start: usize, stops: &[&str]) -> usize {
    find_top_level_token(tokens, start, tokens.len(), stops).unwrap_or(tokens.len())
}

fn find_top_level_token(
    tokens: &[Token],
    mut i: usize,
    end: usize,
    stops: &[&str],
) -> Option<usize> {
    while i < end {
        match tokens[i].text.as_str() {
            "(" | "[" | "{" => {
                i = matching_delimiter(tokens, i)? + 1;
            }
            text if stops.contains(&text) => return Some(i),
            _ => i += 1,
        }
    }
    None
}

fn collect_pattern_idents(tokens: &[Token], start: usize, end: usize, ids: &mut HashSet<String>) {
    for i in start..end {
        if tokens[i].kind != TokenKind::Ident {
            continue;
        }

        let text = tokens[i].text.as_str();
        let previous = i.checked_sub(1).and_then(|index| tokens.get(index));
        let next = tokens.get(i + 1);
        if matches!(text, "mut" | "ref" | "self" | "Self")
            || text.starts_with("r#")
            || previous.is_some_and(|token| token.text == "::")
            || next.is_some_and(|token| matches!(token.text.as_str(), "::" | "(" | "{"))
        {
            continue;
        }
        add_declared_ident(ids, text);
    }
}

fn add_declared_ident(ids: &mut HashSet<String>, name: &str) {
    if name != "_" && !name.starts_with("r#") {
        ids.insert(name.to_string());
    }
}

fn identifier_counts(tokens: &[Token]) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        if token.kind == TokenKind::Ident {
            *counts.entry(token.text.as_str()).or_insert(0) += 1;
        }
    }
    counts
}

fn all_ident_texts(tokens: &[Token]) -> HashSet<String> {
    tokens
        .iter()
        .filter(|token| token.kind == TokenKind::Ident)
        .map(|token| token.text.clone())
        .collect()
}

fn is_rename_candidate(name: &str, count: usize, reserved: &HashSet<&'static str>) -> bool {
    count >= 2
        && name.len() > 1
        && !name.starts_with('_')
        && !name.starts_with("r#")
        && !reserved.contains(name)
}

fn next_short_name(
    next_name_index: &mut usize,
    used: &mut HashSet<String>,
    reserved: &HashSet<&'static str>,
) -> String {
    loop {
        let name = short_name(*next_name_index);
        *next_name_index += 1;
        if !used.contains(&name) && !reserved.contains(name.as_str()) {
            return name;
        }
    }
}

fn short_name(index: usize) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

    if index < CHARS.len() {
        return (CHARS[index] as char).to_string();
    }

    let mut n = index - CHARS.len();
    let mut tail = Vec::new();
    loop {
        tail.push(DIGITS[n % DIGITS.len()] as char);
        n /= DIGITS.len();
        if n == 0 {
            break;
        }
    }
    let mut name = String::from("_");
    for c in tail.iter().rev() {
        name.push(*c);
    }
    name
}

fn reserved_idents() -> HashSet<&'static str> {
    [
        "_",
        "abstract",
        "as",
        "async",
        "await",
        "become",
        "bool",
        "box",
        "break",
        "char",
        "Clone",
        "const",
        "continue",
        "Copy",
        "crate",
        "Debug",
        "do",
        "dyn",
        "else",
        "enum",
        "Eq",
        "Err",
        "exit",
        "extern",
        "f32",
        "f64",
        "false",
        "final",
        "fn",
        "for",
        "i128",
        "i16",
        "i32",
        "i64",
        "i8",
        "if",
        "impl",
        "in",
        "input",
        "isize",
        "Iterator",
        "let",
        "loop",
        "macro",
        "main",
        "match",
        "mod",
        "move",
        "mut",
        "None",
        "Ok",
        "Option",
        "Ord",
        "override",
        "PartialEq",
        "PartialOrd",
        "priv",
        "proconio",
        "pub",
        "ref",
        "Result",
        "return",
        "rustfmt",
        "Self",
        "self",
        "Some",
        "static",
        "std",
        "str",
        "String",
        "struct",
        "super",
        "trait",
        "true",
        "try",
        "type",
        "typeof",
        "u128",
        "u16",
        "u32",
        "u64",
        "u8",
        "union",
        "unsafe",
        "unsized",
        "use",
        "usize",
        "Vec",
        "virtual",
        "where",
        "while",
        "yield",
        "abs",
        "any",
        "as_bytes",
        "as_slice",
        "clamp",
        "clear",
        "cmp",
        "collect",
        "contains",
        "div_ceil",
        "enumerate",
        "fill",
        "filter_map",
        "get",
        "into_iter",
        "is_empty",
        "is_infinite",
        "is_sign_positive",
        "iter",
        "last",
        "last_mut",
        "len",
        "map",
        "map_or",
        "max",
        "min",
        "min_by",
        "new",
        "partial_cmp",
        "partition_point",
        "pop",
        "print",
        "println",
        "push",
        "reserve",
        "rev",
        "round",
        "saturating_sub",
        "select_nth_unstable_by",
        "sort_unstable_by",
        "sort_unstable_by_key",
        "sum",
        "swap",
        "then_with",
        "truncate",
        "try_into",
        "unreachable",
        "unwrap",
        "vec",
        "with_capacity",
    ]
    .iter()
    .copied()
    .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DebugMacroKind {
    Unit,
    Dbg,
}

fn debug_macro_kind(tokens: &[Token], i: usize) -> Option<DebugMacroKind> {
    let name = tokens.get(i)?;
    let bang = tokens.get(i + 1)?;
    let open = tokens.get(i + 2)?;

    if name.kind != TokenKind::Ident || bang.text != "!" || !is_open_delimiter(&open.text) {
        return None;
    }

    match name.text.as_str() {
        "debug_assert" | "debug_assert_eq" | "debug_assert_ne" | "eprintln" => {
            Some(DebugMacroKind::Unit)
        }
        "dbg" => Some(DebugMacroKind::Dbg),
        _ => None,
    }
}

fn matching_delimiter(tokens: &[Token], open: usize) -> Option<usize> {
    let close_text = match tokens.get(open)?.text.as_str() {
        "(" => ")",
        "[" => "]",
        "{" => "}",
        _ => return None,
    };
    let mut stack = vec![close_text];

    for (i, token) in tokens.iter().enumerate().skip(open + 1) {
        match token.text.as_str() {
            "(" => stack.push(")"),
            "[" => stack.push("]"),
            "{" => stack.push("}"),
            ")" | "]" | "}" => {
                if Some(&token.text.as_str()) != stack.last() {
                    return None;
                }
                stack.pop();
                if stack.is_empty() {
                    return Some(i);
                }
            }
            _ => {}
        }
    }

    None
}

fn is_open_delimiter(text: &str) -> bool {
    text == "(" || text == "[" || text == "{"
}

fn is_statement_boundary(token: Option<&Token>) -> bool {
    token.is_none_or(|token| matches!(token.text.as_str(), "{" | "}" | ";" | "=>" | ","))
}

fn punct(text: &str) -> Token {
    Token {
        kind: TokenKind::Punct,
        text: text.to_string(),
    }
}

fn ident(text: &str) -> Token {
    Token {
        kind: TokenKind::Ident,
        text: text.to_string(),
    }
}

fn render_minified(tokens: &[Token]) -> String {
    let mut output = String::new();
    let mut previous: Option<&Token> = None;

    for token in tokens {
        if previous.is_some_and(|previous| needs_space(previous, token)) {
            output.push(' ');
        }
        output.push_str(&token.text);
        previous = Some(token);
    }

    output
}

fn needs_space(left: &Token, right: &Token) -> bool {
    if is_word_like(left) && is_word_like(right) {
        return true;
    }

    if left.kind == TokenKind::Lifetime && right.kind == TokenKind::Ident {
        return true;
    }

    // `<-` is reserved by Rust, rather than lexing as `<` followed by unary
    // `-`. The whitespace is therefore required for comparisons with a
    // negative value.
    if left.text == "<" && right.text == "-" {
        return true;
    }

    if is_word_like(left) && right.text == "#" {
        return true;
    }

    if left.text == "/" && (right.text == "/" || right.text == "*") {
        return true;
    }

    false
}

fn is_word_like(token: &Token) -> bool {
    matches!(
        token.kind,
        TokenKind::Ident | TokenKind::Number | TokenKind::Lifetime | TokenKind::Literal
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minify(input: &str) -> String {
        minify_source(
            input,
            Config {
                strip_debug: true,
                submit_cfg: true,
                shorten_names: false,
            },
        )
        .unwrap()
    }

    fn minify_with_short_names(input: &str) -> String {
        minify_source(
            input,
            Config {
                strip_debug: true,
                submit_cfg: true,
                shorten_names: true,
            },
        )
        .unwrap()
    }

    #[test]
    fn removes_comments_but_keeps_literals() {
        let input = r##"
            fn main() {
                let s = "http://example.com/*x*/";
                let r = r#"text // still literal"#;
                /* block
                   comment */
                println!("{}{}", s, r); // line comment
            }
        "##;
        assert_eq!(
            minify(input),
            r##"fn main(){let s="http://example.com/*x*/";let r=r#"text // still literal"#;println!("{}{}",s,r);}"##
        );
    }

    #[test]
    fn keeps_required_spaces() {
        let input = r#"
            fn main() {
                let mut x = 1u32;
                let y: &'static str = "a";
                if x as usize > 0 { x += 1; }
                let _ = y;
            }
        "#;
        assert_eq!(
            minify(input),
            r#"fn main(){let mut x=1u32;let y:&'static str="a";if x as usize>0{x+=1;}let _=y;}"#
        );
    }

    #[test]
    fn keeps_space_before_negative_value_after_less_than() {
        let input = "fn main(){if value < -9 {}}";
        assert_eq!(minify(input), "fn main(){if value< -9{}}");
    }

    #[test]
    fn command_line_defaults_to_safe_name_preservation() {
        let (config, _, _) =
            parse_args(vec!["input.rs".to_string(), "output.rs".to_string()]).unwrap();
        assert!(!config.shorten_names);
    }

    #[test]
    fn strips_debug_statement_macros() {
        let input = r#"
            fn main() {
                debug_assert!(true);
                debug_assert_eq!(1, 1);
                eprintln!("debug");
                let x = dbg!(3);
                dbg!(x);
                println!("{}", x);
            }
        "#;
        assert_eq!(
            minify(input),
            r#"fn main(){let x=dbg!(3);println!("{}",x);}"#
        );
    }

    #[test]
    fn replaces_unit_debug_macro_in_expression_position() {
        let input = r#"
            fn main() {
                let _ = eprintln!("debug");
                let _ = debug_assert!(true);
            }
        "#;
        assert_eq!(minify(input), r#"fn main(){let _=();let _=();}"#);
    }

    #[test]
    fn supports_nested_block_comments() {
        let input = "fn main(){let a=1;/* a /* nested */ comment */let b=2;}";
        assert_eq!(minify(input), "fn main(){let a=1;let b=2;}");
    }

    #[test]
    fn applies_submit_cfg() {
        let input = r#"
            #[cfg(feature = "local")]
            fn local_only() {}
            #[cfg(feature = "local")]
            fn local_generic() -> Map<A, (B, C)> {}
            #[cfg(feature = "submit")]
            fn submit_only() {}
            fn main() {
                #[cfg(feature = "local")]
                let x = 1;
                #[cfg(feature = "submit")]
                let x = 2;
                println!("{}", x);
            }
        "#;
        assert_eq!(
            minify(input),
            r#"fn submit_only(){}fn main(){let x=2;println!("{}",x);}"#
        );
    }

    #[test]
    fn removes_cfg_fields_without_eating_delimiters() {
        let input = r#"
            struct Result {
                output: f64,
                #[cfg(feature = "local")]
                rollback: usize,
                action: usize,
            }
            fn make() {
                sink(Result {
                    output: 0.0,
                    #[cfg(feature = "local")]
                    rollback: 1,
                    action: 2,
                });
            }
        "#;
        assert_eq!(
            minify(input),
            "struct Result{output:f64,action:usize,}fn make(){sink(Result{output:0.0,action:2,});}"
        );
    }

    #[test]
    fn removes_cfg_let_with_block_expression() {
        let input = r#"
            fn main() {
                prepare();
                #[cfg(feature = "local")]
                let x = if cond() { 1 } else { 2 };
                finish();
            }
        "#;
        assert_eq!(minify(input), "fn main(){prepare();finish();}");
    }

    #[test]
    fn removes_cfg_block_without_eating_outer_statement() {
        let input = r#"
            fn main() {
                if cond() {
                    keep();
                    #[cfg(feature = "local")]
                    {
                        local();
                    }
                }
                finish();
            }
        "#;
        assert_eq!(minify(input), "fn main(){if cond(){keep();}finish();}");
    }

    #[test]
    fn strips_debug_from_derives() {
        let input = r#"
            #[derive(Debug, Clone, Copy)]
            struct LongName {
                long_field: usize,
            }
            fn main() {
                let _ = LongName { long_field: 1 };
            }
        "#;
        let output = minify_with_short_names(input);
        assert!(!output.contains("Debug"));
        assert!(output.contains("derive(Clone,Copy)"));
    }

    #[test]
    fn shortens_declared_submit_identifiers() {
        let input = r#"
            struct LongStruct {
                long_field: usize,
            }
            enum LongEnum {
                LongVariant,
            }
            fn helper_function(mut long_value: usize) -> usize {
                let temporary_value = long_value + 1;
                temporary_value
            }
            fn main() {
                let item = LongStruct {
                    long_field: helper_function(1),
                };
                if let LongEnum::LongVariant = LongEnum::LongVariant {
                    println!("{}", item.long_field);
                }
            }
        "#;
        let output = minify_with_short_names(input);
        assert!(!output.contains("LongStruct"));
        assert!(!output.contains("LongVariant"));
        assert!(!output.contains("helper_function"));
        assert!(!output.contains("temporary_value"));
        assert!(!output.contains("long_field"));
        assert!(output.contains("fn main()"));
        assert!(output.contains("println!"));
    }
}
