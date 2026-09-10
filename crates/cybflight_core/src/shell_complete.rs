//! Tab completion for the USB shell.
//!
//! Board-agnostic and allocation-free so the matching logic is
//! host-tested; the firmware owns the command table (it is cfg-gated on
//! firmware features) and the dynamic word lists (param names, mission
//! names, …) behind [`WordSource`].
//!
//! Grammar: each template is one space-separated command line, e.g.
//! `"blackbox record on"` or `"param set <param>"`. A `<…>` token is a
//! placeholder: it matches any word already typed in that position, and
//! when it is the word under the cursor its candidates come from
//! [`WordSource::visit`].

/// Capacity of the common-prefix buffer. Param names are capped at 48
/// bytes by `ParamName`; a longer candidate still counts as a match but
/// only its first `MAX_WORD` bytes can be inserted.
pub const MAX_WORD: usize = 48;

/// Dynamic candidates for placeholder tokens.
pub trait WordSource {
    /// Call `f` once per candidate for `placeholder` (e.g. `"<param>"`).
    /// Unknown placeholders yield nothing.
    fn visit(&self, placeholder: &str, f: &mut dyn FnMut(&str));
}

/// Result of [`complete`].
pub struct Completion {
    /// Distinct candidates for the word under the cursor.
    pub matches: usize,
    buf: [u8; MAX_WORD + 1],
    len: usize,
}

impl Completion {
    /// Bytes to append to the line: the candidates' common extension past
    /// the partial word, plus a separating space once the match is unique.
    /// Empty when ambiguous with nothing to extend — the caller lists
    /// the candidates instead.
    pub fn insert(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

fn is_placeholder(tok: &str) -> bool {
    tok.starts_with('<')
}

/// Split `line` into the finished words and the partial word under the
/// cursor (empty right after a space).
fn split(line: &str) -> (&str, &str) {
    let word_start = line
        .rfind(|c: char| c.is_ascii_whitespace())
        .map_or(0, |i| i + 1);
    line.split_at(word_start)
}

/// The template token that follows the finished words `done`, if the
/// template's leading tokens match them.
fn next_token<'t>(template: &'t str, done: &str) -> Option<&'t str> {
    let mut toks = template.split(' ');
    for w in done.split_ascii_whitespace() {
        let tok = toks.next()?;
        if !is_placeholder(tok) && tok != w {
            return None;
        }
    }
    toks.next()
}

/// Call `f` once per distinct candidate for the word under the cursor.
pub fn for_each_match(
    templates: &[&str],
    words: &dyn WordSource,
    line: &str,
    f: &mut dyn FnMut(&str),
) {
    let (done, partial) = split(line);
    for (i, t) in templates.iter().enumerate() {
        let Some(tok) = next_token(t, done) else {
            continue;
        };
        // Templates share prefixes ("led on" / "led off"): emit each
        // distinct next token once, at its first occurrence.
        if templates[..i]
            .iter()
            .any(|u| next_token(u, done) == Some(tok))
        {
            continue;
        }
        if is_placeholder(tok) {
            words.visit(tok, &mut |w| {
                if w.starts_with(partial) {
                    f(w)
                }
            });
        } else if tok.starts_with(partial) {
            f(tok);
        }
    }
}

/// Complete the word under the cursor at the end of `line`.
pub fn complete(templates: &[&str], words: &dyn WordSource, line: &str) -> Completion {
    let partial_len = split(line).1.len();
    let mut lcp = [0u8; MAX_WORD];
    let mut lcp_len = 0;
    let mut first_len = 0;
    let mut matches = 0;
    for_each_match(templates, words, line, &mut |c| {
        let c = c.as_bytes();
        if matches == 0 {
            first_len = c.len();
            lcp_len = c.len().min(MAX_WORD);
            lcp[..lcp_len].copy_from_slice(&c[..lcp_len]);
        } else {
            lcp_len = lcp[..lcp_len]
                .iter()
                .zip(c)
                .take_while(|(a, b)| a == b)
                .count();
        }
        matches += 1;
    });

    let mut out = Completion {
        matches,
        buf: [0; MAX_WORD + 1],
        len: 0,
    };
    if matches > 0 && lcp_len >= partial_len {
        let ext = &lcp[partial_len..lcp_len];
        out.buf[..ext.len()].copy_from_slice(ext);
        out.len = ext.len();
        // Only a fully inserted word earns the separator.
        if matches == 1 && lcp_len == first_len {
            out.buf[out.len] = b' ';
            out.len += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMDS: &[&str] = &[
        "help",
        "health",
        "rc",
        "rcstats",
        "stream <topic> on",
        "stream <topic> off",
        "param list",
        "param get <param>",
        "param set <param>",
        "reboot",
        "reboot --dfu",
    ];

    struct Words;
    impl WordSource for Words {
        fn visit(&self, placeholder: &str, f: &mut dyn FnMut(&str)) {
            let list: &[&str] = match placeholder {
                "<topic>" => &["att", "attcontrol", "imu1"],
                "<param>" => &["roll_kp", "roll_ki", "pitch_kp"],
                _ => &[],
            };
            list.iter().for_each(|w| f(w));
        }
    }

    fn run(line: &str) -> (usize, String) {
        let c = complete(CMDS, &Words, line);
        (
            c.matches,
            String::from_utf8(c.insert().to_vec()).unwrap(),
        )
    }

    fn listed(line: &str) -> Vec<String> {
        let mut v = Vec::new();
        for_each_match(CMDS, &Words, line, &mut |w| v.push(w.to_string()));
        v
    }

    #[test]
    fn unique_first_word_completes_with_space() {
        assert_eq!(run("hel"), (1, "p ".into()));
        assert_eq!(run("reb"), (1, "oot ".into()));
        // Already complete: just the separator.
        assert_eq!(run("help"), (1, " ".into()));
    }

    #[test]
    fn ambiguous_extends_to_common_prefix() {
        assert_eq!(run("h"), (2, "e".into()));
        assert_eq!(run("he"), (2, "".into()));
        // A full word that is also a prefix of another stays ambiguous.
        assert_eq!(run("rc"), (2, "".into()));
    }

    #[test]
    fn shared_prefixes_are_deduplicated() {
        assert_eq!(listed(""), ["help", "health", "rc", "rcstats", "stream", "param", "reboot"]);
        assert_eq!(listed("stream att "), ["on", "off"]);
        assert_eq!(listed("reboot "), ["--dfu"]);
    }

    #[test]
    fn placeholders_draw_from_word_source() {
        assert_eq!(run("stream a"), (2, "tt".into()));
        assert_eq!(run("stream i"), (1, "mu1 ".into()));
        assert_eq!(run("param set ro"), (2, "ll_k".into()));
        assert_eq!(run("param get pi"), (1, "tch_kp ".into()));
        assert_eq!(listed("param set "), ["roll_kp", "roll_ki", "pitch_kp"]);
    }

    #[test]
    fn placeholder_matches_any_typed_word() {
        // The topic isn't validated; the next token still completes.
        assert_eq!(run("stream bogus o"), (2, "".into()));
        assert_eq!(run("stream bogus of"), (1, "f ".into()));
    }

    #[test]
    fn nothing_past_the_template_or_unknown_word() {
        assert_eq!(run("xyz").0, 0);
        assert_eq!(run("param set roll_kp ").0, 0);
        assert_eq!(run("help ").0, 0);
        assert_eq!(run("param bogus ").0, 0);
    }

    #[test]
    fn surrounding_whitespace_is_tolerated() {
        assert_eq!(run("  hel"), (1, "p ".into()));
        assert_eq!(run("param   set  ro"), (2, "ll_k".into()));
        assert_eq!(listed("   ").len(), 7);
    }

    #[test]
    fn overlong_candidate_is_truncated_without_separator() {
        struct Long;
        impl WordSource for Long {
            fn visit(&self, _: &str, f: &mut dyn FnMut(&str)) {
                f(core::str::from_utf8(&[b'x'; MAX_WORD + 5]).unwrap());
            }
        }
        let c = complete(&["p <x>"], &Long, "p ");
        assert_eq!(c.matches, 1);
        assert_eq!(c.insert(), &[b'x'; MAX_WORD][..]);
    }
}
