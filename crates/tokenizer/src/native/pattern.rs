//! The pre-tokenizer patterns of byte-level BPE tokenizers, matched natively.
//!
//! The patterns in use (GPT-2, Qwen, GLM, DeepSeek, MiniMax and their kin)
//! draw on a small regex subset: alternation, concatenation, character
//! classes built from Unicode categories, `\s`, ranges and literals, greedy
//! `?` `*` `+` `{m,n}`, a case-insensitive group of literal alternatives for
//! the English clitics, and the lookahead `(?!\S)`. This module parses that
//! subset and matches it the way the engine behind `tokenizers`' `Split`
//! does: leftmost-first alternation, greedy quantifiers that back off one
//! repetition at a time. The engine also yields empty matches, and `Split`
//! cuts the text at them; this matcher does not reproduce that, so a pattern
//! that can match the empty string is declined like everything else outside
//! the subset, and the caller keeps the `tokenizers` pipeline for it.
//!
//! Character categories come from [`super::unicode`], generated from that
//! same engine, so `\p{L}` or `\s` cannot mean one thing here and another
//! there.

use super::unicode;

/// A set of characters: Unicode categories, code point ranges, negation.
#[derive(Debug, Clone, Default)]
struct Class {
    negated: bool,
    categories: u16,
    ranges: Vec<(u32, u32)>,
    /// Membership of U+0000..=U+007F, resolved once.
    ascii: u128,
}

const CAT_LETTER: u16 = 1 << 0;
const CAT_MARK: u16 = 1 << 1;
const CAT_NUMBER: u16 = 1 << 2;
const CAT_PUNCTUATION: u16 = 1 << 3;
const CAT_SYMBOL: u16 = 1 << 4;
const CAT_UPPERCASE: u16 = 1 << 5;
const CAT_LOWERCASE: u16 = 1 << 6;
const CAT_TITLECASE: u16 = 1 << 7;
const CAT_MODIFIER_LETTER: u16 = 1 << 8;
const CAT_OTHER_LETTER: u16 = 1 << 9;
const CAT_SPACE: u16 = 1 << 10;

/// Pairs of ASCII letters that are the full case folding of one character
/// (ß ẞ: ss; ﬀ: ff; ﬁ: fi; ﬂ: fl; ﬃ ﬄ: ffi, ffl; ﬅ ﬆ: st), which the engine
/// matches under `(?i:)` and this matcher does not.
const MULTI_CHARACTER_FOLDS: [[char; 2]; 5] =
    [['s', 's'], ['s', 't'], ['f', 'f'], ['f', 'i'], ['f', 'l']];

const CATEGORY_TABLES: [(u16, &[(u32, u32)]); 11] = [
    (CAT_LETTER, unicode::LETTER),
    (CAT_MARK, unicode::MARK),
    (CAT_NUMBER, unicode::NUMBER),
    (CAT_PUNCTUATION, unicode::PUNCTUATION),
    (CAT_SYMBOL, unicode::SYMBOL),
    (CAT_UPPERCASE, unicode::UPPERCASE),
    (CAT_LOWERCASE, unicode::LOWERCASE),
    (CAT_TITLECASE, unicode::TITLECASE),
    (CAT_MODIFIER_LETTER, unicode::MODIFIER_LETTER),
    (CAT_OTHER_LETTER, unicode::OTHER_LETTER),
    (CAT_SPACE, unicode::SPACE),
];

#[inline]
fn in_ranges(ranges: &[(u32, u32)], code: u32) -> bool {
    let index = ranges.partition_point(|&(_, end)| end < code);
    ranges.get(index).is_some_and(|&(start, _)| start <= code)
}

impl Class {
    fn finish(mut self) -> Self {
        let mut ascii = 0u128;
        for code in 0u32..128 {
            if self.contains_slow(code) {
                ascii |= 1 << code;
            }
        }
        self.ascii = ascii;
        self
    }

    fn contains_slow(&self, code: u32) -> bool {
        let mut hit = in_ranges(&self.ranges, code);
        if !hit && self.categories != 0 {
            hit = CATEGORY_TABLES
                .iter()
                .any(|&(bit, table)| self.categories & bit != 0 && in_ranges(table, code));
        }
        hit != self.negated
    }

    #[inline]
    fn contains(&self, c: char) -> bool {
        let code = u32::from(c);
        if code < 128 {
            (self.ascii >> code) & 1 == 1
        } else {
            self.contains_slow(code)
        }
    }
}

#[derive(Debug, Clone)]
enum Node {
    /// One character of the class.
    Class(usize),
    /// Greedy repetition of one class character, `min..=max` times.
    ClassRepeat {
        class: usize,
        min: u32,
        max: u32,
    },
    /// Greedy repetition of a group, `min..=max` times.
    Repeat {
        node: Box<Node>,
        min: u32,
        max: u32,
    },
    Concat(Vec<Node>),
    /// Alternatives tried in order; the first that lets the rest match wins.
    Alt(Vec<Node>),
    /// `(?!...)` over one class: succeeds when the next character is not in it.
    NotAhead(usize),
}

/// A parsed pattern.
#[derive(Debug, Clone)]
pub(crate) struct Pattern {
    classes: Vec<Class>,
    root: Node,
}

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    classes: Vec<Class>,
    case_insensitive: bool,
}

impl Pattern {
    /// `None` when the pattern uses anything outside the supported subset.
    pub(crate) fn parse(pattern: &str) -> Option<Self> {
        let mut parser = Parser {
            chars: pattern.chars().peekable(),
            classes: Vec::new(),
            case_insensitive: false,
        };
        let root = parser.parse_alternation()?;
        if parser.chars.next().is_some() {
            return None;
        }
        if can_match_empty(&root) {
            return None;
        }
        Some(Self {
            classes: parser.classes,
            root,
        })
    }

    /// The matches in `text`, leftmost first, each continuing after the last.
    pub(crate) fn find_iter<'p, 't>(&'p self, text: &'t str) -> Matches<'p, 't> {
        Matches {
            pattern: self,
            text,
            pos: 0,
        }
    }

    /// The end of the match that starts exactly at `pos`, if any.
    fn match_at(&self, text: &str, pos: usize) -> Option<usize> {
        let mut end = None;
        self.step(&self.root, text, pos, &mut |p| {
            end = Some(p);
            true
        });
        end
    }

    fn step(&self, node: &Node, text: &str, pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
        match node {
            Node::Class(class) => match next_char(text, pos) {
                Some((c, len)) if self.classes[*class].contains(c) => k(pos + len),
                _ => false,
            },
            Node::ClassRepeat { class, min, max } => {
                let class = &self.classes[*class];
                let mut count = 0u32;
                let mut p = pos;
                while count < *max {
                    match next_char(text, p) {
                        Some((c, len)) if class.contains(c) => {
                            p += len;
                            count += 1;
                        }
                        _ => break,
                    }
                }
                loop {
                    if count < *min {
                        return false;
                    }
                    if k(p) {
                        return true;
                    }
                    if count == *min {
                        return false;
                    }
                    p -= 1;
                    while !text.is_char_boundary(p) {
                        p -= 1;
                    }
                    count -= 1;
                }
            }
            Node::Repeat { node, min, max } => self.repeat(node, *min, *max, 0, text, pos, k),
            Node::Concat(nodes) => self.concat(nodes, text, pos, k),
            Node::Alt(alternatives) => alternatives
                .iter()
                .any(|alternative| self.step(alternative, text, pos, k)),
            Node::NotAhead(class) => match next_char(text, pos) {
                Some((c, _)) if self.classes[*class].contains(c) => false,
                _ => k(pos),
            },
        }
    }

    fn concat(
        &self,
        nodes: &[Node],
        text: &str,
        pos: usize,
        k: &mut dyn FnMut(usize) -> bool,
    ) -> bool {
        match nodes.split_first() {
            None => k(pos),
            Some((first, rest)) => self.step(first, text, pos, &mut |p| {
                self.concat(rest, text, p, &mut *k)
            }),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the recursion carries its bounds"
    )]
    fn repeat(
        &self,
        node: &Node,
        min: u32,
        max: u32,
        count: u32,
        text: &str,
        pos: usize,
        k: &mut dyn FnMut(usize) -> bool,
    ) -> bool {
        if count < max
            && self.step(node, text, pos, &mut |p| {
                p != pos && self.repeat(node, min, max, count + 1, text, p, &mut *k)
            })
        {
            return true;
        }
        count >= min && k(pos)
    }
}

/// Iterator over the matches of a [`Pattern`] in a text.
pub(crate) struct Matches<'p, 't> {
    pattern: &'p Pattern,
    text: &'t str,
    pos: usize,
}

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        while self.pos < self.text.len() {
            let start = self.pos;
            match self.pattern.match_at(self.text, start) {
                Some(end) if end > start => {
                    self.pos = end;
                    return Some((start, end));
                }
                _ => {
                    // No match starts here: move on by one character. (A
                    // pattern that can match empty is declined at parse time,
                    // so an empty match does not occur; it would count as no
                    // match.)
                    let (_, len) = next_char(self.text, start)?;
                    self.pos = start + len;
                }
            }
        }
        None
    }
}

/// Whether `node` can match the empty string. The engine yields empty
/// matches (all but one right after the previous match), and `Split` cuts the
/// text at them: `\p{N}*` over "ab1" gives the pieces "a", "b", "1". This
/// matcher does not reproduce that, so a pattern whose root can match empty is
/// declined at parse time and no stage ever sees an empty match.
fn can_match_empty(node: &Node) -> bool {
    match node {
        Node::Class(_) => false,
        Node::ClassRepeat { min, .. } => *min == 0,
        Node::Repeat { node, min, .. } => *min == 0 || can_match_empty(node),
        Node::Concat(nodes) => nodes.iter().all(can_match_empty),
        Node::Alt(alternatives) => alternatives.iter().any(can_match_empty),
        Node::NotAhead(_) => true,
    }
}

#[inline]
fn next_char(text: &str, pos: usize) -> Option<(char, usize)> {
    let byte = *text.as_bytes().get(pos)?;
    if byte < 0x80 {
        return Some((byte as char, 1));
    }
    let c = text[pos..].chars().next()?;
    Some((c, c.len_utf8()))
}

impl Parser<'_> {
    fn add_class(&mut self, class: Class) -> usize {
        self.classes.push(class.finish());
        self.classes.len() - 1
    }

    fn parse_alternation(&mut self) -> Option<Node> {
        let mut alternatives = vec![self.parse_concat()?];
        while self.chars.peek() == Some(&'|') {
            self.chars.next();
            alternatives.push(self.parse_concat()?);
        }
        Some(if alternatives.len() == 1 {
            alternatives.pop()?
        } else {
            Node::Alt(alternatives)
        })
    }

    fn parse_concat(&mut self) -> Option<Node> {
        let mut nodes = Vec::new();
        let mut previous_letter = None;
        while let Some(&c) = self.chars.peek() {
            if c == '|' || c == ')' {
                break;
            }
            nodes.push(self.parse_quantified()?);
            // Under `(?i:)` the engine also matches a run of letters against
            // the one character whose full case folding is that run (ß and ẞ
            // for "ss", ﬁ for "fi", ﬆ for "st", ...), which a matcher that
            // folds letter by letter cannot do: such runs are declined.
            let letter = (self.case_insensitive
                && c.is_ascii_alphabetic()
                && matches!(nodes.last(), Some(Node::Class(_))))
            .then(|| c.to_ascii_lowercase());
            if let (Some(first), Some(second)) = (previous_letter, letter) {
                if MULTI_CHARACTER_FOLDS.contains(&[first, second]) {
                    return None;
                }
            }
            previous_letter = letter;
        }
        Some(if nodes.len() == 1 {
            nodes.pop()?
        } else {
            Node::Concat(nodes)
        })
    }

    fn parse_quantified(&mut self) -> Option<Node> {
        let atom = self.parse_atom()?;
        let (min, max) = match self.chars.peek() {
            Some('?') => (0, 1),
            Some('*') => (0, u32::MAX),
            Some('+') => (1, u32::MAX),
            Some('{') => {
                self.chars.next();
                let min = self.parse_number()?;
                let max = match self.chars.next()? {
                    '}' => min,
                    ',' => {
                        if self.chars.peek() == Some(&'}') {
                            u32::MAX
                        } else {
                            self.parse_number()?
                        }
                    }
                    _ => return None,
                };
                if max != min && self.chars.next()? != '}' {
                    return None;
                }
                if min > max {
                    return None;
                }
                return self.quantified(atom, min, max);
            }
            _ => return Some(atom),
        };
        self.chars.next();
        self.quantified(atom, min, max)
    }

    fn quantified(&mut self, atom: Node, min: u32, max: u32) -> Option<Node> {
        // Lazy (`??`, `*?`) and possessive (`?+`, `*+`) forms are not supported.
        if matches!(self.chars.peek(), Some('?' | '+')) {
            return None;
        }
        Some(match atom {
            Node::Class(class) => Node::ClassRepeat { class, min, max },
            Node::NotAhead(_) => return None,
            // A group repeats by recursion, one frame per repetition: only the
            // optional form is accepted, longer runs stay with `tokenizers`.
            _ if max > 1 => return None,
            node => Node::Repeat {
                node: Box::new(node),
                min,
                max,
            },
        })
    }

    fn parse_number(&mut self) -> Option<u32> {
        let mut digits = String::new();
        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() {
                digits.push(c);
                self.chars.next();
            } else {
                break;
            }
        }
        digits.parse().ok()
    }

    fn parse_atom(&mut self) -> Option<Node> {
        let c = self.chars.next()?;
        // Under `(?i:...)` only ASCII letters and plain ASCII literals are
        // folded the way the engine folds them, letter by letter; a class, an
        // escape or a non-ASCII literal there would match differently, so it
        // is declined (so are lookaheads and the letter runs the engine folds
        // as one character; see `parse_group` and `parse_concat`).
        if self.case_insensitive && (c == '[' || c == '\\' || !c.is_ascii()) {
            return None;
        }
        match c {
            '(' => self.parse_group(),
            '[' => {
                let class = self.parse_class_body()?;
                Some(Node::Class(self.add_class(class)))
            }
            '\\' => {
                let class = self.parse_escape_as_class()?;
                Some(Node::Class(self.add_class(class)))
            }
            '.' | '^' | '$' | '*' | '+' | '?' | '{' | '}' | ')' | '|' | ']' => None,
            literal => Some(Node::Class(self.literal_class(literal))),
        }
    }

    fn parse_group(&mut self) -> Option<Node> {
        if self.chars.next()? != '?' {
            return None;
        }
        match self.chars.next()? {
            ':' => {
                let inner = self.parse_alternation()?;
                (self.chars.next()? == ')').then_some(inner)
            }
            'i' => {
                if self.chars.next()? != ':' {
                    return None;
                }
                let was = std::mem::replace(&mut self.case_insensitive, true);
                let inner = self.parse_alternation();
                self.case_insensitive = was;
                let inner = inner?;
                (self.chars.next()? == ')').then_some(inner)
            }
            '!' => {
                // The option would reach into the lookahead's class, which
                // this matcher does not fold: declined under `(?i:)`.
                if self.case_insensitive {
                    return None;
                }
                let class = match self.chars.next()? {
                    '\\' => self.parse_escape_as_class()?,
                    '[' => self.parse_class_body()?,
                    _ => return None,
                };
                if self.chars.next()? != ')' {
                    return None;
                }
                let class = self.add_class(class);
                Some(Node::NotAhead(class))
            }
            _ => None,
        }
    }

    /// A literal character as a class: itself, or under `(?i:)` every
    /// character the engine folds together with it.
    fn literal_class(&mut self, c: char) -> usize {
        let mut class = Class::default();
        if self.case_insensitive && c.is_ascii_alphabetic() {
            let lower = c.to_ascii_lowercase();
            let upper = c.to_ascii_uppercase();
            class.ranges.push((u32::from(lower), u32::from(lower)));
            class.ranges.push((u32::from(upper), u32::from(upper)));
            if let Some((_, extra)) = unicode::FOLD_EXTRA
                .iter()
                .find(|(letter, _)| *letter == lower as u8)
            {
                for &e in *extra {
                    class.ranges.push((u32::from(e), u32::from(e)));
                }
            }
            class.ranges.sort_unstable();
        } else {
            class.ranges.push((u32::from(c), u32::from(c)));
        }
        self.add_class(class)
    }

    /// An escape outside a class, as a class of its own.
    fn parse_escape_as_class(&mut self) -> Option<Class> {
        let mut class = Class::default();
        match self.chars.next()? {
            'p' => class.categories |= self.parse_property()?,
            'P' => {
                class.categories |= self.parse_property()?;
                class.negated = true;
            }
            's' => class.categories |= CAT_SPACE,
            'S' => {
                class.categories |= CAT_SPACE;
                class.negated = true;
            }
            other => {
                let c = escaped_literal(other)?;
                class.ranges.push((u32::from(c), u32::from(c)));
            }
        }
        Some(class)
    }

    /// `{L}`, `{Lu}`, ... after `\p`.
    fn parse_property(&mut self) -> Option<u16> {
        if self.chars.next()? != '{' {
            return None;
        }
        let mut name = String::new();
        loop {
            match self.chars.next()? {
                '}' => break,
                c => name.push(c),
            }
        }
        Some(match name.as_str() {
            "L" => CAT_LETTER,
            "M" => CAT_MARK,
            "N" => CAT_NUMBER,
            "P" => CAT_PUNCTUATION,
            "S" => CAT_SYMBOL,
            "Lu" => CAT_UPPERCASE,
            "Ll" => CAT_LOWERCASE,
            "Lt" => CAT_TITLECASE,
            "Lm" => CAT_MODIFIER_LETTER,
            "Lo" => CAT_OTHER_LETTER,
            _ => return None,
        })
    }

    /// The body of `[...]`, after the opening bracket.
    fn parse_class_body(&mut self) -> Option<Class> {
        let mut class = Class::default();
        if self.chars.peek() == Some(&'^') {
            self.chars.next();
            class.negated = true;
        }
        let mut first = true;
        loop {
            let c = self.chars.next()?;
            let item = match c {
                ']' if !first => break,
                '\\' => match self.chars.next()? {
                    'p' => {
                        class.categories |= self.parse_property()?;
                        first = false;
                        continue;
                    }
                    's' => {
                        class.categories |= CAT_SPACE;
                        first = false;
                        continue;
                    }
                    'P' | 'S' | 'd' | 'D' | 'w' | 'W' | 'h' | 'H' => return None,
                    other => escaped_literal(other)?,
                },
                // Nested classes, intersections and POSIX brackets are not supported.
                '[' => return None,
                '&' if self.chars.peek() == Some(&'&') => return None,
                literal => literal,
            };
            first = false;
            // A range `a-b`, unless the dash ends the class.
            if self.chars.peek() == Some(&'-') {
                self.chars.next();
                match self.chars.peek() {
                    Some(&']') => {
                        class.ranges.push((u32::from(item), u32::from(item)));
                        class.ranges.push((u32::from('-'), u32::from('-')));
                        continue;
                    }
                    Some(&'\\') => {
                        self.chars.next();
                        let end = match self.chars.next()? {
                            'p' | 's' | 'P' | 'S' => return None,
                            other => escaped_literal(other)?,
                        };
                        if end < item {
                            return None;
                        }
                        class.ranges.push((u32::from(item), u32::from(end)));
                    }
                    Some(_) => {
                        let end = self.chars.next()?;
                        if end == '[' || end < item {
                            return None;
                        }
                        class.ranges.push((u32::from(item), u32::from(end)));
                    }
                    None => return None,
                }
            } else {
                class.ranges.push((u32::from(item), u32::from(item)));
            }
        }
        class.ranges.sort_unstable();
        Some(class)
    }
}

/// The character an escape stands for, when it is a plain literal escape.
fn escaped_literal(c: char) -> Option<char> {
    Some(match c {
        'r' => '\r',
        'n' => '\n',
        't' => '\t',
        'f' => '\x0C',
        'v' => '\x0B',
        'e' => '\x1B',
        'a' => '\x07',
        // Any escaped non-alphanumeric character is itself.
        c if !c.is_alphanumeric() => c,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use tokenizers::utils::SysRegex;

    use super::*;

    const GPT2: &str =
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN35: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const DIGITS: &str = r"\p{N}{1,3}";
    const CJK: &str = "[一-龥぀-ゟ゠-ヿ]+";
    const DEEPSEEK: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
    const CASED: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    const PATTERNS: [&str; 8] = [GPT2, QWEN2, QWEN35, CL100K, DIGITS, CJK, DEEPSEEK, CASED];

    /// Characters that exercise every class the patterns use: ASCII of all
    /// kinds, spaces and line breaks of several scripts, letters with and
    /// without case, combining marks, digits beyond ASCII, CJK, kana, emoji,
    /// and the clitic letters in their folded forms.
    const POOL: &[char] = &[
        'a',
        'b',
        'e',
        'd',
        'l',
        'm',
        'r',
        's',
        't',
        'v',
        'A',
        'E',
        'L',
        'S',
        'T',
        'Z',
        'ſ',
        'ß',
        'é',
        'Ö',
        'ǅ',
        'ʰ',
        'ª',
        '0',
        '1',
        '9',
        '٣',
        '๒',
        '²',
        '½',
        ' ',
        '\t',
        '\n',
        '\r',
        '\u{0B}',
        '\u{0C}',
        '\u{85}',
        '\u{A0}',
        '\u{2003}',
        '\u{2028}',
        '\u{3000}',
        '\'',
        '"',
        ',',
        '.',
        '-',
        '/',
        '_',
        '~',
        '[',
        ']',
        '\\',
        '^',
        '`',
        '{',
        '|',
        '}',
        '@',
        '#',
        '$',
        '%',
        '&',
        '*',
        '+',
        ':',
        ';',
        '<',
        '=',
        '>',
        '?',
        '!',
        '(',
        ')',
        '€',
        '©',
        '→',
        '\u{301}',
        '\u{93E}',
        '\u{E31}',
        '\u{E48}',
        '\u{9BE}',
        '\u{C4D}',
        'ก',
        'ข',
        'ा',
        'क',
        'ب',
        'я',
        'Я',
        'α',
        'Ω',
        '中',
        '文',
        '龥',
        'あ',
        'ん',
        'ア',
        'ー',
        '한',
        '😀',
        '👍',
        '\u{1F3FD}',
        '\u{200D}',
        '\u{FE0F}',
        '\u{FFFD}',
        '\u{0}',
        '\u{7F}',
        '\u{AD}',
        '\u{1F}',
    ];

    fn oracle_matches(pattern: &str, text: &str) -> Vec<(usize, usize)> {
        let re = SysRegex::new(pattern).expect("the engine compiles the pattern");
        re.find_iter(text).collect()
    }

    fn pseudo_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_text(state: &mut u64, len: usize) -> String {
        (0..len)
            .map(|_| POOL[(pseudo_random(state) % POOL.len() as u64) as usize])
            .collect()
    }

    #[test]
    fn every_known_pattern_parses() {
        for pattern in PATTERNS {
            assert!(Pattern::parse(pattern).is_some(), "{pattern}");
        }
    }

    #[test]
    fn unsupported_syntax_is_rejected() {
        for pattern in [
            r"a.b",
            r"^a",
            r"a$",
            r"(a)",
            r"a*?",
            r"a++",
            r"[[:alpha:]]",
            r"[a&&b]",
            r"\d+",
            r"(?=a)",
            r"\p{Han}",
            r"a{3,1}",
            r"[z-a]",
            r"(?:ab)+",
            r"(?: ?\p{L})*",
            r"(?:ab){2}",
            r"(?i:[sdmt]|ll|ve|re)",
            r"(?i:\p{Ll})",
            r"(?i:é)",
            r"\p{N}*",
            r"a?",
            r"(?!\S)",
            r"\s*|\p{L}+",
            r"(?:\p{L}+)?",
            r"a|",
            r"",
            r"(?i:'s(?![a-z]))",
            r"(?i:'s(?!\S))",
            r"(?i:ss)",
            r"(?i:'st)",
            r"(?i:fi|fl|ff)",
        ] {
            assert!(Pattern::parse(pattern).is_none(), "{pattern}");
        }
    }

    #[test]
    fn matches_agree_with_the_engine_on_random_text() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for pattern in PATTERNS {
            let parsed = Pattern::parse(pattern).expect("parses");
            for round in 0..1500 {
                let text = random_text(&mut state, 1 + round % 24);
                let ours: Vec<(usize, usize)> = parsed.find_iter(&text).collect();
                assert_eq!(
                    ours,
                    oracle_matches(pattern, &text),
                    "pattern {pattern:?} text {text:?}"
                );
            }
        }
    }

    #[test]
    fn matches_agree_with_the_engine_on_clitics_and_whitespace_runs() {
        let texts = [
            "I'm sure they'RE here, it'S fine, we'll see, you'd know, I've, don't, isn'T",
            "I'ſ and WE'LL and it'ſ and 'ſ'ſ",
            "a  b   c\n\n\nd \n e\r\n\r\nf  \n  g",
            "   leading and trailing   ",
            "tabs\t\tand\u{A0}nbsp\u{3000}ideographic\u{2028}sep",
            "123456789 ١٢٣٤ ๑๒๓ 1,234.56",
            "日本語のテキストと한국어와中文混在",
            "e\u{301}cole ka\u{93E} th\u{E31}\u{E48} b\u{9BE}",
            "snake_case-kebab.dot/slash\\back `tick` ~tilde~ [br] {cu} (pa) <an>",
            "emoji 😀👍🏽 and ✨ and \u{FE0F} and \u{200D}",
            "",
            "\n",
            " ",
            "'",
            "''s",
        ];
        for pattern in PATTERNS {
            let parsed = Pattern::parse(pattern).expect("parses");
            for text in texts {
                let ours: Vec<(usize, usize)> = parsed.find_iter(text).collect();
                assert_eq!(
                    ours,
                    oracle_matches(pattern, text),
                    "pattern {pattern:?} text {text:?}"
                );
            }
        }
    }

    #[test]
    fn long_runs_do_not_recurse_per_character() {
        let parsed = Pattern::parse(QWEN35).expect("parses");
        let spaces = " ".repeat(2_000_000);
        let ours: Vec<(usize, usize)> = parsed.find_iter(&spaces).collect();
        assert_eq!(ours, vec![(0, 2_000_000)]);
        let digits = "7".repeat(300_000);
        assert_eq!(parsed.find_iter(&digits).count(), 300_000);
        let mixed = "\n".repeat(100_000) + &" ".repeat(100_000) + "x";
        let ours: Vec<(usize, usize)> = parsed.find_iter(&mixed).collect();
        assert_eq!(ours, oracle_matches(QWEN35, &mixed));
    }

    #[test]
    fn patterns_that_can_match_empty_are_declined() {
        // The engine yields empty matches and `Split` cuts the text at them:
        // `\p{N}*` over "ab1" matches at 0, at 1 and at 2..3, so the pieces
        // are "a", "b" and "1". A matcher that skipped the empty matches
        // would give "ab" and "1", and the ids could differ.
        assert_eq!(
            oracle_matches(r"\p{N}*", "ab1"),
            vec![(0, 0), (1, 1), (2, 3)]
        );
        for (pattern, text) in [
            (r"\p{N}*", "ab1"),
            (r"a?", "ab1"),
            (r"(?i:'s)?", "x's y"),
            (r"\s*|\p{L}+", "ab 1"),
            (r"\p{N}{0,3}", "ab12"),
            (r"(?!\S)", "a b"),
            (r"a*b*", "ab1"),
            (r"(?:\p{L}+)?", "ab 1"),
            (r"", "ab"),
            (r"a|", "ab"),
        ] {
            if let Some(parsed) = Pattern::parse(pattern) {
                assert_eq!(
                    parsed.find_iter(text).collect::<Vec<_>>(),
                    oracle_matches(pattern, text),
                    "{pattern:?} is accepted but matches {text:?} differently from the engine"
                );
            }
            assert!(
                Pattern::parse(pattern).is_none(),
                "{pattern:?} can match empty and must be declined"
            );
        }
    }

    #[test]
    fn case_insensitive_groups_decline_classes_and_non_ascii_literals() {
        // Under `(?i:)` the engine folds classes, properties and non-ASCII
        // literals as well; this matcher folds ASCII letters only, so a
        // group with anything else is declined rather than matched
        // differently.
        assert_eq!(
            oracle_matches(r"(?i:[a-z]+)", "ABC \u{17F} \u{212A}"),
            vec![(0, 3), (4, 6), (7, 10)]
        );
        assert_eq!(oracle_matches(r"(?i:é)", "É"), vec![(0, 2)]);
        for (pattern, text) in [
            (r"(?i:[a-z]+)", "ABC \u{17F} \u{212A}"),
            (r"(?i:[sdmt]|ll|ve|re)", "'S 'LL"),
            (r"(?i:\p{Ll})", "A"),
            (r"(?i:é)", "É"),
            (r"(?i:'[a-z])", "'S"),
            (r"(?i:ǆ)", "Ǆ ǅ"),
        ] {
            if let Some(parsed) = Pattern::parse(pattern) {
                assert_eq!(
                    parsed.find_iter(text).collect::<Vec<_>>(),
                    oracle_matches(pattern, text),
                    "{pattern:?} is accepted but matches {text:?} differently from the engine"
                );
            }
            assert!(
                Pattern::parse(pattern).is_none(),
                "{pattern:?} must be declined"
            );
        }
    }

    #[test]
    fn case_insensitive_groups_decline_lookaheads_and_multi_character_folds() {
        // The option reaches into a lookahead: `(?![a-z])` under `(?i:)`
        // also rejects A-Z, ſ and K.
        assert_eq!(
            oracle_matches(r"(?i:'s(?![a-z]))", "'sA 'sb 's"),
            vec![(8, 10)]
        );
        // A run of ASCII letters also matches the one character whose full
        // case folding is that run: ß and ẞ for "ss", ﬁ for "fi", ﬆ for "st".
        assert_eq!(oracle_matches(r"(?i:ss)", "ß"), vec![(0, 2)]);
        assert_eq!(oracle_matches(r"(?i:'st)", "'ﬆ"), vec![(0, 4)]);
        for (pattern, text) in [
            (r"(?i:'s(?![a-z]))", "'sA 'sb 's"),
            (r"(?i:'s(?!\p{Ll}))", "'sA"),
            (r"(?i:'s(?!\S))", "'sA 's"),
            (r"(?i:ss)", "ß ẞ ss"),
            (r"(?i:'st|'ss)", "'ﬆ 'ß"),
            (r"(?i:fi|fl|ff)", "ﬁ ﬂ ﬀ"),
            (r"(?i:ffi)", "ﬃ"),
            (r"(?i:'s|'t|'re|'ve|'m|'ll|'d|'st)", "'ﬅ"),
        ] {
            if let Some(parsed) = Pattern::parse(pattern) {
                assert_eq!(
                    parsed.find_iter(text).collect::<Vec<_>>(),
                    oracle_matches(pattern, text),
                    "{pattern:?} is accepted but matches {text:?} differently from the engine"
                );
            }
            assert!(
                Pattern::parse(pattern).is_none(),
                "{pattern:?} must be declined"
            );
        }
        // The clitic groups in use fold letter by letter and stay accepted.
        for pattern in [QWEN2, QWEN35, CL100K, CASED] {
            assert!(Pattern::parse(pattern).is_some(), "{pattern}");
        }
    }

    #[test]
    fn group_repeats_and_long_runs_on_a_small_stack() {
        // A group under `+`, `*` or `{n,}` would recurse once per repetition
        // and overflow the stack on a long run, which aborts the process; such
        // patterns are declined, the optional form recurses at most once and
        // class runs iterate, so long inputs match on a 256 KiB stack.
        let worker = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                for pattern in [
                    r"(?:\r?\n)+",
                    r"(?: ?\p{L})+",
                    r"(?:ab)*",
                    r"(?:a|b){2,}",
                    r"(?:'s){1,3}",
                ] {
                    assert!(Pattern::parse(pattern).is_none(), "{pattern}");
                }
                let text = "\r\n".repeat(50_000) + &" x".repeat(50_000) + &"'s".repeat(5_000);
                for pattern in [GPT2, QWEN35, CASED] {
                    let parsed = Pattern::parse(pattern).expect("parses");
                    let ours: Vec<(usize, usize)> = parsed.find_iter(&text).collect();
                    assert_eq!(ours, oracle_matches(pattern, &text), "{pattern}");
                }
            })
            .expect("spawn");
        worker
            .join()
            .expect("the matcher must not overflow a small stack");
    }
}
