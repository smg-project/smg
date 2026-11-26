"""Hold the layout of a package that keeps its unit tests under `tests/unit`.

The rule binds a package that holds a `tests/unit` folder, and no other package.
In such a package:

- No case and no test module's body stands under `src`.
- A test module is declared under `src` by a pointer: the test gate, a path
  attribute and `mod name;`, at the first level of its source file. The path
  attribute names the rule's place, which is `tests/unit/` + the path of the
  source file that holds the pointer (without `.rs`; the folder alone for a
  `mod.rs` and a `lib.rs`) + the module's name. The file is `<place>.rs`, or
  `<place>/mod.rs` where it has file children.
- No test module stands inside an inline module. The compiler reads a path
  attribute there from a folder of the inline module's name, and a module
  without file children has no such folder. An inline module that holds tests
  becomes a file of its own, and the pointer stands at that file's first level.
- A pointer may name a file of the package's `tests` folder outside
  `tests/unit` only where a test program of the package loads the same file.
- Every file under `tests/unit` is named by a pointer of a source file that the
  package compiles, or by a `mod` line of a named file. A file that nothing
  names is compiled by nobody, and its cases run nowhere.

A gate is a test gate only where it cannot hold without `test`: the plain
`cfg(test)`, or an `all` with such a member. An `any` with another member, or a
`not`, gates product code too.

A package in mid-move says so by a file `tests/unit/MOVING.md`. There the first
rule is not asked: a test module that has not moved yet may stand under `src`,
as a body, as a `mod` line with no path attribute or with one that names a file
under `src`, or inside an inline module, and a case may stand there too. Every
other rule binds: a pointer that names a file outside `src` names the rule's
place, and every file under `tests/unit` is named. The last change of the move
deletes the file.

Usage: check_unit_test_layout.py [ROOT]. ROOT is the repository's folder, by
default the one that holds this script's folder. Each fault is one line on
standard output, and the exit code is 1 where there is one.
"""

import posixpath
import re
import sys
from pathlib import Path

UNIT = "tests/unit"
# A package that holds this file under its `tests/unit` is in mid-move.
MOVING = "MOVING.md"
# Where a fault's reader finds the rule: this file, by its path in the repository.
RULE = "scripts/check_unit_test_layout.py"
# Folders the walk for packages does not enter, with every folder whose name
# begins with a dot: build output and what other tools install.
SKIPPED = {"target", "node_modules", "__pycache__"}
IDENT = r"(?:r#)?[A-Za-z_][A-Za-z0-9_]*"
MOD = re.compile(r"(?:\bpub(?:\s*\([^()]*\))?\s+)?\bmod\s+(?P<name>" + IDENT + r")\s*(?P<end>[;{])")
CASE = re.compile(r"#\[\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)*test\s*[(\]]")
MACRO = re.compile(r"\bmacro_rules\s*!\s*" + IDENT + r"\s*[({\[]")
PATH_ATTRIBUTE = re.compile(r'#\[\s*path\s*=\s*"([^"]*)"\s*\]')
MANIFEST_PATH = re.compile(r'(?m)^\s*path\s*=\s*"([^"]+\.rs)"')


LEXEME = re.compile(r'//[^\n]*(?:\n[ \t]*//[^\n]*)*|/\*|(?<![A-Za-z0-9_])[bc]?r(#*)"|"|\'')
STRING_END = re.compile(r'(?:[^"\\]|\\.)*"', re.DOTALL)
CHARACTER = re.compile(r"'(?:\\.[^'\n]*|[^\\'\n])'", re.DOTALL)
COMMENT_EDGE = re.compile(r"/\*|\*/")
NOT_LINE_FEED = re.compile(r"[^\n]")


def blanked(text):
    """`text` with each comment and the inside of each string and character
    literal turned to spaces. Line feeds stay, so what is left is code at the
    places it has in the file."""
    out = []
    size = len(text)
    done = 0

    def wipe(start, end):
        nonlocal done
        out.append(text[done:start])
        out.append(NOT_LINE_FEED.sub(" ", text[start:end]))
        done = end

    at = 0
    while True:
        found = LEXEME.search(text, at)
        if not found:
            break
        start, word = found.start(), found.group()
        if word.startswith("//"):
            wipe(start, found.end())
            at = found.end()
        elif word == "/*":
            depth, at = 1, found.end()
            while depth:
                edge = COMMENT_EDGE.search(text, at)
                if not edge:
                    at = size
                    break
                depth += 1 if edge.group() == "/*" else -1
                at = edge.end()
            wipe(start, at)
        elif word == '"':
            close = STRING_END.match(text, found.end())
            at = close.end() if close else size
            wipe(found.end(), max(found.end(), at - 1))
        elif word == "'":
            # A character literal closes with a quote. A lifetime has none.
            literal = CHARACTER.match(text, start)
            if literal:
                wipe(start + 1, literal.end() - 1)
                at = literal.end()
            else:
                at = start + 1
        else:
            # A raw string: it closes at a quote with as many `#` as opened it.
            close = '"' + found.group(1)
            stop = text.find(close, found.end())
            stop = size if stop < 0 else stop
            wipe(found.end(), stop)
            at = min(size, stop + len(close))
    out.append(text[done:])
    return "".join(out)


BRACKET = re.compile(r"[()\[\]{}]")


def closing(code, at):
    """The place behind the bracket that closes the one at `at`."""
    depth = 0
    for found in BRACKET.finditer(code, at):
        if found.group() in "([{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return found.end()
    return len(code)


def members(code):
    """The members of a list written without white space, cut at its own commas."""
    out, depth, start = [], 0, 0
    for at, char in enumerate(code):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            out.append(code[start:at])
            start = at + 1
    out.append(code[start:])
    return [member for member in out if member]


def only_in_tests(predicate):
    """Whether a cfg predicate cannot hold without `test`."""
    if predicate == "test":
        return True
    for name in ("all", "any"):
        if predicate.startswith(name + "(") and predicate.endswith(")"):
            parts = members(predicate[len(name) + 1 : -1])
            if name == "all":
                return any(only_in_tests(part) for part in parts)
            return bool(parts) and all(only_in_tests(part) for part in parts)
    return False


def gates_to_tests(attribute):
    """Whether an attribute holds its item to test builds alone."""
    code = "".join(blanked(attribute).split())
    if not (code.startswith("#[cfg(") and code.endswith(")]")):
        return False
    return only_in_tests(code[len("#[cfg(") : -len(")]")])


def raw(name):
    """A module's name without the mark of a raw identifier."""
    return name[2:] if name.startswith("r#") else name


def path_of(attributes):
    """The string of the path attribute among `attributes`, or None."""
    for attribute in attributes:
        found = PATH_ATTRIBUTE.search(attribute)
        if found:
            return found.group(1)
    return None


class Source:
    """One Rust file as the check reads it: its `mod` lines, each with the
    inline modules it stands in, and the bodies of its macro rules."""

    def __init__(self, text):
        self.text = text
        self.macros = []
        # Each `mod` line: where its head begins, its name, whether it opens a
        # body, and the inline modules it stands in, outermost first.
        self.mods = []
        self.blank = None
        if not MOD.search(text):
            # No `mod` line, in code or in a comment: the text is not taken
            # apart for one.
            return
        self.macros = [
            (found.start(), closing(self.code, found.end() - 1))
            for found in MACRO.finditer(self.code)
        ]
        lines = [found for found in MOD.finditer(self.code) if not self.in_macro(found.start())]
        bodies = [
            (found.end() - 1, closing(self.code, found.end() - 1), found.group("name"))
            for found in lines
            if found.group("end") == "{"
        ]
        for found in lines:
            at = found.start()
            chain = [name for start, end, name in bodies if start < at < end]
            name = found.group("name")
            self.mods.append(
                (at, raw(name), found.group("end") == "{", [raw(inner) for inner in chain])
            )

    @property
    def code(self):
        """The file's text as code: see [`blanked`]."""
        if self.blank is None:
            self.blank = blanked(self.text)
        return self.blank

    def cases(self):
        """Where each case of the file begins: an attribute whose path ends in
        `test`, outside the rules of a macro."""
        if not CASE.search(self.text):
            return []
        macros = [
            (found.start(), closing(self.code, found.end() - 1))
            for found in MACRO.finditer(self.code)
        ]
        return [
            found.start()
            for found in CASE.finditer(self.code)
            if not any(start <= found.start() < end for start, end in macros)
        ]

    def in_macro(self, at):
        return any(start <= at < end for start, end in self.macros)

    def line(self, at):
        return self.text.count("\n", 0, at) + 1

    def attributes(self, at):
        """The attributes of the item whose head begins at `at`, in their
        order, and the place where the first begins (`at` where there is none).
        Each is read from its closing bracket back to the bracket that opens
        it, so one that the format step wraps over lines is read whole."""
        out = []
        while True:
            end = at
            while end > 0 and self.code[end - 1].isspace():
                end -= 1
            if end == 0 or self.code[end - 1] != "]":
                break
            depth, start = 0, end - 1
            while start >= 0:
                if self.code[start] == "]":
                    depth += 1
                elif self.code[start] == "[":
                    depth -= 1
                    if depth == 0:
                        break
                start -= 1
            if start < 1 or self.code[start - 1] != "#":
                break
            out.append(self.text[start - 1 : end])
            at = start - 1
        out.reverse()
        return out, at


def loaded(package, roots, sources):
    """The files a set of roots loads, each by its path from the package's
    folder: the roots, and each file a `mod name;` line of a loaded file names.

    The compiler looks for such a child in the folder of the file that holds
    the line where that file owns its folder (a root, a `mod.rs`, a file that a
    path attribute loads), and in the folder of the file's own name for any
    other file. Each inline module the line stands in adds its name. A path
    attribute on the line names the file itself: at a file's first level from
    the file's own folder, inside inline modules from the folder where the
    innermost one's children stand."""
    named = set()
    queue = [(root, True) for root in roots]
    while queue:
        file, owns = queue.pop()
        if file in named or not file.endswith(".rs") or not (package / file).is_file():
            continue
        named.add(file)
        folder, base = posixpath.split(file)
        children = folder if owns or base == "mod.rs" else posixpath.join(folder, base[:-3])
        if file not in sources:
            sources[file] = Source((package / file).read_text(encoding="utf-8"))
        source = sources[file]
        for at, name, body, chain in source.mods:
            if body:
                continue
            place = posixpath.join(children, *chain)
            path = path_of(source.attributes(at)[0])
            if path is not None:
                start = place if chain else folder
                queue.append((posixpath.normpath(posixpath.join(start, path)), True))
                continue
            queue.append((posixpath.join(place, name + ".rs"), False))
            queue.append((posixpath.join(place, name, "mod.rs"), False))
    return named


def roots_of(package):
    """The package's roots as cargo finds them by itself, with each file its
    manifest names by a path: (those of product code, those of test programs)."""
    manifest = (package / "Cargo.toml").read_text(encoding="utf-8")
    named = [path for path in MANIFEST_PATH.findall(manifest) if not path.startswith("..")]
    product = ["src/lib.rs", "src/main.rs"]
    product += sorted(path.relative_to(package).as_posix() for path in package.glob("src/bin/*.rs"))
    product += sorted(
        path.relative_to(package).as_posix() for path in package.glob("src/bin/*/main.rs")
    )
    product += [path for path in named if path.startswith("src/")]
    programs = sorted(path.relative_to(package).as_posix() for path in package.glob("tests/*.rs"))
    programs += sorted(
        path.relative_to(package).as_posix() for path in package.glob("tests/*/main.rs")
    )
    programs += [path for path in named if path.startswith("tests/")]
    return product, programs


def check_package(package, label, moving=False):
    """The faults of one bound package, each a line that begins with its place.

    `moving` says the package is in mid-move: what has not moved yet is no fault.
    """
    faults = []
    sources = {}
    product, programs = roots_of(package)
    compiled = loaded(package, product, sources)
    shared = None
    pointers = []
    for path in sorted((package / "src").rglob("*.rs")):
        if not path.is_file():
            continue
        rel = path.relative_to(package).as_posix()
        if rel not in sources:
            sources[rel] = Source(path.read_text(encoding="utf-8"))
        source = sources[rel]
        folder, base = posixpath.split(rel)
        stem = base[:-3]
        own = [UNIT] + [part for part in folder.split("/")[1:] if part]
        if base not in ("mod.rs", "lib.rs"):
            own.append(stem)
        bodies = []
        for at, name, body, chain in source.mods:
            if any(start <= at < end for start, end in bodies):
                # Inside a test module's body, which is one fault already.
                continue
            attributes, first = source.attributes(at)
            if not any(gates_to_tests(attribute) for attribute in attributes):
                continue
            place = f"{label}/{rel}:{source.line(first)}"
            if body:
                bodies.append((at, closing(source.code, source.code.index("{", at))))
            if chain:
                if not moving:
                    faults.append(
                        f"{place}: a test module inside the inline module `{chain[-1]}`: that "
                        "module becomes a file of its own, and the pointer stands at that file's "
                        "first level"
                    )
                continue
            if body:
                if not moving:
                    faults.append(
                        f"{place}: a test module's body under src: it stands in a file under {UNIT}"
                    )
                continue
            rule = "/".join(own + [name])
            string = path_of(attributes)
            asked = posixpath.relpath(rule + ".rs", folder)
            if string is None:
                if not moving:
                    faults.append(
                        f"{place}: `mod {name}` holds no path attribute: "
                        f'the rule\'s is #[path = "{asked}"]'
                    )
                continue
            # At a file's first level the compiler reads the string from the
            # file's own folder, however the file itself was loaded.
            target = posixpath.normpath(posixpath.join(folder, string))
            if target in (rule + ".rs", rule + "/mod.rs"):
                if not (package / target).is_file():
                    faults.append(f"{place}: `mod {name}` names {target}, which is no file")
                elif rel not in compiled:
                    faults.append(
                        f"{place}: no `mod` line of the package names this source file, "
                        f"so its pointer compiles nothing: the cases of {target} run nowhere"
                    )
                else:
                    pointers.append(target)
                continue
            if not target.startswith("tests/") or target.startswith(UNIT + "/"):
                if not (moving and target.startswith("src/")):
                    faults.append(
                        f'{place}: `mod {name}` names "{string}": the rule\'s path is "{asked}"'
                    )
                continue
            if shared is None:
                shared = loaded(package, programs, sources)
            if target not in shared:
                faults.append(
                    f"{place}: `mod {name}` names {target}: a pointer outside {UNIT} stands only "
                    "where a test program of the package loads the same file"
                )
        cases = [at for at in source.cases() if not any(start <= at < end for start, end in bodies)]
        if cases and not moving:
            count = "a case" if len(cases) == 1 else f"{len(cases)} cases, the first here,"
            faults.append(
                f"{label}/{rel}:{source.line(cases[0])}: {count} under src: "
                f"a case stands in its module's file under {UNIT}"
            )
    named = loaded(package, pointers, sources)
    for path in sorted((package / UNIT).rglob("*.rs")):
        if not path.is_file():
            continue
        rel = path.relative_to(package).as_posix()
        if rel == f"{UNIT}/main.rs":
            faults.append(f"{label}/{rel}: cargo reads this file as a test program of its own")
        elif rel not in named:
            faults.append(
                f"{label}/{rel}: no pointer under src and no module of a named file names it: "
                "its cases run nowhere"
            )
    return faults


def packages(root):
    """Each folder under `root` that holds a manifest, in the order of the paths."""
    found = []
    pending = [root]
    while pending:
        folder = pending.pop()
        if (folder / "Cargo.toml").is_file():
            found.append(folder)
        for child in folder.iterdir():
            if child.name in SKIPPED or child.name.startswith("."):
                continue
            if child.is_dir() and not child.is_symlink():
                pending.append(child)
    return sorted(found)


def check(root):
    """The faults of every package under `root` that the rule binds."""
    faults = []
    for package in packages(root):
        if (package / UNIT).is_dir() and (package / "src").is_dir():
            moving = (package / UNIT / MOVING).is_file()
            faults += check_package(package, package.relative_to(root).as_posix(), moving)
    return faults


def main(arguments):
    root = Path(arguments[0]) if arguments else Path(__file__).resolve().parents[1]
    faults = check(root.resolve())
    for fault in faults:
        print(fault)
    if faults:
        count = "1 fault" if len(faults) == 1 else f"{len(faults)} faults"
        print(f"{count} of the unit-test layout: the rule stands at the head of {RULE}")
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
