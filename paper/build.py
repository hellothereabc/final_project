"""Build the whole paper, or one module of it as a standalone PDF.

The paper is split into section files under `sections/`. A module groups the
protocol and the results of one experiment so that it can be read, reviewed, or
shown on its own:

    python paper/build.py              # list the modules and their state
    python paper/build.py paper        # the whole paper -> paper/main.pdf
    python paper/build.py lighting     # one module    -> paper/build/lighting.pdf

A module build reuses the full paper's labels, so a cross-reference to a
section that is not in the module still prints the number it has in the paper
rather than `??`. That requires the full paper to have been compiled at least
once; `build.py <module>` does it automatically when `main.aux` is missing.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> (title shown on the module's own cover, method files, result files)
MODULES = {
    "framing":     ("Framing: abstract, problem, contributions",
                    [], [], ["frontmatter", "introduction"]),
    "background":  ("Background and related work",
                    [], [], ["background"]),
    "protocol":    ("Shared protocol: data, metrics, statistics",
                    ["method_protocol", "method_statistics"], [], []),
    "annotation":  ("Experiment 1: annotation convention and cross-dataset transfer",
                    ["method_protocol"], ["result_annotation"], []),
    "occlusion":   ("Experiment 2: occlusion",
                    ["method_occlusion"], ["result_occlusion"], []),
    "calibration": ("Experiment 3: confidence calibration",
                    [], ["result_calibration"], []),
    "ladder":      ("Experiment 4: model size and on-device deployment",
                    ["method_ladder", "method_deployment"], ["result_ladder"], []),
    "lighting":    ("Experiment 5: illumination and colour augmentation",
                    ["method_lighting", "method_statistics"], ["result_lighting"], []),
    "artefact":    ("Artefact and reproducibility",
                    [], [], ["artefact"]),
    "synthesis":   ("Discussion, limitations, conclusion",
                    [], [], ["discussion", "limitations", "conclusion"]),
}


PREAMBLE = r"""\documentclass[journal,compsoc]{IEEEtran}
\input{preamble-body}

\title{@TITLE@}
\author{Module of: Tomato Ripeness Detection and Counting on an Edge Device%
\thanks{Extract built by \texttt{python paper/build.py @NAME@}. Section, table
and figure numbers are those of the full paper.}}

% The full paper's labels, so a reference to a section outside this module
% prints the number it has in the paper. \newlabel is preamble-only.
\makeatletter\InputIfFileExists{build/xrefs.tex}{}{}\makeatother

\begin{document}
\maketitle

"""


def compile_tex(name: str) -> None:
    subprocess.run(["tectonic", "-X", "compile", name, "--keep-logs", "--keep-intermediates"],
                   cwd=HERE, check=True, stdout=subprocess.DEVNULL)


def labels_in(files: list[str]) -> set[str]:
    own = set()
    for stem in files:
        text = (HERE / "sections" / f"{stem}.tex").read_text()
        own.update(re.findall(r"\\label\{([^}]*)\}", text))
    return own


def xrefs(exclude: set[str]) -> str:
    """The full paper's labels, so a module's \\ref prints the paper's number."""
    aux = HERE / "main.aux"
    if not aux.exists():
        compile_tex("main.tex")
    kept = [line for line in aux.read_text().splitlines()
            if line.startswith(r"\newlabel{")
            and re.match(r"\\newlabel\{([^}]*)\}", line).group(1) not in exclude]
    return "\n".join(kept) + "\n"


def build_module(name: str) -> Path:
    title, methods, results, extras = MODULES[name]
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    (out / "xrefs.tex").write_text(xrefs(labels_in(methods + results + extras)))

    body = []
    if methods:
        body.append(r"\section{Protocol}")
        body += [rf"\input{{sections/{stem}}}" for stem in methods]
    if results:
        body.append(r"\section{Results}")
        body += [rf"\input{{sections/{stem}}}" for stem in results]
    body += [rf"\input{{sections/{stem}}}" for stem in extras]

    cites = any("\\cite" in (HERE / "sections" / f"{stem}.tex").read_text()
                for stem in methods + results + extras)
    if cites:  # an empty thebibliography is a LaTeX error, so only emit it when used
        body.append("\\bibliographystyle{IEEEtran}")
        body.append("\\bibliography{references}")

    template = PREAMBLE + "\n".join(body) + "\n\\end{document}\n"
    driver = HERE / f"_module_{name}.tex"
    driver.write_text(template.replace("@TITLE@", title).replace("@NAME@", name))

    compile_tex(driver.name)
    for suffix in (".pdf", ".log"):
        produced = HERE / f"_module_{name}{suffix}"
        if produced.exists():
            produced.replace(out / f"{name}{suffix}")
    for leftover in HERE.glob(f"_module_{name}.*"):
        leftover.unlink()
    return out / f"{name}.pdf"


def pages(pdf: Path) -> str:
    log = pdf.with_suffix(".log")
    if log.exists():
        found = re.search(r"Output written on \S+ \((\d+) pages", log.read_text())
        if found:
            return found.group(1)
    return "?"


def main() -> None:
    if len(sys.argv) < 2:
        print("modules:")
        for name, (title, methods, results, extras) in MODULES.items():
            files = methods + results + extras
            words = sum(len((HERE / "sections" / f"{f}.tex").read_text().split()) for f in files)
            pending = sum((HERE / "sections" / f"{f}.tex").read_text().count(r"\pending")
                          for f in files)
            flag = f"  {pending} pending" if pending else ""
            print(f"  {name:<12} {words:>5} words  {title}{flag}")
        print("\nusage: python paper/build.py <module|paper>")
        return

    target = sys.argv[1]
    if target == "paper":
        compile_tex("main.tex")
        print(f"paper/main.pdf, {pages(HERE / 'main.pdf')} pages")
        return
    if target not in MODULES:
        raise SystemExit(f"unknown module {target!r}; run without arguments to list them")
    pdf = build_module(target)
    print(f"{pdf.relative_to(HERE.parent)}, {pages(pdf)} pages")


if __name__ == "__main__":
    main()
