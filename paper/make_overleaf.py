"""Assemble an Overleaf-ready project from the sources in this directory.

Produces paper/overleaf/ and paper/overleaf.zip: upload the zip to Overleaf
(New Project -> Upload Project) and it compiles there with no further edits.

Two shapes:

    python paper/make_overleaf.py                    the whole paper
    python paper/make_overleaf.py --only lighting    the whole skeleton, with
                                                     only that module's text

The second shape keeps every section of the paper in place - headings, their
labels, and the reading order - and replaces the text of the sections that are
not ready with a visible placeholder. The structure is therefore honest about
what exists: a reader sees the shape of the paper and which parts are written.
Because the stubs carry the real labels, cross-references resolve to the
numbers the sections will have in the finished paper.

Filling a stub later is a copy from `paper/sections/`, or a rebuild without the
flag. The section files in the package are the same files as in the repository,
so there is never a merge to do.

What goes in, and why:

  IEEEtran.cls, IEEEtran.bst   the official IEEE class and bibliography style,
                               v1.8b from CTAN, vendored so the project does not
                               depend on what a given TeX installation carries.
                               The course requires the IEEE LaTeX template for a
                               journal or conference paper.
  main.tex                     the driver: class options, then one \\input per
                               section, in reading order.
  preamble.tex                 packages, macros, title, author block.
  sections/*.tex               the text, one file per section, including the
                               appendix carrying the repository link that the
                               submission rules require.
  figures/                     only the figures actually referenced.
  references.bib               the bibliography.
  README.md                    how to open it in Overleaf and what is left to do.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

import build  # module registry: which sections belong to which experiment

HERE = Path(__file__).resolve().parent
OUT = HERE / "overleaf"

# Reading order of the paper. Keep in step with main.tex.
FRONT = ["frontmatter", "introduction", "background"]
METHODS = ["method_protocol", "method_occlusion", "method_ladder",
           "method_lighting", "method_statistics", "method_deployment",
           "method_implementation"]
RESULTS = ["result_annotation", "result_occlusion", "result_calibration",
           "result_ladder", "result_lighting"]
BACK = ["discussion", "artefact", "limitations", "conclusion"]
APPENDIX = ["appendix_repository"]
ALL = FRONT + METHODS + RESULTS + BACK + APPENDIX

MAIN = r"""%% ---------------------------------------------------------------------------
%% Tomato Ripeness Detection and Counting on an Edge Device
%% MSc Computing, ATU - journal paper format (IEEEtran)
%%
%% Compile with pdfLaTeX. In Overleaf: Menu -> Compiler -> pdfLaTeX,
%% Main document -> main.tex.
%%
%% The class file IEEEtran.cls in this project is the official IEEE one
%% (v1.8b). [journal] is the IEEE journal paper layout; swap in [conference]
%% for the conference format, or add ,compsoc for the Computer Society look.
%% ---------------------------------------------------------------------------
\documentclass[journal]{IEEEtran}

\input{preamble}

\begin{document}

\maketitle

%% --- Front matter ----------------------------------------------------------
%s

%% --- Methods ---------------------------------------------------------------
\section{Materials and Methods}
\label{sec:methods}
%s

%% --- Results ---------------------------------------------------------------
\section{Results}
\label{sec:results}
%s

%% --- Discussion and close --------------------------------------------------
%s

%% --- Bibliography ----------------------------------------------------------
%s\bibliographystyle{IEEEtran}
\bibliography{references}

%% --- Appendix --------------------------------------------------------------
%% Required: the submission rules penalise a missing repository link as a
%% failure to submit the project. Replace the placeholder URL before handing in.
%s

\end{document}
"""

PREAMBLE = r"""%% Packages ------------------------------------------------------------------
\usepackage[utf8]{inputenc}   % pdfLaTeX reads the sources as UTF-8
\usepackage[T1]{fontenc}      % proper hyphenation and copy-pasteable output
\usepackage{booktabs}         % \toprule, \midrule, \bottomrule
\usepackage{graphicx}
\usepackage{url}
\usepackage{amssymb}           % \checkmark

\graphicspath{{figures/}}

%% A number that is still being measured, or a section that is not written yet,
%% prints as a bold marker so that it cannot reach a submitted version
%% unnoticed. Before submitting, check:
%%     grep -rn "pending" sections/
\newcommand{\pending}[1]{\textbf{[pending: #1]}}

%% Title and author ----------------------------------------------------------
\title{Tomato Ripeness Detection and Counting on an Edge Device: How
Annotation, Occlusion, Confidence Threshold and Hardware Affect the
Reported Error}

\author{%
  David~Kalandadze\\
  {\normalsize Supervisor: Brian~McGinley}%
  \thanks{The author is with the Department of Computer Science, Atlantic
  Technological University, Galway, Ireland (e-mail: G00486658@atu.ie).}%
  \thanks{MSc in Computing dissertation, submitted October 2026.
  Supervisor: Brian McGinley.}%
}
"""

STUB_NOTE = ("%% Placeholder. The written version of this section lives in the\n"
             "%% repository at paper/sections/%s.tex; copy it in when it is ready,\n"
             "%% or rebuild the package with: python paper/make_overleaf.py\n")

# Front matter has no \section of its own; a stub still has to say something.
FRONTMATTER_STUB = r"""\begin{abstract}
\pending{abstract}
\end{abstract}

\begin{IEEEkeywords}
agricultural computer vision, object detection, tomato ripeness, occlusion,
domain shift, calibration, data augmentation, illumination robustness, model
scaling, Core ML, edge inference
\end{IEEEkeywords}
"""


def headings(text: str) -> list[str]:
    """Section and subsection headings, each with the label that follows it."""
    out, lines = [], text.splitlines()
    for i, line in enumerate(lines):
        if not re.match(r"\\(sub)?section\{", line) and not line.startswith(r"\appendices"):
            continue
        out.append(line)
        following = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if following.startswith(r"\label{"):
            out.append(following)
    return out


def stub(stem: str) -> str:
    """The shape of a section without its text: headings, labels, a marker."""
    source = (HERE / "sections" / f"{stem}.tex").read_text()
    note = STUB_NOTE % stem
    if stem == "frontmatter":
        return note + "\n" + FRONTMATTER_STUB
    blocks = []
    for line in headings(source):
        blocks.append(line)
        if line.startswith(r"\label{") or line.startswith(r"\appendices"):
            continue
        title = re.sub(r"\\(sub)?section\{(.*)\}$", r"\2", line)
        blocks.append("")
        blocks.append(rf"\pending{{{title.lower()} - not written yet}}")
        blocks.append("")
    return note + "\n" + "\n".join(blocks).strip() + "\n"


def used_figures(stems: list[str]) -> set[str]:
    names = set()
    for stem in stems:
        names.update(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}",
                                (HERE / "sections" / f"{stem}.tex").read_text()))
    return names


def readme(filled: list[str], modules: list[str]) -> str:
    if modules:
        written = "\n".join(f"- `sections/{stem}.tex`" for stem in filled)
        names = ", ".join(f"`{m}`" for m in modules)
        flags = " ".join(modules)
        state = f"""This project holds the **whole skeleton** of the dissertation: every section
is in place, in reading order, with its heading and its label. The sections
that are written carry their text; the rest print a bold `[pending: ...]`
marker instead, so the shape of the paper is visible and nothing unfinished can
be mistaken for finished.

Written so far ({names}):

{written}

Everything else is a placeholder."""
        grow = f"""## Filling in the rest

The section files here are the same files as in the repository, so there is no
merge to do. Two ways to fill a stub:

- paste the text into the file in Overleaf, or
- rebuild the package and upload it over this project:

```bash
python paper/make_overleaf.py                    # every section, written
python paper/make_overleaf.py --only {flags}  # this shape again
```
"""
    else:
        state = """This project holds the full paper: every section with its text."""
        grow = """## Rebuilding

```bash
python paper/make_overleaf.py
```

The section files here are the same files as in the repository, so Overleaf and
the repository never diverge by more than a rebuild."""

    return f"""# Overleaf project

Upload `overleaf.zip` to Overleaf: **New Project -> Upload Project**, pick the
zip. Everything needed to compile is inside; nothing else has to be installed.

After it opens, check two settings in **Menu**:

- **Compiler**: pdfLaTeX
- **Main document**: `main.tex`

Then press Recompile. The first run takes a few seconds because it also runs
BibTeX for the bibliography.

## State

{state}

## What is where

| Path | What it is |
|---|---|
| `main.tex` | The driver. Class options and the order the sections are read in. |
| `preamble.tex` | Packages, the `\\pending` macro, title and author block. |
| `sections/` | The text, one file per section. This is where you write. |
| `figures/` | The figures referenced by those sections. |
| `references.bib` | The bibliography. |
| `IEEEtran.cls`, `IEEEtran.bst` | The official IEEE class and bibliography style, v1.8b, vendored so the project does not depend on Overleaf's TeX Live version. |

The section files are grouped so that each experiment is one unit: its protocol
file under Methods and its results file under Results. `method_lighting.tex`
goes with `result_lighting.tex`, `method_ladder.tex` with `result_ladder.tex`,
and so on. Writing or reviewing one experiment means touching two files.

{grow}

## Before submitting

1. **Author block** in `preamble.tex`: name, student number, department,
   e-mail, supervisor. It currently says `Author~Name`.
2. **Repository link** in `sections/appendix_repository.tex`. The submission
   rules treat a missing GitHub link as a failure to submit the project, so
   this one is not cosmetic.
3. **Unfinished sections and numbers**: `grep -rn "pending" sections/` must come
   back empty. Anything still marked prints as a bold `[pending: ...]`.
4. **Citations**: `references.bib` is still short for a journal-length paper.
5. **Figure and table references**: Overleaf will list any undefined ones in
   the log after a recompile.

## Switching format

`main.tex` starts with `\\documentclass[journal]{{IEEEtran}}`. For the conference
format use `[conference]`; for the IEEE Computer Society look add `,compsoc`.
The text does not need to change either way.
"""


def main() -> None:
    modules = []
    if "--only" in sys.argv:
        modules = [a for a in sys.argv[sys.argv.index("--only") + 1:] if not a.startswith("--")]
        unknown = [m for m in modules if m not in build.MODULES]
        if not modules or unknown:
            raise SystemExit(f"--only takes module names, one of: {', '.join(build.MODULES)}")

    written = {stem for m in modules for group in build.MODULES[m][1:] for stem in group}
    filled = [s for s in ALL if not modules or s in written or s in APPENDIX]

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "sections").mkdir(parents=True)
    (OUT / "figures").mkdir()

    inputs = lambda stems: "\n".join(rf"\input{{sections/{stem}}}" for stem in stems)
    # A bibliography with no \cite in the text is a LaTeX error. While the
    # citing sections are still placeholders, list every entry instead, so the
    # bibliography is part of the visible skeleton.
    cites = any(r"\cite" in (HERE / "sections" / f"{stem}.tex").read_text() for stem in filled)
    nocite = ("" if cites else
              "%% Skeleton only: shows every entry of references.bib while the citing\n"
              "%% sections are placeholders. Delete once the text cites its sources.\n"
              "\\nocite{*}\n")
    (OUT / "main.tex").write_text(MAIN % (inputs(FRONT), inputs(METHODS), inputs(RESULTS),
                                          inputs(BACK), nocite, inputs(APPENDIX)))
    (OUT / "preamble.tex").write_text(PREAMBLE)
    (OUT / "README.md").write_text(readme(filled, modules))

    for stem in ALL:
        target = OUT / "sections" / f"{stem}.tex"
        if stem in filled:
            shutil.copy2(HERE / "sections" / f"{stem}.tex", target)
        else:
            target.write_text(stub(stem))
    for name in sorted(used_figures(filled)):
        shutil.copy2(HERE / "figures" / name, OUT / "figures" / name)
    shutil.copy2(HERE / "references.bib", OUT)
    for name in ("IEEEtran.cls", "IEEEtran.bst"):
        shutil.copy2(HERE / "vendor" / "IEEEtran" / name, OUT / name)

    archive = HERE / "overleaf.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))

    shape = (f"{', '.join(modules)} written, {len(ALL) - len(filled)} sections stubbed"
             if modules else "full paper")
    print(f"{OUT.relative_to(HERE.parent)}: {shape}, "
          f"{len(list((OUT / 'figures').iterdir()))} figures")
    print(f"{archive.relative_to(HERE.parent)}: {archive.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
