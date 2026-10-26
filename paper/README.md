# Paper draft

The draft uses the official IEEE journal layout (`IEEEtran`, `compsoc` mode).
This is the safest source format for an IEEE submission. The Typst
`charged-ieee` package is useful for quick drafts, but it is a community
template and IEEE's author workflow officially supports Word and LaTeX.

Generate the figures from the committed notebook outputs and reported values:

```bash
python paper/make_figures.py
```

Then compile in a TeX environment that provides `IEEEtran`:

```bash
cd paper
latexmk -pdf main.tex
```

The current machine does not have `latexmk`, `pdflatex`, or `typst` installed,
so the source has been checked structurally but not compiled locally.

Before submission:

1. Replace the author and affiliation placeholders.
2. Complete the larger-model and power experiment in `experiment_plan.md`.
3. Add a field test or own controlled dataset. This is especially important
   for the two highest-impact agricultural engineering targets.
4. Select a venue and adapt the class options/page limit to its author kit.
5. Re-run all reported tables from saved JSON rather than copying values by
   hand.

