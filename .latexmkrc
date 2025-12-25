# .latexmkrc - Latexmk configuration for QGP Light-Ion Project
#
# This file configures latexmk behavior for consistent builds.
# Reference: https://mgeier.github.io/latexmk.html

# Use pdflatex to generate PDF directly
$pdf_mode = 1;

# Default output directory (can be overridden with -outdir)
$out_dir = 'build';

# Use bibtex for bibliography
$bibtex_use = 2;

# Clean up additional auxiliary files
$clean_ext = "bbl nav out snm synctex.gz fdb_latexmk fls";

# Recorder mode for better dependency tracking
$recorder = 1;

# Don't prompt for errors
# Use wrapper script that distinguishes fatal errors from recoverable ones
# (e.g., "Infinite glue shrinkage" which LaTeX 2024+ marks as "ignored")
$pdflatex = './scripts/pdflatex-wrapper.sh -interaction=nonstopmode -file-line-error %O %S';

# Force mode: continue even if pdflatex returns non-zero
# Required because latexmk checks log files independently of exit code
$force_mode = 1;

# Print informative but not verbose output
$silent = 0;

# Maximum number of passes before declaring failure
$max_repeat = 5;
