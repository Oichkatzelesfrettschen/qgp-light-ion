#!/bin/bash
# pdflatex-wrapper.sh
#
# Wrapper for pdflatex that distinguishes between fatal errors and
# recoverable errors like "Infinite glue shrinkage" which LaTeX 2024+
# explicitly marks as "ignored" in the log.
#
# Background:
# - LaTeX 2024-06-01 changed how infinite shrink errors are handled
# - These errors now appear as "Infinite shrink error above ignored !"
# - pdflatex still returns exit code 1, but the PDF is generated correctly
# - This wrapper checks if the PDF was created and if errors are recoverable
#
# Usage: Called by latexmk via $pdflatex configuration

# Run pdflatex with all provided arguments
pdflatex "$@"
PDFLATEX_EXIT=$?

# If pdflatex succeeded, we're done
if [ $PDFLATEX_EXIT -eq 0 ]; then
    exit 0
fi

# pdflatex failed - check if it's a recoverable error

# Extract the output directory and base name from arguments
OUTDIR=""
TEXFILE=""
for arg in "$@"; do
    case "$arg" in
        -output-directory=*)
            OUTDIR="${arg#-output-directory=}"
            ;;
        *.tex)
            TEXFILE="$arg"
            ;;
    esac
done

# Determine the PDF path
if [ -n "$TEXFILE" ]; then
    BASENAME=$(basename "$TEXFILE" .tex)
    if [ -n "$OUTDIR" ]; then
        PDFFILE="$OUTDIR/$BASENAME.pdf"
        LOGFILE="$OUTDIR/$BASENAME.log"
    else
        PDFFILE="$BASENAME.pdf"
        LOGFILE="$BASENAME.log"
    fi
else
    # Fallback: can't determine output files
    exit $PDFLATEX_EXIT
fi

# Check if PDF was created (primary success indicator)
if [ ! -f "$PDFFILE" ]; then
    echo "pdflatex-wrapper: PDF not created, actual error occurred"
    exit $PDFLATEX_EXIT
fi

# Check if the log contains only recoverable errors
# Recoverable errors have "above ignored" or similar markers
if [ -f "$LOGFILE" ]; then
    # Count actual fatal errors (lines starting with "! " that aren't followed by "ignored")
    FATAL_ERRORS=$(grep -c "^! " "$LOGFILE" 2>/dev/null | tr -d '[:space:]' || echo "0")
    IGNORED_ERRORS=$(grep -c "error above ignored" "$LOGFILE" 2>/dev/null | tr -d '[:space:]' || echo "0")

    # Ensure we have valid integers
    FATAL_ERRORS=${FATAL_ERRORS:-0}
    IGNORED_ERRORS=${IGNORED_ERRORS:-0}

    # If all errors are recoverable (ignored), return success
    if [ "$FATAL_ERRORS" -eq "$IGNORED_ERRORS" ] 2>/dev/null || [ "$FATAL_ERRORS" -eq 0 ] 2>/dev/null; then
        # PDF exists and all errors were recoverable
        exit 0
    fi
fi

# There were actual fatal errors
exit $PDFLATEX_EXIT
