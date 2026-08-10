#!/usr/bin/env bash
# Assemble a flat, self-contained arXiv submission from paper-equity/.
#
# arXiv unpacks the tarball into one directory and runs its own TeX Live. Two things
# therefore have to change relative to the working copy: figures are referenced through
# ../reports/figures, which does not exist inside the tarball, and the bibliography must be
# shipped pre-compiled because arXiv's bibtex pass cannot be relied on for a custom style.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
out="$here/arxiv"

rm -rf "$out"
mkdir -p "$out"

cp "$here/paper.tex" "$here/neurips_2023.sty" "$here/references.bib" "$out/"
mkdir -p "$out/tables"
cp "$here"/tables/*.tex "$out/tables/" 2>/dev/null || true

# Figures, flattened. Only the PDFs the manuscript actually includes.
grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$here/paper.tex" \
  | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u > "$out/.figlist"
while read -r fig; do
  [ -z "$fig" ] && continue
  src="$root/reports/figures/$fig"
  [ -f "$src" ] || src="$root/reports/figures/${fig%.pdf}.pdf"
  cp "$src" "$out/$(basename "$fig")"
done < "$out/.figlist"
rm -f "$out/.figlist"

# The tarball is flat, so the graphics path must be too.
perl -0pi -e 's{\\graphicspath\{\{\.\./reports/figures/\}\}}{% arXiv submission: figures ship beside the source.\n\\graphicspath\{\{./\}\}}' "$out/paper.tex"

# Build once here to produce paper.bbl, which arXiv needs shipped rather than regenerated.
( cd "$out" && tectonic -X compile paper.tex --keep-intermediates --keep-logs >/dev/null 2>&1 )
[ -f "$out/paper.bbl" ] || { echo "FAIL: paper.bbl was not produced" >&2; exit 1; }

# arXiv wants source, not build products.
rm -f "$out"/paper.{aux,log,out,blg,xdv,pdf} "$out"/*.synctex.gz

( cd "$here" && tar czf arxiv-submission.tar.gz -C arxiv . )
echo "wrote $here/arxiv-submission.tar.gz"
tar tzf "$here/arxiv-submission.tar.gz" | sort
