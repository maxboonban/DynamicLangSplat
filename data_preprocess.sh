Here’s the revised shell script—now it removes the entire “left” suffix including every preceding underscore, so foo_left.png and foo__left.png both become foo.png.

#!/usr/bin/env bash
# prep_rgb.sh ─ copy data/<dataset>/ → temp/<dataset>/,
# clean temp/<dataset>/rgb/<RES>x as requested
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <dataset_name> <resolution 1‑4>" >&2
  exit 1
fi
mkdir -p input_data/
DATASET="$1"            # e.g. plate
RES="$2"                # 1–4  → rgb/<RES>x
[[ "$RES" =~ ^[1-4]$ ]] || { echo "Resolution must be 1‑4." >&2; exit 1; }

SRC="data/${DATASET}"
DST="input_data/${DATASET}"
RGB_DIR="${DST}/rgb/${RES}x"

echo "==  Copying ${SRC} → ${DST}"
rm -rf "$DST"
rsync -a "$SRC/" "$DST/"

[[ -d "$RGB_DIR" ]] || { echo "Expected ${RGB_DIR} not found." >&2; exit 1; }

echo "--  Deleting right‑eye images"
find "$RGB_DIR" -type f \( -name '*_right.png' -o -name '*__right.png' \) -delete

echo "--  Renaming left‑eye images"
find "$RGB_DIR" -type f \( -name '*_left.png' -o -name '*__left.png' \) -print0 |
while IFS= read -r -d '' f; do
  dir=$(dirname "$f")
  base=$(basename "$f")
  clean=$(sed -E 's/_+left\.png$/.png/' <<< "$base")
  mv "$f" "$dir/$clean"
done

echo "==  Done.  Clean images live in ${RGB_DIR}"