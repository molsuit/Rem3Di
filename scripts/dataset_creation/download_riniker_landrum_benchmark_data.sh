#!/usr/bin/env bash
#
#
# Downloads Riniker Landrum Benchmark
# Recursively gunzips all *.gz, then moves every *.dat it discovers
# into the directory you launched this script from.
#
# Usage:
#   ./unpack-and-collect.sh           # scan the current directory tree
#   ./unpack-and-collect.sh /path/to/tree
#
# Notes:
#   • By default the .gz originals are KEPT.   (Change --keep to --force if
#     you want them removed.)
#   • Existing .dat names are never overwritten; duplicates get a –1, –2 … suffix.
#   • Requires GNU coreutils + GNU gzip (standard on Linux; on macOS use Homebrew).

curl -L "https://static-content.springer.com/esm/art%3A10.1186%2F1758-2946-5-26/MediaObjects/13321_2013_467_MOESM5_ESM.gz" \
| tar -xz

set -euo pipefail

# ---------- configuration ---------------------------------------------------
root_dir="${1:-.}"         # where to start searching
dest_dir="$PWD"            # where the .dat files end up
keep_gz=true               # set false to delete .gz after extraction
# ---------------------------------------------------------------------------



echo "🔍  Scanning: $root_dir"
echo "📦  Destination for .dat files: $dest_dir"
echo

# 1 / 2  ── decompress every *.gz we can find
echo "➤  Decompressing .gz files…"
find "$root_dir" -type f -name '*.gz' -print0 |
while IFS= read -r -d '' gzfile; do
    if [[ "$keep_gz" == true ]]; then
        gzip --keep --decompress "$gzfile"        # leaves original .gz
    else
        gzip --force --decompress "$gzfile"       # removes original .gz
    fi
done
echo "    done."

# 3 ── collect *.dat files
echo "➤  Gathering .dat files…"
find "$root_dir" -type f -name '*.dat' -print0 |
while IFS= read -r -d '' datfile; do
    base="$(basename "$datfile")"
    target="$dest_dir/$base"
    # Avoid clobbering if a file with the same name already exists here
    if [[ -e "$target" ]]; then
        n=1
        while [[ -e "${dest_dir}/${base%.*}-$n.dat" ]]; do
            ((n++))
        done
        target="${dest_dir}/${base%.*}-$n.dat"
    fi
    mv -- "$datfile" "$target"
done
echo "    done."

echo -e "\n✅  All .gz files extracted and .dat files collected."
