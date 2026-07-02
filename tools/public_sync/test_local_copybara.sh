#!/usr/bin/env bash
set -euo pipefail

history="${HISTORY:-full}"
submodules="${SUBMODULES:-off}"

case "$history" in
  full | first-parent) ;;
  *) echo "HISTORY must be full or first-parent" >&2; exit 2 ;;
esac

case "$submodules" in
  off | recursive) ;;
  *) echo "SUBMODULES must be recursive or off" >&2; exit 2 ;;
esac

tmpdir="$(mktemp -d)"
if [ "${KEEP_TMP:-0}" = 1 ]; then
  echo "keeping temp dir: $tmpdir"
else
  trap 'rm -rf "$tmpdir"' EXIT
fi

git clone --bare . "$tmpdir/source.git"
git --git-dir="$tmpdir/source.git" update-ref refs/heads/formosa/main HEAD
git init --bare "$tmpdir/dest.git"

cp tools/public_sync/copy.bara.sky "$tmpdir/copy.bara.sky"
perl -0pi -e "s#GITLAB_REPO = \"[^\"]+\"#GITLAB_REPO = \"file://$tmpdir/source.git\"#; s#GITHUB_REPO = \"[^\"]+\"#GITHUB_REPO = \"file://$tmpdir/dest.git\"#" "$tmpdir/copy.bara.sky"

if [ "$history" = first-parent ]; then
  perl -0pi -e 's/first_parent = False,/first_parent = True,/' "$tmpdir/copy.bara.sky"
fi

if [ "$submodules" = recursive ]; then
  perl -0pi -e 's/submodules = "NO",/submodules = "RECURSIVE",/' "$tmpdir/copy.bara.sky"
fi

mkdir -p "$tmpdir/home"
HOME="$tmpdir/home" git config --global user.name "Local Copybara Test"
HOME="$tmpdir/home" git config --global user.email "copybara-test@example.invalid"
HOME="$tmpdir/home" copybara migrate "$tmpdir/copy.bara.sky" public_mirror --init-history --force

git clone "$tmpdir/dest.git" "$tmpdir/public-work"
<<<<<<< Updated upstream
git -C "$tmpdir/public-work" checkout main
git -C "$tmpdir/public-work" log --oneline --decorate -5
git -C "$tmpdir/public-work" rev-list --count main
=======
git -C "$tmpdir/public-work" checkout formosa/main
git -C "$tmpdir/public-work" log --oneline --decorate -5
git -C "$tmpdir/public-work" rev-list --count formosa/main
>>>>>>> Stashed changes
if rg -n "git@git\\.caslab|git\\.caslab|wiki\\.caslab" "$tmpdir/public-work"; then
  echo "private URLs remain in public checkout" >&2
  exit 1
fi

echo "$tmpdir/public-work"
