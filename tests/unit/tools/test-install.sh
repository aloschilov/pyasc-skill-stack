#!/usr/bin/env bash
# Exercise ownership and idempotence using a disposable skill tree and HOME.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export HOME="$scratch/home"
mkdir -p "$HOME" "$scratch/source tree/skills/example" "$scratch/target directory"
cp "$ROOT/install.sh" "$scratch/source tree/install.sh"
printf '%s\n' '---' 'name: example' 'description: Test installer ownership.' '---' > "$scratch/source tree/skills/example/SKILL.md"
source_tree="$scratch/source tree"
target="$scratch/target directory"

bash "$source_tree/install.sh" --target "$target"
bash "$source_tree/install.sh" --target "$target"
test "$(readlink "$target/example")" = "$source_tree/skills/example"
test ! -e "$HOME/.cursor"
test ! -e "$HOME/.config"

# Neither live nor dangling foreign links may be replaced.
mkdir "$scratch/foreign"
for destination in "$scratch/foreign" "$scratch/missing"; do
    rm "$target/example"
    ln -s "$destination" "$target/example"
    if bash "$source_tree/install.sh" --target "$target"; then
        echo 'FAIL: foreign link was accepted'; exit 1
    fi
    test "$(readlink "$target/example")" = "$destination"
done
rm "$target/example"
mkdir "$target/example"
printf 'preserve\n' > "$target/example/user-file"
if bash "$source_tree/install.sh" --target "$target"; then
    echo 'FAIL: existing directory was accepted'; exit 1
fi
test "$(cat "$target/example/user-file")" = preserve
rm -r "$target/example"

# Prune removed skills only when the link points into this installation.
ln -s "$source_tree/skills/removed" "$target/removed"
ln -s "$scratch/other-missing" "$target/unrelated"
bash "$source_tree/install.sh" --target "$target"
test ! -L "$target/removed"
test -L "$target/unrelated"
# Renames migrate only owned dangling links; foreign historical links survive.
ln -s "$source_tree/skills/pyasc-env-check" "$target/pyasc-env-check"
mkdir "$source_tree/skills/asctile-env-check"
printf '%s\n' '---' 'name: asctile-env-check' 'description: Renamed skill fixture.' '---' > "$source_tree/skills/asctile-env-check/SKILL.md"
bash "$source_tree/install.sh" --target "$target"
test ! -L "$target/pyasc-env-check"
test "$(readlink "$target/asctile-env-check")" = "$source_tree/skills/asctile-env-check"
ln -s "$scratch/other-missing" "$target/pyasc-env-check"
bash "$source_tree/install.sh" --target "$target"
test -L "$target/pyasc-env-check"
echo 'PASS: installer preserves foreign content and prunes only owned stale links'
