#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)/scripts/check_release.sh"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cd "$T"
printf 'id: yolo26_detect\nversion: 1.2.0\n' > app.yaml
printf '# Changelog\n\n## [1.2.0] - 2026-10-08\n- x\n\n## [1.1.0] - 2026-10-01\n' > CHANGELOG.md
"$S" v1.2.0 >/dev/null || { echo "FAIL: matching tag rejected"; exit 1; }
! "$S" v1.1.0 >/dev/null 2>&1 || { echo "FAIL: wrong tag accepted"; exit 1; }
! "$S" 1.2.0 >/dev/null 2>&1 || { echo "FAIL: tag without v accepted"; exit 1; }
printf '# Changelog\n\n## [1.1.0] - 2026-10-01\n' > CHANGELOG.md
! "$S" v1.2.0 >/dev/null 2>&1 || { echo "FAIL: stale changelog accepted"; exit 1; }
# the unreleased placeholder date must be rejected (set the real date at publish)
printf '# Changelog\n\n## [1.2.0] - YYYY-MM-DD\n- x\n' > CHANGELOG.md
! "$S" v1.2.0 >/dev/null 2>&1 || { echo "FAIL: placeholder-date header accepted"; exit 1; }
OUT=$("$S" v1.2.0 2>&1 || true); [[ "$OUT" == *"real date"* ]] || { echo "FAIL: placeholder message unclear"; exit 1; }
printf '# Changelog\n\n## [1.2.0] - 2026-10-08\n- x\n' > CHANGELOG.md
"$S" v1.2.0 >/dev/null || { echo "FAIL: real-date header rejected"; exit 1; }
echo "check_release: ok"
