#!/usr/bin/env bash
# Release gate: tag, app.yaml version and top CHANGELOG entry must agree.
set -euo pipefail
TAG=${1:?usage: check_release.sh vX.Y.Z}
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "tag '$TAG' is not vX.Y.Z" >&2; exit 1; }
V=${TAG#v}
APP=$(sed -n 's/^version:[[:space:]]*//p' app.yaml | head -1)
LOG=$(sed -n 's/^## \[\([0-9.]*\)\].*/\1/p' CHANGELOG.md | head -1)
[ "$APP" = "$V" ] || { echo "app.yaml version $APP != tag $V" >&2; exit 1; }
[ "$LOG" = "$V" ] || { echo "CHANGELOG top entry $LOG != tag $V" >&2; exit 1; }
echo "release $V: tag, app.yaml and CHANGELOG agree"
