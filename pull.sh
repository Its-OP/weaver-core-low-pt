#!/bin/bash
# =============================================================================
# Pull latest changes for the weaver repository.
#
# Usage:
#   bash pull.sh              # pull from master
#   bash pull.sh my-branch    # pull from a specific branch
# =============================================================================
set -euo pipefail

BRANCH="${1:-main}"

echo "Pulling branch '${BRANCH}' for weaver..."
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"
echo "Done. weaver is now on branch '${BRANCH}'."
