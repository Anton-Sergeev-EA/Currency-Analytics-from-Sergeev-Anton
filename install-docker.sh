#!/usr/bin/env bash
#
# Full VDS bootstrap for the Docker Compose deployment at
# https://anton-analytics.ru: installs Docker if missing, makes sure Docker
# itself survives a reboot, installs the systemd unit that brings the
# whole stack (app + nginx, both docker-compose.yml and
# docker-compose.prod.yml) back up on every boot, and - if an e-mail is
# given - obtains the Let's Encrypt certificate.
#
# Usage (as root, from the project root):
#   sudo ./install-docker.sh you@example.com      # full setup incl. HTTPS
#   sudo ./install-docker.sh                       # skip the cert step
#     (run scripts/init_letsencrypt.sh yourself later, once DNS is ready)
#
# Assumes DNS A-records for anton-analytics.ru and www.anton-analytics.ru
# already point at this server - HTTPS bootstrap fails otherwise.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
EMAIL="${1:-}"

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this as root." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "==> Docker not found - installing via the official convenience script..."
    curl -fsSL https://get.docker.com | sh
fi

echo "==> Enabling Docker to start on boot (so restart:always containers come back after a reboot)..."
systemctl enable --now docker

echo "==> Copying .env.example to .env if you don't already have one..."
[ -f "$PROJECT_ROOT/.env" ] || cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
echo "    Review $PROJECT_ROOT/.env (especially SECRET_KEY / JWT_SECRET_KEY) before going further."

echo "==> Installing the systemd unit for the Compose stack..."
sed "s#/root/Currency-Analytics-from-Sergeev-Anton#$PROJECT_ROOT#" \
    "$PROJECT_ROOT/currency-analytics-docker.service" > /etc/systemd/system/currency-analytics-docker.service
systemctl daemon-reload
systemctl enable currency-analytics-docker

if [ -n "$EMAIL" ]; then
    echo "==> Bootstrapping HTTPS for anton-analytics.ru via Let's Encrypt..."
    "$PROJECT_ROOT/scripts/init_letsencrypt.sh" "$EMAIL"
else
    echo "==> Starting the stack over plain HTTP (no e-mail given, skipping the certificate step)..."
    systemctl start currency-analytics-docker
    echo "    Once DNS is confirmed, run: sudo scripts/init_letsencrypt.sh you@example.com"
fi

echo "==> Done."
systemctl status currency-analytics-docker --no-pager || true
