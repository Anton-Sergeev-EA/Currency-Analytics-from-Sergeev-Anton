#!/usr/bin/env bash
#
# One-time bootstrap for HTTPS on anton-analytics.ru via the Docker Compose
# prod stack (docker-compose.yml + docker-compose.prod.yml).
#
# nginx's 443 server block needs *some* certificate file to exist just
# to start, but Let's Encrypt's webroot challenge needs nginx already
# answering on port 80 to prove domain ownership - a chicken-and-egg
# problem. This script breaks that cycle the standard way: it drops in
# a throwaway self-signed certificate so the stack can start at all,
# requests the real certificate over the now-running HTTP-01 challenge,
# then reloads nginx with the real one. It also registers a
# --deploy-hook so every future automatic renewal (certbot's own
# systemd timer/cron, installed system-wide when certbot is installed)
# reloads the nginx *container*, not a host nginx that doesn't exist here.
#
# Usage (as root, from the project root, DNS for anton-analytics.ru and
# www.anton-analytics.ru must already point at this server's IP):
#   sudo scripts/init_letsencrypt.sh you@example.com
set -euo pipefail

DOMAIN="anton-analytics.ru"
WWW_DOMAIN="www.anton-analytics.ru"
EMAIL="${1:?Usage: scripts/init_letsencrypt.sh you@example.com}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEBROOT="$PROJECT_ROOT/certbot-webroot"
LE_LIVE_DIR="/etc/letsencrypt/live/$DOMAIN"
COMPOSE="docker compose -f $PROJECT_ROOT/docker-compose.yml -f $PROJECT_ROOT/docker-compose.prod.yml"

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this as root (it writes to /etc/letsencrypt and manages Docker)." >&2
    exit 1
fi

mkdir -p "$WEBROOT/.well-known/acme-challenge"

if [ ! -f "$LE_LIVE_DIR/fullchain.pem" ]; then
    echo "==> No certificate yet for $DOMAIN - creating a temporary self-signed one so nginx can start on :443..."
    mkdir -p "$LE_LIVE_DIR"
    openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
        -keyout "$LE_LIVE_DIR/privkey.pem" \
        -out "$LE_LIVE_DIR/fullchain.pem" \
        -subj "/CN=$DOMAIN" >/dev/null 2>&1
fi

echo "==> Starting the stack (nginx will serve the temporary certificate for now)..."
cd "$PROJECT_ROOT"
$COMPOSE up -d

if ! command -v certbot >/dev/null 2>&1; then
    echo "==> certbot not found - installing (Debian/Ubuntu apt)..."
    apt-get update -qq && apt-get install -y -qq certbot
fi

echo "==> Requesting the real Let's Encrypt certificate via the HTTP-01 (webroot) challenge..."
rm -rf "$LE_LIVE_DIR"   # drop the temporary self-signed cert so certbot writes fresh, real files
certbot certonly \
    --webroot -w "$WEBROOT" \
    -d "$DOMAIN" -d "$WWW_DOMAIN" \
    --email "$EMAIL" --agree-tos --non-interactive \
    --deploy-hook "$COMPOSE exec -T nginx nginx -s reload"

echo "==> Reloading nginx with the real certificate..."
$COMPOSE exec -T nginx nginx -s reload

echo "==> Done. https://$DOMAIN should now be live with a trusted certificate."
echo "    Renewal is automatic (certbot's own systemd timer/cron, installed with the package)"
echo "    and will reload the nginx container via the --deploy-hook registered above."
