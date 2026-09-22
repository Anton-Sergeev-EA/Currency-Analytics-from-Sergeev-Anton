#!/bin/bash
echo "Setting up Currency Analytics service..."
cp currency_analytics.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable currency_analytics
systemctl restart currency_analytics

echo "Setting up cron jobs..."
# /api/refresh now requires an X-Admin-Key header matching SECRET_KEY from
# .env (it used to be open to anyone on the internet, no credential at
# all - see admin.py's require_admin_key docstring). Read it from .env at
# install time rather than hardcoding a value into the crontab.
ADMIN_KEY=$(grep -m1 '^SECRET_KEY=' .env 2>/dev/null | cut -d= -f2-)
(crontab -l 2>/dev/null; echo "30 10 * * * curl -X POST -H \"X-Admin-Key: $ADMIN_KEY\" http://127.0.0.1:8002/api/refresh"; echo "0 16 * * * curl -X POST -H \"X-Admin-Key: $ADMIN_KEY\" http://127.0.0.1:8002/api/refresh") | crontab -

echo "Done! Service is running and cron is configured."
systemctl status currency_analytics --no-pager
