#!/bin/bash
set -e

echo "Setting up Currency Analytics service..."
cp currency_analytics.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable currency_analytics
systemctl restart currency_analytics

# cron_setup.txt is the single source of truth for the refresh schedule --
# this used to hardcode a second, different schedule inline (10:30 and
# 16:00 only) that had drifted out of sync with what cron_setup.txt
# documents. Installing straight from that file means there's only one
# place left to edit when the schedule needs to change.
echo "Setting up cron jobs from cron_setup.txt..."
CRON_LINES=$(grep -vE '^\s*#|^\s*$' cron_setup.txt)
(crontab -l 2>/dev/null; echo "$CRON_LINES") | crontab -

echo "Done! Service is running and cron is configured."
systemctl status currency_analytics --no-pager
