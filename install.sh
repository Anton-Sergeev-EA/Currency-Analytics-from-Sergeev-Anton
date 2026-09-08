#!/bin/bash
echo "Setting up Currency Analytics service..."
cp currency_analytics.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable currency_analytics
systemctl restart currency_analytics

echo "Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "30 10 * * * curl -X POST http://127.0.0.1:8000/api/refresh"; echo "0 16 * * * curl -X POST http://127.0.0.1:8000/api/refresh") | crontab -

echo "Done! Service is running and cron is configured."
systemctl status currency_analytics --no-pager
