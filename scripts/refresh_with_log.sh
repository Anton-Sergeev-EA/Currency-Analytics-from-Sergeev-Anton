#!/bin/bash
# Обновление курсов валют с логгированием

LOG_FILE="/root/currency-analyzer/logs/cron_refresh.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting currency refresh..." >> $LOG_FILE

# /api/force-refresh now requires an X-Admin-Key header matching SECRET_KEY
# from .env (see admin.py's require_admin_key docstring - it used to be
# open to anyone on the internet with no credential at all). Read it from
# the deployment's own .env rather than hardcoding it here, so this script
# doesn't need editing every time the key rotates. Adjust ENV_FILE if your
# deployment directory isn't the one below.
ENV_FILE="/root/Currency-Analytics-from-Sergeev-Anton/.env"
ADMIN_KEY=$(grep -m1 '^SECRET_KEY=' "$ENV_FILE" 2>/dev/null | cut -d= -f2-)

# Выполняем запрос
RESPONSE=$(curl -s -X POST -H "X-Admin-Key: $ADMIN_KEY" http://127.0.0.1:8000/api/force-refresh)

# Проверяем результат
if echo "$RESPONSE" | grep -q '"status":"success"'; then
    echo "[$DATE] ✓ SUCCESS: $RESPONSE" >> $LOG_FILE
else
    echo "[$DATE] ✗ ERROR: $RESPONSE" >> $LOG_FILE
fi

echo "----------------------------------------" >> $LOG_FILE
