#!/bin/bash
# Обновление курсов валют с логгированием

LOG_FILE="/root/currency-analyzer/logs/cron_refresh.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting currency refresh..." >> $LOG_FILE

# Выполняем запрос
RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/api/force-refresh)

# Проверяем результат
if echo "$RESPONSE" | grep -q '"status":"success"'; then
    echo "[$DATE] ✓ SUCCESS: $RESPONSE" >> $LOG_FILE
else
    echo "[$DATE] ✗ ERROR: $RESPONSE" >> $LOG_FILE
fi

echo "----------------------------------------" >> $LOG_FILE
