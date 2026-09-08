import sys
import os
from pathlib import Path
import requests
import json
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ab_testing.ab_service import get_ab_service

def test_ab_endpoints(base_url="http://localhost:8000"):
    """Тестирует эндпоинты A/B-теста."""
    
    print(f"Тестирование A/B-теста на {base_url}")
    
    print("\nПолучение статуса A/B-теста:")
    response = requests.get(f"{base_url}/api/ab-test/status")
    print(f"Статус: {json.dumps(response.json(), indent=2)}")
    
    print("\nТестовые прогнозы:")
    for i in range(5):
        payload = {
            "currency_pair": "USD/RUB",
            "forecast_days": 3,
            "user_id": f"test_user_{i}"
        }
        response = requests.post(f"{base_url}/api/ab-test/predict", json=payload)
        result = response.json()
        print(f"Прогноз {i+1}: Вариант {result['variant']}, Курс: {result['predicted_rate']:.4f}")
        time.sleep(0.1)
    
    print("\nСтатистика A/B-теста:")
    response = requests.get(f"{base_url}/api/ab-test/stats?days=1")
    stats = response.json()
    print(f"   {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    test_ab_endpoints()
