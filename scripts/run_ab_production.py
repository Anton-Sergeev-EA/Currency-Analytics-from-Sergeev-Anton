import json
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent / 'src'))


def test_ab_endpoints(base_url="http://localhost:8002"):
    """Смоук-тест эндпоинтов A/B-теста на запущенном сервере."""

    print(f"Тестирование A/B-теста на {base_url}")

    print("\nПолучение статуса A/B-теста:")
    response = requests.get(f"{base_url}/api/ab-test/status")
    print(f"Статус: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    print("\nТестовые прогнозы:")
    for i in range(5):
        params = {"user_id": f"test_user_{i}", "currency": "usd_rate", "days": 3}
        response = requests.post(f"{base_url}/api/ab-test/predict", params=params)
        result = response.json()
        if response.status_code != 200:
            print(f"Прогноз {i + 1}: ошибка {response.status_code} - {result}")
            continue
        print(f"Прогноз {i + 1}: вариант {result['variant']} ({result['model_name']}), курс: {result['predicted_rate']:.4f}")
        time.sleep(0.1)

    print("\nСтатистика A/B-теста:")
    response = requests.get(f"{base_url}/api/ab-test/stats", params={"days": 30})
    stats = response.json()
    print(f"   {json.dumps(stats, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    test_ab_endpoints()
