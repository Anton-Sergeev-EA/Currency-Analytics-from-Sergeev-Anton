import asyncio
import sys
from pathlib import Path

# ab_service.py imports things as `src.xxx` (absolute, from the repo
# root), so the repo root - not just src/ - needs to be on sys.path.
sys.path.append(str(Path(__file__).parent.parent))

from src.ab_testing.ab_service import get_ab_service


async def main():
    """Обновляет фактические курсы для A/B-теста."""
    print("Обновление фактических курсов для A/B-теста...")

    service = get_ab_service()
    updated = await service.update_actual_rates()
    print(f"Обновлено записей: {updated}")

    stats = service.get_ab_stats(days=30)
    print("\nТекущая статистика A/B-теста:")
    print(f"- Вариант A (ml_ensemble): MAE = {stats.get('variant_A', {}).get('mae', 'N/A')}")
    print(f"- Вариант B (persistence_baseline): MAE = {stats.get('variant_B', {}).get('mae', 'N/A')}")

    if 'statistical_significance' in stats:
        sig = stats['statistical_significance']
        print(f"- P-значение: {sig['p_value']:.6f}")
        print(f"- Статистически значимо: {sig['is_significant']}")
    if 'recommendation' in stats:
        print(f"- Рекомендация: {stats['recommendation']}")


if __name__ == "__main__":
    asyncio.run(main())
