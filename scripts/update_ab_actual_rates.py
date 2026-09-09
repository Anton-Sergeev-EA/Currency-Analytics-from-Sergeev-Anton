import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ab_testing.ab_service import get_ab_service

def main():
    """Обновляет фактические курсы для A/B-теста."""
    print("Обновление фактических курсов для A/B-теста.")
    
    service = get_ab_service()
    service.update_actual_rates()
    
    print("Фактические курсы обновлены!")
    
    stats = service.get_ab_stats(days=30)
    print("\n Текущая статистика A/B-теста:")
    print(f"- Модель A: MAE = {stats.get('variant_A', {}).get('mae', 'N/A')}")
    print(f"- Модель B: MAE = {stats.get('variant_B', {}).get('mae', 'N/A')}")
    
    if 'statistical_significance' in stats:
        sig = stats['statistical_significance']
        print(f"- P-значение: {sig['p_value']:.6f}")
        print(f"- Статистически значимо: {sig['is_significant']}")
        print(f"- Рекомендация: {stats.get('recommendation', 'N/A')}")

if __name__ == "__main__":
    main()
