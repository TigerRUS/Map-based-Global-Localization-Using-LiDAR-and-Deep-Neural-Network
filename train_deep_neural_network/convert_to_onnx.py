import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import onnx
import onnxruntime as ort
import warnings
warnings.filterwarnings('ignore')

class SimpleLocalizationCNN(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def convert_to_onnx_with_correct_size():
    print("=" * 70)
    print("Convert to ONNX")
    print("=" * 70)
    
    BATCH_SIZE = 1
    CHANNELS = 2
    HEIGHT = 384   # INPUT_HEIGHT
    WIDTH = 320    # INPUT_WIDTH
    
    MODEL_PATH = 'localization_model.pth'
    ONNX_PATH = 'localization_model.onnx'
    
    print(f"\n📊 Параметры модели:")
    print(f"   Входные размеры: {BATCH_SIZE} x {CHANNELS} x {HEIGHT} x {WIDTH}")
    print(f"   (HEIGHT={HEIGHT}, WIDTH={WIDTH})")
    
    # Загрузка модели
    print(f"\n1. Загрузка модели из {MODEL_PATH}")
    device = torch.device('cpu')
    model = SimpleLocalizationCNN(input_channels=CHANNELS)
    
    # Загрузка весов
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"   ✓ Модель загружена успешно")
        print(f"   Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ Ошибка загрузки модели: {e}")
        return None, None
    
    # Создание примера входных данных с правильными размерами
    print(f"\n2. Создание примера входных данных...")
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, device=device)
    print(f"   Форма входного тензора: {dummy_input.shape}")
    
    # Проверка forward pass
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"   Форма выходного тензора: {output.shape}")
            print(f"   ✓ Forward pass успешен")
        except Exception as e:
            print(f"   ✗ Ошибка forward pass: {e}")
            return None, None
    
    # Экспорт в ONNX
    print(f"\n3. Экспорт в ONNX формат...")
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],      # входной тензор
        output_names=['output'],    # выходной тензор (x, y, sin, cos)
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"   ✓ Модель сохранена в {ONNX_PATH}")
    
    # Проверка ONNX модели
    print(f"\n4. Проверка ONNX модели...")
    try:
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("   ✓ Модель прошла проверку!")
    except Exception as e:
        print(f"   ✗ Ошибка проверки: {e}")
        return None, None
    
    # Тестирование с ONNX Runtime
    print(f"\n5. Тестирование с ONNX Runtime...")
    
    # Создаем сессию ONNX Runtime
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    # Подготовка входных данных
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    
    # Инференс
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Сравнение результатов PyTorch и ONNX
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()
    
    max_diff = np.max(np.abs(pytorch_output - ort_outputs[0]))
    print(f"   Максимальная разница: {max_diff:.6f}")
    
    if max_diff < 1e-4:
        print("   ✓ Результаты совпадают!")
    else:
        print(f"   ⚠ Предупреждение: небольшая разница в результатах")
    
    # Информация о модели
    print(f"\n6. Информация о модели:")
    print(f"   Вход: {ort_session.get_inputs()[0].name}")
    print(f"   Форма: {ort_session.get_inputs()[0].shape}")
    print(f"   Тип: {ort_session.get_inputs()[0].type}")
    print(f"   Выход: {ort_session.get_outputs()[0].name}")
    print(f"   Форма: {ort_session.get_outputs()[0].shape}")
    print(f"   Выходные значения: x, y, sin(angle), cos(angle)")
    
    # Размер файла
    model_size = Path(ONNX_PATH).stat().st_size / (1024 * 1024)
    print(f"   Размер файла: {model_size:.2f} MB")
    
    print("\n" + "=" * 70)
    print("✅ КОНВЕРТАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 70)
    
    return ONNX_PATH, ort_session


def test_with_random_data(onnx_path):
    """Тестирование со случайными данными правильного размера"""
    
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ СО СЛУЧАЙНЫМИ ДАННЫМИ")
    print("=" * 70)
    
    # Загрузка модели
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Создание случайных данных правильного размера
    HEIGHT = 384
    WIDTH = 320
    
    random_input = np.random.randn(1, 2, HEIGHT, WIDTH).astype(np.float32)
    print(f"\nВходные данные: {random_input.shape}")
    
    # Инференс
    ort_inputs = {ort_session.get_inputs()[0].name: random_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    pose = ort_outputs[0][0]
    
    print("\nРезультат предсказания:")
    print(f"  x = {pose[0]:.4f}")
    print(f"  y = {pose[1]:.4f}")
    print(f"  sin(angle) = {pose[2]:.4f}")
    print(f"  cos(angle) = {pose[3]:.4f}")
    
    # Восстановление угла
    angle = np.arctan2(pose[2], pose[3])
    angle_deg = np.degrees(angle)
    print(f"  angle = {angle_deg:.2f}°")
    
    # Проверка нормы sin/cos
    norm = np.sqrt(pose[2]**2 + pose[3]**2)
    print(f"  sin²+cos² = {norm:.4f} (должно быть ≈1.0)")
    
    return pose


def benchmark_performance(onnx_path, num_runs=1000):
    """Бенчмаркинг производительности"""
    
    print("\n" + "=" * 70)
    print("БЕНЧМАРКИНГ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 70)
    
    import time
    
    # Загрузка модели
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Создание тестовых данных правильного размера
    HEIGHT = 384
    WIDTH = 320
    test_input = np.random.randn(1, 2, HEIGHT, WIDTH).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    
    # Прогрев
    print("Прогрев модели...")
    for _ in range(50):
        _ = ort_session.run(None, ort_inputs)
    
    # Измерение времени
    print(f"Запуск {num_runs} инференсов...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # в миллисекундах
    
    print(f"\nРезультаты ({num_runs} запусков):")
    print(f"  Среднее время: {avg_time:.3f} мс")
    print(f"  Частота: {1000/avg_time:.1f} FPS")
    print(f"  Минимальное время: ~{avg_time*0.8:.3f} мс")
    print(f"  Максимальное время: ~{avg_time*1.2:.3f} мс")
    
    return avg_time


def usage_example():
    """Пример использования ONNX модели в коде"""
    
    print("\n" + "=" * 70)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ")
    print("=" * 70)
    
    example_code = '''
import onnxruntime as ort
import numpy as np

# 1. Загрузка модели
session = ort.InferenceSession('localization_model_320x384.onnx')

# 2. Подготовка входных данных (размеры: 384x320)
# map_img и scan_img должны быть numpy массивами размера 384x320
map_img = ...  # ваша карта, float32, нормализована [0,1]
scan_img = ... # ваш скан, float32, нормализован [0,1]

# 3. Объединение каналов
input_tensor = np.stack([map_img, scan_img], axis=0)  # (2, 384, 320)
input_tensor = input_tensor[np.newaxis, ...]          # (1, 2, 384, 320)
input_tensor = input_tensor.astype(np.float32)

# 4. Инференс
ort_inputs = {session.get_inputs()[0].name: input_tensor}
result = session.run(['output'], ort_inputs)

# 5. Получение результата
pose = result[0][0]  # [x, y, sin(angle), cos(angle)]
x, y, sin_angle, cos_angle = pose

# 6. Восстановление угла
angle_rad = np.arctan2(sin_angle, cos_angle)
angle_deg = np.degrees(angle_rad)

print(f"Position: ({x:.3f}, {y:.3f})")
print(f"Angle: {angle_deg:.1f}°")
'''
    
    print(example_code)


def save_model_info(onnx_path):
    """Сохранение информации о модели в файл"""
    
    info = {
        'model_name': 'Localization CNN',
        'input_shape': [1, 2, 384, 320],
        'input_description': '2 канала: карта (канал 0) и скан (канал 1), нормализованные [0,1]',
        'output_shape': [1, 4],
        'output_description': 'x, y, sin(angle), cos(angle)',
        'angle_recovery': 'angle = atan2(sin, cos)',
        'onnx_opset': 14,
        'file_size_mb': Path(onnx_path).stat().st_size / (1024 * 1024)
    }
    
    info_path = Path(onnx_path).with_suffix('.json')
    import json
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n📄 Информация о модели сохранена в {info_path}")


if __name__ == '__main__':
    try:
        # Конвертация
        onnx_path, session = convert_to_onnx_with_correct_size()
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        print("Убедитесь, что файл localization_model.pth существует")
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()