import torch

try:
    print("Попытка загрузить модель...")
    model_state = torch.load('models/ProECGNetbeststtc.pth', map_location='cpu')
    print("Модель успешно загружена!")
    print("Содержимое:", list(model_state.keys()))
except Exception as e:
    print(f"Ошибка загрузки: {e}")