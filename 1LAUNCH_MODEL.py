import os
import torch
import speech_recognition as sr
from transformers import LlamaForCausalLM, LlamaTokenizer

max_split_size_mb = 23 * 1024  # Мегабайты

# Определение объекта Recognizer
recognizer = sr.Recognizer()

# Путь к папке с файлами модели 
model_dir = "D:/Programs/NDLGPT/modelformix/my"

# Проверка на наличие модели
if os.path.exists(model_dir):
    print("Модель проверяется на наличие...")
    print("\033[92mУспех!\033[0m Модель найдена и загружается...")
else:
    print("\033[91mОшибка!\033[0m Модель не найдена. Убедитесь, что путь указан верно.")
    exit()

# Использование CUDA для работы с видеокартой, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация токенизатора и модели на устройстве "cuda"
tokenizer = LlamaTokenizer.from_pretrained(model_dir, local_files_only=True)
model = LlamaForCausalLM.from_pretrained(model_dir, local_files_only=True)
model.to(device)
model.eval()


def generate_response(prompt, tokenizer, model, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device).long()  # Преобразование в Long
    attention_mask = torch.ones(input_ids.shape, device=device)
    with torch.no_grad():
        torch.cuda.empty_cache()  # Освобождаем часть памяти на видеокарте
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=150, do_sample=True, top_p=0.9, temperature=0.5)
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


while True:
    with sr.Microphone() as source:
        print("Говорите:")
        audio = recognizer.listen(source)
    
    try:
        user_input = recognizer.recognize_google(audio, language="ru-RU")
        print("Вы сказали:", user_input)
        
        
        print("Подождите, генерирую ответ...")
        assistant_response = generate_response(user_input, tokenizer, model, device)
        print("Ответ помощника:", assistant_response)

    except sr.UnknownValueError:
        print("Извините, не удалось распознать вашу речь.")
    except sr.RequestError:
        print("Извините, произошла ошибка при запросе к сервису распознавания речи.")
