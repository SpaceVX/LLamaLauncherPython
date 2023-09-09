import os
import torch
import pyttsx3
import subprocess
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader

# Приветствие и запрос пути к папке с моделью
print("Хай! Вас приветствует АмНям Тренер моделей LLAMA GPT!")
print("Папка с тренированной моделью будет в папке model_output!")

model_dir = input("Введите полный путь папки, где находится модель/модели для тренировки: ")
output_dir = "model_output"

if not os.path.exists(output_dir):
    print("Модель не найдена. Установка модели...")

    model = LlamaForCausalLM.from_pretrained(model_dir)
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Установка модели завершена. Введите 'done'.")

else:
    print("Модель уже установлена. Процедура установки пропущена.")

while True:
    user_input = input("Вы: ")
    
    if user_input.strip().lower() == "done":
        print("Модель загружается. Введите команду для обучения.")
        model = LlamaForCausalLM.from_pretrained(output_dir)
        tokenizer = LlamaTokenizer.from_pretrained(output_dir)
        break

print("Скажите команду для обучения!")

dialogs = []

def train_new_command(dialogs, new_command, new_action):
    dialogs.append((new_command, new_action))
    dataset = CustomDataset(dialogs, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 5
    total_steps = len(dataloader) * num_epochs
    current_step = 0

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            current_step += 1
            progress = current_step / total_steps * 100
            print(f"Эпоха [{epoch+1}/{num_epochs}], Прогресс: {progress:.2f}%")

    print("Тренировка успешно выполнена! Модель обучена новой команде!")

class CustomDataset(Dataset):
    def __init__(self, dialogs, tokenizer, max_length):
        self.dialogs = dialogs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        question, answer = self.dialogs[idx]
        encoded_input = self.tokenizer.encode(question, add_special_tokens=False)
        encoded_output = self.tokenizer.encode(answer, add_special_tokens=False)
        return {"input_ids": encoded_input, "labels": encoded_output}

dataset = CustomDataset(dialogs, tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 10
total_steps = len(dataloader) * num_epochs
current_step = 0

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        current_step += 1
        progress = current_step / total_steps * 100
        print(f"Эпоха [{epoch+1}/{num_epochs}], Прогресс: {progress:.2f}%")

print("Тренировка успешно выполнена! Файл перезаписан и готов к использованию!")

def execute_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении команды: {e}")

def train_new_command(dialogs, new_command, new_action):
    dialogs.append((new_command, new_action))
    dataset = CustomDataset(dialogs, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 5
    total_steps = len(dataloader) * num_epochs
    current_step = 0

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            current_step += 1
            progress = current_step / total_steps * 100
            print(f"Эпоха [{epoch+1}/{num_epochs}], Прогресс: {progress:.2f}%")

def voice_assistant(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if "открыть браузер" in generated_text.lower():
        execute_command("start chrome")
    elif "открыть папку" in generated_text.lower():
        execute_command("start explorer")
    # Добавьте другие команды и действия здесь
    
    print("Модель:", generated_text)
    engine = pyttsx3.init()
    engine.say(generated_text)
    engine.runAndWait()

while True:
    user_input = input("Вы: ")
    
    if user_input.lower() == "обучи новую команду":
        new_command = input("Введите новую команду: ")
        new_action = input(f"Введите действие для команды '{new_command}': ")
        train_new_command(dialogs, new_command, new_action)
    else:
        voice_assistant(user_input)
