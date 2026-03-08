import tomllib
import tomli_w
import subprocess
import re

# Получаем текущие версии пакетов
result = subprocess.run(['uv', 'pip', 'list'], capture_output=True, text=True)
lines = result.stdout.split('\n')[2:-1]  # Пропускаем заголовки

# Создаем словарь пакетов
packages = {}
for line in lines:
    if line.strip():
        parts = line.split()
        if len(parts) >= 2:
            packages[parts[0].lower()] = parts[1]

# Читаем текущий pyproject.toml
with open('pyproject.toml', 'rb') as f:
    pyproject = tomllib.load(f)

# Обновляем зависимости
if 'project' in pyproject and 'dependencies' in pyproject['project']:
    new_deps = []
    for dep in pyproject['project']['dependencies']:
        # Извлекаем имя пакета (до версии или спецсимволов)
        package_name = re.split(r'[>=<~!]', dep)[0].strip().lower()
        if package_name in packages:
            new_deps.append(f"{package_name}=={packages[package_name]}")
        else:
            new_deps.append(dep)
    
    pyproject['project']['dependencies'] = new_deps

# Сохраняем обновленный pyproject.toml
with open('pyproject.toml', 'wb') as f:
    tomli_w.dump(pyproject, f)

print("pyproject.toml обновлен с фиксированными версиями!")