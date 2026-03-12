# Деплой на сервере

## Требования к серверу

| Параметр | Минимум | Рекомендуется |
|----------|---------|---------------|
| RAM | 8 GB | 16 GB |
| Диск | 30 GB | 50 GB |
| CPU | 4 vCPU | 8 vCPU |
| ОС | Ubuntu 22.04+ / Debian 12+ | Ubuntu 22.04 LTS |

ML-инференс работает на CPU, GPU не нужен. LLM-запросы идут через OpenRouter API.

---

## Деплой через Docker (рекомендуется)

### 1. Установить Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Склонировать репозиторий

```bash
git clone <repo-url> nlp-hw
cd nlp-hw
```

### 3. Создать `.env`

```bash
cp .env.example .env
nano .env
```

Вставить API-ключ:

```
OPENROUTER_API_KEY=sk-or-...
```

### 4. Собрать Docker-образ

Первый раз занимает ~15–20 минут (компилируется `pywhispercpp`):

```bash
make docker-build
```

### 5. Скачать ML-модели (~5 GB, один раз)

Модели сохраняются в именованный Docker-volume `model_cache` и переживают пересборку образа:

```bash
make docker-models
```

### 6. Добавить пользователя

```bash
make docker-add-user U=admin P=yourpassword D="Admin"
```

### 7. Запустить

```bash
make docker-up
```

Приложение доступно на `http://ваш-сервер:8501`.

---

## Управление

```bash
make docker-logs      # смотреть логи в реальном времени
make docker-down      # остановить
make docker-shell     # bash внутри контейнера
make docker-add-user U=user2 P=pass D="User 2"  # добавить ещё одного пользователя
```

Перезапуск после обновления кода:

```bash
git pull
make docker-build
make docker-up
```

---

## Настройка через переменные окружения

Все параметры можно переопределить в `.env`:

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `OPENROUTER_API_KEY` | — | **Обязательна** |
| `PORT` | `8501` | Порт Streamlit |
| `WHISPER_MODEL` | `large-v3-turbo-q5_0` | Модель Whisper |
| `LLM_MODEL` | `nvidia/nemotron-nano-9b-v2:free` | Модель через OpenRouter |
| `WHISPER_THREADS` | `4` | Потоков на один воркер Whisper |
| `WHISPER_WORKERS` | `2` | Параллельных воркеров Whisper |
| `LANGUAGE` | `ru` | Язык транскрипции |

На сервере с 8 ядрами оптимально: `WHISPER_THREADS=4`, `WHISPER_WORKERS=2`.

---

## Nginx + HTTPS (опционально)

Если нужен публичный доступ через домен:

### Установить Nginx и Certbot

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

### Конфиг `/etc/nginx/sites-available/nlp-hw`

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;

        # WebSocket — обязательно для Streamlit
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 7200;  # 2 часа — длинные видео
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/nlp-hw /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Получить HTTPS-сертификат
sudo certbot --nginx -d yourdomain.com
```

---

## Данные и бэкап

| Путь на хосте | Содержимое |
|---------------|------------|
| `./data/users.yaml` | Пользователи (bcrypt-хеши) |
| `./data/chats/` | История чатов по пользователям |
| `./output/` | Транскрипты и FAISS-индексы видео |
| Docker volume `model_cache` | ML-модели (~5 GB) |

Для бэкапа достаточно скопировать `./data/` и `./output/`. Модели можно скачать заново командой `make docker-models`.

---

## Диагностика

```bash
# Проверить, что контейнер работает
docker compose ps

# Логи с момента старта
make docker-logs

# Проверить health check
curl http://localhost:8501/_stcore/health

# Сколько памяти использует контейнер
docker stats nlp-hw-app-1
```

### Частые проблемы

**Контейнер падает при старте**
```bash
make docker-logs  # посмотреть ошибку
# Обычно: .env не заполнен или data/users.yaml не существует
```

**`make docker-up` говорит "No users found"**
```bash
make docker-add-user U=admin P=pass D="Admin"
```

**Ошибка "Out of memory" при обработке видео**
— Увеличить RAM или уменьшить `WHISPER_WORKERS=1`.

**Модели скачиваются каждый раз заново**
— Volume `model_cache` не примонтирован. Проверить `docker volume ls | grep model_cache`.
