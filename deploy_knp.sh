#!/bin/bash
###############################################################################
# deploy_knp.sh — развёртывание Kaspersky Neuromorphic Platform 2.0.0
#                 с эмулятором нейроморфного процессора AltAI-2
#
# Применение:
#   chmod +x deploy_knp.sh
#   ./deploy_knp.sh             # Docker-режим (рекомендуется, любой Linux)
#   ./deploy_knp.sh --native    # нативная установка (Debian 12 / Ubuntu 22.04–24.04)
#
# Что делает Docker-режим:
#   1. Ставит Docker (apt), если его нет
#   2. Скачивает официальный SDK-образ kasperskydh/knp-sdk-image
#      (Debian 12, Python 3.11, Boost 1.81, KNP 2.0.0, AltAI backend 1.0.3)
#   3. Прогревает образ (доустановка CUDA внутри) и сохраняет снапшот knp-sdk:cuda
#   4. Скачивает датасет MNIST и создаёт ~/knp/run_knp.sh — ярлык запуска
#   5. Проверяет, что эмулятор AltAI-2 загружается
#
# Почему Docker: бинарники KNP/AltAI собраны под Python 3.11/3.12 и Boost
# 1.81/1.83. На системах с другими версиями (Ubuntu 26.04, Arch и т.п.)
# нативная установка импортироваться не будет — Docker даёт эталонное окружение.
###############################################################################

set -euo pipefail

KNP_DIR="$HOME/knp"
SDK_IMAGE="kasperskydh/knp-sdk-image:latest"
SNAP_IMAGE="knp-sdk:cuda"
REL="https://github.com/KasperskyLab/knp/releases/download/v2.0.0"
RAW="https://raw.githubusercontent.com/KasperskyLab/knp/v2.0.0"

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m[!] $*\033[0m"; }
die()  { echo -e "\033[1;31m[x] $*\033[0m" >&2; exit 1; }

need_sudo() { command -v sudo >/dev/null || die "Нужен sudo."; SUDO="sudo"; }
SUDO=""

###############################################################################
# Режим 1: Docker (универсальный)
###############################################################################
deploy_docker() {
    need_sudo

    log "Проверяю Docker..."
    if ! command -v docker >/dev/null; then
        log "Устанавливаю docker.io..."
        $SUDO apt-get update -qq
        $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
        $SUDO systemctl enable --now docker
    fi
    docker --version || $SUDO docker --version

    log "Скачиваю SDK-образ $SDK_IMAGE (~2 ГБ, при первом запуске — долго)..."
    $SUDO docker pull "$SDK_IMAGE"

    log "Прогрев образа (внутри доустанавливается CUDA Toolkit) и сохранение снапшота $SNAP_IMAGE..."
    $SUDO docker rm -f knp_warmup 2>/dev/null || true
    $SUDO docker run --name knp_warmup "$SDK_IMAGE" bash -c "echo warmup done" >/dev/null 2>&1
    $SUDO docker commit -m "CUDA preinstalled" knp_warmup "$SNAP_IMAGE" >/dev/null
    $SUDO docker rm knp_warmup >/dev/null

    log "Готовлю рабочую папку $KNP_DIR..."
    mkdir -p "$KNP_DIR"
    cd "$KNP_DIR"
    if [ ! -f MNIST.bin ]; then
        wget -nv "$RAW/examples/mnist-learn/MNIST.zip"
        unzip -oq MNIST.zip
    fi

    cat > "$KNP_DIR/run_knp.sh" <<EOF
#!/bin/bash
# Shell в контейнере KNP с эмулятором AltAI-2. Папка ~/knp доступна как /work
$SUDO docker run --rm -it -v $KNP_DIR:/work $SNAP_IMAGE bash
EOF
    chmod +x "$KNP_DIR/run_knp.sh"

    cat > "$KNP_DIR/test_emulator.py" <<'EOF'
from knp.base_framework import BackendLoader
b = BackendLoader().load('/usr/lib/libknp-altai-backend.so')
print('Эмулятор AltAI-2 загружен, backend uid:', b.uid)
EOF

    log "Проверка: загружаю эмулятор AltAI-2..."
    $SUDO docker run --rm -v "$KNP_DIR:/work" "$SNAP_IMAGE" \
        python3 /work/test_emulator.py 2>&1 | grep -E "Эмулятор|Error|Traceback"

    cat > "$KNP_DIR/README_KNP.txt" <<'EOF'
KNP 2.0.0 + эмулятор AltAI-2 — памятка
======================================
Запуск shell:        ~/knp/run_knp.sh
Проверка эмулятора:  python3 /work/test_emulator.py   (внутри контейнера)

Пример (обучение на CPU, инференс на эмуляторе AltAI-2):
  cd /work && yes | mnist_learn_example -m altai -t 200 -i 100 \
    --images /work/MNIST.bin --labels /work/MNIST.target \
    --training_backend /usr/lib/libknp-cpu-single-threaded-backend.so \
    --inference_backend /usr/lib/libknp-altai-backend.so \
    --model_path /work/model_altai

Бэкенды: /usr/lib/libknp-cpu-single-threaded-backend.so  (обучение, CPU)
         /usr/lib/libknp-cpu-multi-threaded-backend.so
         /usr/lib/libknp-altai-backend.so               (эмулятор AltAI-2 / чип)

Ограничения v2.0.0: STDP на AltAI нет; ANN2SNN (golden model) требует Python 3.12.
Документация: https://support.kaspersky.com/neuromorphic-platform/2.0
EOF

    log "Готово! Запуск: $KNP_DIR/run_knp.sh"
}

###############################################################################
# Режим 2: Нативная установка deb-пакетов (Debian 12 / Ubuntu 22.04–24.04)
###############################################################################
deploy_native() {
    need_sudo
    . /etc/os-release
    log "Дистрибутив: $PRETTY_NAME"
    case "$ID-$VERSION_ID" in
        debian-12*|ubuntu-22.04*|ubuntu-24.04*) ;;
        *) warn "Не тестированный дистрибутив. Бинарники требуют Python 3.11/3.12 и Boost 1.81/1.83."
           warn "При ошибках импорта используйте режим: ./deploy_knp.sh (Docker)";;
    esac

    log "Обновляю индекс пакетов и ставлю зависимости..."
    $SUDO apt-get update -qq
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y wget unzip python3-pip

    mkdir -p "$KNP_DIR" && cd "$KNP_DIR"

    log "Скачиваю пакеты KNP 2.0.0..."
    [ -f pkg-deb.zip ] || wget -nv "$REL/pkg-deb.zip"
    [ -f knp_ann2snn.deb ] || wget -nv "$REL/knp_ann2snn.deb"
    [ -f knp-altai-backend_1.0.2_amd64.deb ] || wget -nv "$REL/knp-altai-backend_1.0.2_amd64.deb"
    [ -f knp-altai-backend-dev_1.0.2_amd64.deb ] || wget -nv "$REL/knp-altai-backend-dev_1.0.2_amd64.deb"
    [ -d pkg-deb ] || unzip -oq pkg-deb.zip -d pkg-deb

    log "Устанавливаю deb-пакеты (лицензия Мотив НТ принимается автоматически)..."
    # NB: knp-examples пропущен — в нём битая зависимость cpp-framework-dev
    (yes) | $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ./pkg-deb/knp-cpp-framework_2.0.0_amd64.deb \
        ./pkg-deb/knp-cpp-framework-dev_2.0.0_amd64.deb \
        ./pkg-deb/knp-cpu-single-threaded-backend_2.0.0_amd64.deb \
        ./pkg-deb/knp-cpu-single-threaded-backend-dev_2.0.0_amd64.deb \
        ./pkg-deb/knp-cpu-multi-threaded-backend_2.0.0_amd64.deb \
        ./pkg-deb/knp-cpu-multi-threaded-backend-dev_2.0.0_amd64.deb \
        ./pkg-deb/knp-python3-framework_2.0.0_amd64.deb \
        ./pkg-deb/knp-documentation_2.0.0_all.deb \
        ./knp_ann2snn.deb \
        ./knp-altai-backend_1.0.2_amd64.deb \
        ./knp-altai-backend-dev_1.0.2_amd64.deb

    log "Проверка: загружаю эмулятор AltAI-2..."
    # Python-модули deb-пакетов лежат в /usr/lib/python3 — добавляем в путь
    PYTHONPATH="/usr/lib/python3:${PYTHONPATH:-}" python3 - <<'EOF'
from knp.base_framework import BackendLoader
b = BackendLoader().load('/usr/lib/libknp-altai-backend.so')
print('Эмулятор AltAI-2 загружен, backend uid:', b.uid)
EOF

    log "Готово! Пакеты в системе: dpkg -l | grep knp"
    warn "Если python3 -c 'import knp' падает — запускайте с PYTHONPATH=/usr/lib/python3"
    warn "или используйте Docker-режим: ./deploy_knp.sh"
}

###############################################################################
case "${1:-docker}" in
    --native|native) deploy_native ;;
    --docker|docker|"") deploy_docker ;;
    -h|--help) head -n 24 "$0"; exit 0 ;;
    *) die "Неизвестный аргумент: $1 (используйте --docker или --native)" ;;
esac
