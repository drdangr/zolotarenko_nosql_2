#!/usr/bin/env bash
# Запуск усього пайплайну ДЗ-2 (macOS/Linux). Аргументи пробрасуються у run_pipeline.py.
# Приклади:  ./run.sh   |   ./run.sh --with-map   |   ./run.sh --from 4
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DIR/run_pipeline.py" "$@"
