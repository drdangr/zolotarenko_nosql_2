@echo off
REM Запуск усього пайплайну ДЗ-2 (Windows). Аргументи пробрасуються у run_pipeline.py.
REM Приклади:  run.bat   |   run.bat --with-map   |   run.bat --from 4
python "%~dp0run_pipeline.py" %*
