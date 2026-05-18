@echo off
cd /d "%~dp0"
python -m streamlit run src/ui/dashboard.py --server.port 8501 --server.headless false
