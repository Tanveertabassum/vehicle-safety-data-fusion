#!/usr/bin/env bash
# Rebuild everything from raw data (about 5-6 minutes on one CPU core with 3 GB RAM, after the ~130 MB download).
set -euo pipefail
cd "$(dirname "$0")/src"
python 00_download.py
python 01_ingest.py
python 02_clean_integrate.py
python 03_build_vehicle_table.py
python 04_recall_model.py
python 05_text_classifier.py
python 06_write_dq_report.py
python 07_export_app_assets.py
echo "Done. Start the app with: streamlit run app/streamlit_app.py"
