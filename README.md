[![CI](https://github.com/lenamueller/StochastischeSpeicherbemessung/actions/workflows/python-app.yml/badge.svg)](https://github.com/lenamueller/StochastischeSpeicherbemessung/actions/workflows/python-app.yml)
[![repo status - active](https://img.shields.io/badge/repo_status-active-green)](https://)
[![python - 3.11.5](https://img.shields.io/badge/python-3.11.5-ffe05c?logo=python&logoColor=4685b7)](https://)
[![field of application - hydrology](https://img.shields.io/badge/field_of_application-hydrology-00aaff)](https://)

# Content

#### code/
- utils/*.py:    helper functions
- main.py:       main script
- setup.py:      setup script

#### data/
- Daten_*_raw.txt:  raw data
- Daten_*_detrended.txt:  detrended data
- Daten_*_seasonal.txt:  seasonal data
- Daten_*_residual.txt:  residual data

####  images/
images generated with main.py
#### reports/
reports generated with main.py

# Usage
#### 1. Clone the repository
```bash
git clone git@github.com:lenamueller/StochastischeSpeicherbemessung.git
```
#### 2. Install the requirements
```bash
pip install -r requirements.txt
```
#### 3. Run the code (e.g. for Klingenthal)
```bash 
python code/main.py
```
