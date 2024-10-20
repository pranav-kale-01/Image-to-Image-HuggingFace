install: 
	pip install --ugprade pip && pip install -r requirements.txt 

run: 
	black *.py && .\venv\Scripts\activate && streamlit run main.py

package:
	pip freeze > requirements.txt

all: 
	format install run