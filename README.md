# SimpleQandASystem
Simple Q&A system. Knowledge will be gathered from local files. Will provide a simple UI for asking questions and reading answers. A LLM powered backend to read the query, search the knowledge base, then answer. A knowledge base represented by a Vector DB

# Notes:
Docling services might require running as administrator on windows. Error 22 is the clue.


# Pre-requesites
* Python 3.13 - https://www.python.org/downloads/release/python-3130/
* Docker Desktop - https://docs.docker.com/desktop/
* Ollama - https://ollama.com

# Setup:
* Run scripts/build_venv with the bare python
* Setup the project interpreter to be the python from the venv folder
* Run scripts/setup_qdrant

# Indexing
* Create a folder "domaindata" in the root (default name) or choose a path from ROOT and update configs/indexer_config.json -> data_directory
* Place all your pdf's inside the folder.
* Run indexer/main.py
* NB!: Indexing is slow so working with a large number of PDFs will be very slow.



# Backend
* In a terminal run an ollama instance, default model is llama2, so run "ollama run llama2". This will download a lightweight model that will be used by the Backend.
* NB! Launch an ollama model that is the same as the configs/backend_config.json
* Run backend/main.py
* NB!: Make sure port 8001 is free. Can change port via configs/backend_config.json

# UI
* Run ui/main.py
* NB!: Make sure port 8000 is free. Can change port via configs/ui_config.json and in configs/backend_config.json