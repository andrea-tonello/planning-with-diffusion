# Installation and usage
## Built with python 3.11

### With `pip`
- Clone the repo
- Create a virtual env inside: `python3.11 -m venv .venv`
- Activate it with: `source .venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- ***Usage***. *Append `-h` to see all options, e.g. `python generate_data.py -h`* or check code
  - (Optional) Generate a desired dataset with `python generate_data.py` 
  - (Optional) Train with `python train.py`
  - Evaluate model/s with `python evaluate.py`

### With `uv`
- Clone the repo
- Sync environment inside: `uv sync`
- ***Usage***. *Append `-h` to see all options, e.g. `uv run generate_data.py -h`* or check code
  - (Optional) Generate a desired dataset with `uv run generate_data.py` 
  - (Optional) Train with `uv run train.py`
  - Evaluate model/s with `uv run evaluate.py`

*Model weights are already saved in the repo. However, it is possibile to re-train the models if you need it, using the main.ipynb notebook.*