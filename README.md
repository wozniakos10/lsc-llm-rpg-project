For install pre-commit hooks:
```
pre-commit install
```


For create uv env
```
uv sync
```

There was problem for install vosk by uv in macos, please use:

```
pip install vosk
```

Create .env file which contains:

```
MODELS_PATH=PATH
ASSETS_PATH=PATH
```

For run app:
```
python src/llm_rpg/interface/app.py
```