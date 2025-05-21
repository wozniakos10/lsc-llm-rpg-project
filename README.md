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

For download model from hugginface:
```
huggingface-cli download speakleash/Bielik-4.5B-v3.0-Instruct-GGUF --local-dir models
```

Setup llama-cpp-python with cuda/metal:
```
https://github.com/abetlen/llama-cpp-python
```

For run app:
```
python src/llm_rpg/interface/app.py
```