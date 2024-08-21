
### generate tokenizer files
```bash
python tok.py
```

### get a translation
```bash
time PATH=$PATH:/usr/local/cuda-12.5/bin/ cargo run --release --features cuda --bin translate --  --tokenizer tokenizer-marian-base-de.json --tokenizer-dec tokenizer-marian-base-en.json  --text "viele danke"
```

### translation service

```bash
time PATH=$PATH:/usr/local/cuda-12.5/bin/ cargo run --release --features cuda --bin translation-service --  --tokenizer tokenizer-marian-base-de.json --tokenizer-dec tokenizer-marian-base-en.json
```


### links to check
```bash
https://huggingface.co/Helsinki-NLP/opus-mt-de-en
https://huggingface.co/lmz/candle-marian/tree/main
```
