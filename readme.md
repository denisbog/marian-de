
### generate tokenizer files
```bash
python tok.py
```

### get a translation
```bash
time PATH=$PATH:/usr/local/cuda-12.5/bin/ cargo run --release --features cuda --  --tokenizer tokenizer-marian-base-de.json --tokenizer-dec tokenizer-marian-base-en.json  --text "viele danke"
```

### links to check
```bash
https://huggingface.co/Helsinki-NLP/opus-mt-de-en
https://huggingface.co/lmz/candle-marian/tree/main
```
