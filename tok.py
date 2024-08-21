from convert_slow_tokenizer import MarianConverter
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en", use_fast=False)
fast_tokenizer = MarianConverter(tokenizer, index=0).converted()
fast_tokenizer.save(f"tokenizer-marian-base-de.json")
fast_tokenizer = MarianConverter(tokenizer, index=1).converted()
fast_tokenizer.save(f"tokenizer-marian-base-en.json")
