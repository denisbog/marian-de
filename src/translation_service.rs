use anyhow::Error as E;
use candle_transformers::models::marian;
use clap::Parser;

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long, default_value = "tokenizer-marian-base-de.json")]
    tokenizer: String,

    #[arg(long, default_value = "tokenizer-marian-base-en.json")]
    tokenizer_dec: String,
}
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result};
use translation::opus_mt_de_en;

pub fn device() -> Result<Device> {
    if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

struct Translator {
    device: Device,
    config: marian::Config,
    model: marian::MTModel,
    tokenizer: Tokenizer,
    tokenizer_dec: Tokenizer,
    logits_processor: candle_transformers::generation::LogitsProcessor,
}

impl Translator {
    pub fn init(args: Args) -> anyhow::Result<Self> {
        use hf_hub::api::sync::Api;

        let config = opus_mt_de_en();

        let tokenizer =
            Tokenizer::from_file(&std::path::PathBuf::from(args.tokenizer)).map_err(E::msg)?;

        let tokenizer_dec =
            Tokenizer::from_file(&std::path::PathBuf::from(args.tokenizer_dec)).map_err(E::msg)?;

        let device = device()?;
        let vb = {
            let model = match args.model {
                Some(model) => std::path::PathBuf::from(model),
                None => Api::new()?
                    .repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-de-en".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/4".to_string(),
                    ))
                    .get("model.safetensors")?,
            };
            unsafe { VarBuilder::from_mmaped_safetensors(&[&model], DType::F32, &device)? }
        };
        let model = marian::MTModel::new(&config, vb)?;

        let logits_processor =
            candle_transformers::generation::LogitsProcessor::new(1337, None, None);

        Ok(Translator {
            device,
            config,
            model,
            tokenizer,
            tokenizer_dec,
            logits_processor,
        })
    }

    pub fn translate(&mut self, text: String) -> anyhow::Result<String> {
        let encoder_xs = {
            let mut tokens = self
                .tokenizer
                .encode(text, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            tokens.push(self.config.eos_token_id);
            let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            self.model.encoder().forward(&tokens, 0)?
        };
        let mut token_ids = vec![self.config.decoder_start_token_id];

        for index in 0..1000 {
            let context_size = if index >= 1 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], &self.device)?.unsqueeze(0)?;
            let logits = self.model.decode(&input_ids, &encoder_xs, start_pos)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = self.logits_processor.sample(&logits)?;
            token_ids.push(token);
            if token == self.config.eos_token_id || token == self.config.forced_eos_token_id {
                break;
            }
        }
        self.model.reset_kv_cache();
        let temp: String = self
            .tokenizer_dec
            .decode(&token_ids[1..token_ids.len() - 1], true)
            .unwrap();
        Ok(temp)
    }
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut translator = Translator::init(args)?;
    let out = translator.translate("viele danke".to_string());
    println!("{}", out?);
    let out = translator.translate("viele danke".to_string());
    println!("{}", out?);
    Ok(())
}
