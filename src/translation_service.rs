use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    Base,
}

// TODO: Maybe add support for the conditional prompt.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long, default_value = "tokenizer-marian-base-de.json")]
    tokenizer: Option<String>,

    #[arg(long, default_value = "tokenizer-marian-base-en.json")]
    tokenizer_dec: Option<String>,

    /// Choose the variant of the model to run.
    #[arg(long, default_value = "base")]
    which: Which,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result};
use translation::{marian, TokenOutputStream};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
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
    tokenizer_dec: TokenOutputStream,
    logits_processor: candle_transformers::generation::LogitsProcessor,
}

impl Translator {
    pub fn init(args: Args) -> anyhow::Result<Self> {
        use hf_hub::api::sync::Api;

        let config = match args.which {
            Which::Base => marian::Config::opus_mt_de_en(),
        };
        let tokenizer = {
            let tokenizer = match args.tokenizer {
                Some(tokenizer) => std::path::PathBuf::from(tokenizer),
                None => {
                    let name = match args.which {
                        Which::Base => "tokenizer-marian-base-de.json",
                    };
                    Api::new()?
                        .model("lmz/candle-marian".to_string())
                        .get(name)?
                }
            };
            Tokenizer::from_file(&tokenizer).map_err(E::msg)?
        };

        let tokenizer_dec = {
            let tokenizer = match args.tokenizer_dec {
                Some(tokenizer) => std::path::PathBuf::from(tokenizer),
                None => {
                    let name = match args.which {
                        Which::Base => "tokenizer-marian-base-en.json",
                    };
                    Api::new()?
                        .model("lmz/candle-marian".to_string())
                        .get(name)?
                }
            };
            Tokenizer::from_file(&tokenizer).map_err(E::msg)?
        };
        let tokenizer_dec = TokenOutputStream::new(tokenizer_dec);

        let device = device(args.cpu)?;
        let vb = {
            let model = match args.model {
                Some(model) => std::path::PathBuf::from(model),
                None => match args.which {
                    Which::Base => Api::new()?
                        .repo(hf_hub::Repo::with_revision(
                            "Helsinki-NLP/opus-mt-de-en".to_string(),
                            hf_hub::RepoType::Model,
                            "refs/pr/4".to_string(),
                        ))
                        .get("model.safetensors")?,
                },
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

        let mut out: Vec<String> = vec![];
        for index in 0..1000 {
            let context_size = if index >= 1 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], &self.device)?.unsqueeze(0)?;
            let logits = self.model.decode(&input_ids, &encoder_xs, start_pos)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = self.logits_processor.sample(&logits)?;
            token_ids.push(token);
            if let Some(t) = self.tokenizer_dec.next_token(token)? {
                out.push(t);
            }
            if token == self.config.eos_token_id || token == self.config.forced_eos_token_id {
                break;
            }
        }
        self.model.reset_kv_cache();
        self.tokenizer_dec.clear();
        Ok(out.join(""))
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
