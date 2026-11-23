//! Model-specific vision processors.
//!
//! This module contains implementations of `VisionPreProcessor` for various
//! vision-language model families.
//!
//! # Supported Models
//!
//! - **DeepSeek-V4.1** (`deepseek_v41`): contain-fit resize with gray padding, 14px patches and a 3x3 aligner downsample
//! - **Gemma 4** (`gemma4`): Aspect-preserving block-aligned resize, 16px patches padded to the soft-token budget, with video as frames
//! - **GLM-5.3-Flash** (`glm53_flash`): Aspect-preserving aligned-canvas image/video preprocessing
//! - **LLaVA 1.5** (`llava`): CLIP-based preprocessing with configurable aspect ratio
//! - **LLaVA-NeXT** (`llava`): Multi-crop anyres processing
//! - **Qwen2-VL** (`qwen2_vl`): Dynamic resolution with smart resizing
//! - **Qwen2.5-VL** (`qwen2_vl`): Same processor as Qwen2-VL (identical preprocessing)
//! - **Qwen3-VL** (`qwen3_vl`): Similar to Qwen2-VL but with patch_size=16 and [0.5,0.5,0.5] normalization
//! - **Qwen3-Omni** (`qwen3_omni_vision`): Qwen3 vision preprocessing with Omni video limits and timing metadata
//! - **Inkling** (`inkling`): Optional aspect-preserving resize with CLIP normalization and padded patch columns
//! - **Inkling2** (`inkling2`): Native image grids, checkerboard alpha and BF16 patch inputs
//! - **Kimi-K2.5** (`kimi_k25`): MoonViT resize and zero-padding to patch alignment
//! - **Kimi-K3** (`kimi_k3`): Same MoonViT stack with a 4x patch budget and
//!   chessboard alpha compositing
//! - **Phi3-Vision** (`phi3_vision`): Dynamic HD transform with 336x336 tiles
//! - **LLaMA 4 Vision** (`llama4_vision`): Tile-based processing with 336x336 tiles and global tile
//! - **Pixtral/Mistral3** (`pixtral`): CLIP-based preprocessing with dynamic resolution
//! - **Nemotron VL / Omni** (`nemotron_omni`): C-RADIO tiled or dynamic-resolution images and video frames, geometry
//!   read from the checkpoint's configs, as packed patch rows (Nemotron-H Omni) or in the reference tensor layout
//! - **Step3-VL** (`step3_vl`): a 728px global view and 504px sliding-window crops with CLIP normalization

pub mod deepseek_v41;
pub(crate) mod gemma4;
pub(crate) mod glm53_flash;
pub mod inkling;
pub(crate) mod inkling2;
pub(crate) mod kimi_k25;
pub(crate) mod kimi_k3;
pub(crate) mod llama4_vision;
pub(crate) mod llava;
pub(crate) mod minimax_m3;
pub(crate) mod moonvit;
pub(crate) mod muse_spark;
pub mod nemotron_omni;
pub mod phi3_vision;
pub(crate) mod pixtral;
pub(crate) mod qwen2_vl;
pub(crate) mod qwen3_omni_vision;
pub(crate) mod qwen3_vl;
pub mod qwen_vl_base;
pub mod step3_vl;

pub use deepseek_v41::DeepseekV41Processor;
pub use gemma4::Gemma4Processor;
pub use glm53_flash::Glm53FlashProcessor;
pub use inkling::InklingImageProcessor;
pub use inkling2::Inkling2ImageProcessor;
pub use kimi_k25::KimiK25Processor;
pub use kimi_k3::KimiK3Processor;
pub use llama4_vision::Llama4VisionProcessor;
pub use llava::{ImageAspectRatio, LlavaNextProcessor, LlavaProcessor};
pub use minimax_m3::MiniMaxM3VisionProcessor;
pub use muse_spark::MuseSparkVisionProcessor;
pub use nemotron_omni::NemotronOmniProcessor;
pub use phi3_vision::Phi3VisionProcessor;
pub use pixtral::PixtralProcessor;
pub use qwen2_vl::Qwen2VLProcessor;
pub use qwen3_omni_vision::Qwen3OmniVisionProcessor;
pub use qwen3_vl::Qwen3VLProcessor;
pub use step3_vl::Step3VLProcessor;
