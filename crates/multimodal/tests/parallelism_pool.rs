//! Pool mode preprocesses and fixes the process-wide configuration. On Linux,
//! procfs also verifies that exactly `n` named threads are built on first use.

use std::num::NonZeroUsize;

use image::{DynamicImage, Rgb, RgbImage};
use llm_multimodal::{
    configure_parallelism, vision::processors::Qwen2VLProcessor, Parallelism, PreProcessorConfig,
    VisionPreProcessor,
};

#[cfg(target_os = "linux")]
fn pool_threads() -> std::io::Result<usize> {
    use llm_multimodal::vision::execution::POOL_THREAD_NAME_PREFIX;
    std::fs::read_dir("/proc/self/task")?.try_fold(0, |count, task| {
        let name = std::fs::read_to_string(task?.path().join("comm"))?;
        Ok(count + usize::from(name.trim().starts_with(POOL_THREAD_NAME_PREFIX)))
    })
}

#[test]
fn pool_mode_preprocesses_and_rejects_reconfiguration() {
    let three = NonZeroUsize::new(3).expect("three");
    assert_eq!(configure_parallelism(Parallelism::Pool(three)), Ok(()));
    #[cfg(target_os = "linux")]
    assert_eq!(
        pool_threads().expect("read Linux pool threads"),
        0,
        "the pool is built lazily"
    );
    let processor = Qwen2VLProcessor::new();
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1792, 1344, Rgb([90, 160, 30])));
    let out = processor
        .preprocess(&[image], &PreProcessorConfig::default())
        .expect("preprocess");
    assert!(out.total_feature_tokens() > 0);
    #[cfg(target_os = "linux")]
    assert_eq!(pool_threads().expect("read Linux pool threads"), 3);
    // A later, different choice is refused and reports what is in force.
    assert_eq!(
        configure_parallelism(Parallelism::Inline),
        Err(Parallelism::Pool(three))
    );
}
