//! `Parallelism::Pool(n)` is exactly `n` threads, named, built on first use,
//! and nothing of the machine-sized global pool.

use std::num::NonZeroUsize;

use image::{DynamicImage, Rgb, RgbImage};
use llm_multimodal::{
    configure_parallelism,
    vision::{execution::POOL_THREAD_NAME_PREFIX, processors::Qwen2VLProcessor},
    Parallelism, PreProcessorConfig, VisionPreProcessor,
};

fn pool_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .map(|tasks| {
            tasks
                .flatten()
                .filter_map(|task| std::fs::read_to_string(task.path().join("comm")).ok())
                .filter(|name| name.trim().starts_with(POOL_THREAD_NAME_PREFIX))
                .count()
        })
        .unwrap_or(0)
}

#[test]
fn pool_mode_builds_exactly_the_named_threads_on_first_use() {
    let three = NonZeroUsize::new(3).expect("three");
    assert_eq!(configure_parallelism(Parallelism::Pool(three)), Ok(()));
    assert_eq!(pool_threads(), 0, "the pool is built lazily");
    let processor = Qwen2VLProcessor::new();
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1792, 1344, Rgb([90, 160, 30])));
    processor
        .preprocess(&[image], &PreProcessorConfig::default())
        .expect("preprocess");
    assert_eq!(pool_threads(), 3);
    // A later, different choice is refused and reports what is in force.
    assert_eq!(
        configure_parallelism(Parallelism::Inline),
        Err(Parallelism::Pool(three))
    );
}
