//! The field-layout tables more than one registry entry declares.
//!
//! A tower that patchifies its images ships the pixels of a batch flat over
//! the patches, with each item's grid and patch count one row per item
//! beside them; a tower that keeps whole images ships one row per image with
//! its sizes beside it. Each table the entries share is written once here, so
//! a tensor added to it reaches every entry that declares the table.

use std::collections::HashMap;

use crate::types::{EncoderFieldLayouts, FieldLayout, Modality};

/// The per-modality layouts of a tower patchified over `*_grid_thw` grids: a
/// video batch is sliced by `patches_per_video` and carries its grids and
/// frame spacing, an image batch by `patches_per_image` with its grids. One
/// map per modality: a video batch must not advertise the image sizes key, or
/// the other way round, or a request carrying both is sliced by a tensor the
/// batch does not have.
pub(super) fn grid_patch_layouts_for(modality: Modality) -> EncoderFieldLayouts {
    match modality {
        Modality::Video => EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_video"),
            HashMap::from([
                ("video_grid_thw".to_string(), FieldLayout::Batched),
                ("patches_per_video".to_string(), FieldLayout::Batched),
                ("video_second_per_grid".to_string(), FieldLayout::Batched),
            ]),
        ),
        _ => EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_image"),
            HashMap::from([
                ("image_grid_thw".to_string(), FieldLayout::Batched),
                ("patches_per_image".to_string(), FieldLayout::Batched),
            ]),
        ),
    }
}

/// The same tower's legacy field map: the image pixels flat over
/// `patches_per_image`, the image and video grids, counts and frame spacing
/// one row per item.
pub(super) fn grid_patch_fields() -> HashMap<String, FieldLayout> {
    HashMap::from([
        (
            "pixel_values".to_string(),
            FieldLayout::flat("patches_per_image"),
        ),
        ("image_grid_thw".to_string(), FieldLayout::Batched),
        ("patches_per_image".to_string(), FieldLayout::Batched),
        ("video_grid_thw".to_string(), FieldLayout::Batched),
        ("patches_per_video".to_string(), FieldLayout::Batched),
        ("video_second_per_grid".to_string(), FieldLayout::Batched),
    ])
}

/// The field map of a tower patchified over one `grid_thws` tensor: the
/// pixels flat over `patches_per_image`, the grids and counts one row per
/// image.
pub(super) fn grid_thws_patch_fields() -> HashMap<String, FieldLayout> {
    HashMap::from([
        (
            "pixel_values".to_string(),
            FieldLayout::flat("patches_per_image"),
        ),
        ("grid_thws".to_string(), FieldLayout::Batched),
        ("patches_per_image".to_string(), FieldLayout::Batched),
    ])
}

/// The field map of a tower that keeps whole images: one row per image for
/// the pixels and for `image_sizes`.
pub(super) fn batched_with_image_sizes() -> HashMap<String, FieldLayout> {
    HashMap::from([
        ("pixel_values".to_string(), FieldLayout::Batched),
        ("image_sizes".to_string(), FieldLayout::Batched),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_keys(fields: &HashMap<String, FieldLayout>) -> Vec<&str> {
        let mut keys: Vec<&str> = fields.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn a_grid_patch_tower_declares_each_modality_with_its_own_tensors() {
        let video = grid_patch_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("patches_per_video"));
        assert_eq!(
            sorted_keys(&video.model_specific),
            [
                "patches_per_video",
                "video_grid_thw",
                "video_second_per_grid"
            ]
        );
        let image = grid_patch_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::flat("patches_per_image"));
        assert_eq!(
            sorted_keys(&image.model_specific),
            ["image_grid_thw", "patches_per_image"]
        );
        for layouts in [&video, &image] {
            assert!(layouts
                .model_specific
                .values()
                .all(|layout| *layout == FieldLayout::Batched));
        }
    }

    #[test]
    fn the_grid_patch_field_map_lists_the_image_and_video_tensors() {
        let fields = grid_patch_fields();
        assert_eq!(
            fields.get("pixel_values"),
            Some(&FieldLayout::flat("patches_per_image"))
        );
        assert_eq!(
            sorted_keys(&fields),
            [
                "image_grid_thw",
                "patches_per_image",
                "patches_per_video",
                "pixel_values",
                "video_grid_thw",
                "video_second_per_grid"
            ]
        );
        assert!(fields
            .iter()
            .filter(|(key, _)| *key != "pixel_values")
            .all(|(_, layout)| *layout == FieldLayout::Batched));
    }

    #[test]
    fn a_grid_thws_tower_slices_its_pixels_by_the_patch_count() {
        let fields = grid_thws_patch_fields();
        assert_eq!(
            fields.get("pixel_values"),
            Some(&FieldLayout::flat("patches_per_image"))
        );
        assert_eq!(fields.get("grid_thws"), Some(&FieldLayout::Batched));
        assert_eq!(fields.get("patches_per_image"), Some(&FieldLayout::Batched));
        assert_eq!(fields.len(), 3);
    }

    #[test]
    fn a_whole_image_tower_batches_its_pixels_and_sizes() {
        let fields = batched_with_image_sizes();
        assert_eq!(fields.get("pixel_values"), Some(&FieldLayout::Batched));
        assert_eq!(fields.get("image_sizes"), Some(&FieldLayout::Batched));
        assert_eq!(fields.len(), 2);
    }
}
