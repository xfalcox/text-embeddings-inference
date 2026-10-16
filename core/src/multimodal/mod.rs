/// Preprocessing of image and video inputs for multimodal embedding models
mod resample;

use crate::TextEmbeddingsError;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use image::{DynamicImage, ImageDecoder, ImageReader};
use std::io::Cursor;
use text_embeddings_backend::BatchImage;

/// Preprocessing parameters shared by images and video frames
#[derive(Debug, Clone, PartialEq)]
pub struct VisionProcessorConfig {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    /// Maximum number of soft tokens per image (or video frame), which bounds its resolution
    pub max_soft_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoProcessorConfig {
    pub frame: VisionProcessorConfig,
    /// Maximum number of frames, uniformly subsampled from the provided ones if exceeded
    pub max_frames: usize,
}

/// Placeholder tokens (as strings) wrapping and standing in for the soft tokens
#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalTokens {
    pub begin_image: String,
    pub end_image: String,
    pub image: String,
    pub video: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalConfig {
    pub image: VisionProcessorConfig,
    pub video: Option<VideoProcessorConfig>,
    pub tokens: MultimodalTokens,
}

/// A part of a multimodal input, with the images and video frames as base64-encoded files e.g.
/// PNG or JPEG, optionally as data URLs
#[derive(Debug, Clone)]
pub enum ContentPart {
    Text(String),
    Image(String),
    Video(Vec<String>),
}

/// Decodes a base64-encoded image, optionally given as a data URL
fn decode_base64(image: &str) -> Result<Vec<u8>, TextEmbeddingsError> {
    let image = image.trim();
    if image.starts_with("http://") || image.starts_with("https://") {
        return Err(TextEmbeddingsError::Validation(
            "Image URLs are not supported, images must be base64-encoded".to_string(),
        ));
    }
    let data = match image.strip_prefix("data:") {
        Some(data_url) => match data_url.split_once(";base64,") {
            Some((_, data)) => data,
            None => {
                return Err(TextEmbeddingsError::Validation(
                    "Image data URLs must be base64-encoded".to_string(),
                ))
            }
        },
        None => image,
    };
    STANDARD
        .decode(data)
        .map_err(|err| TextEmbeddingsError::Validation(format!("Image is not valid base64: {err}")))
}

impl VisionProcessorConfig {
    /// Computes the size to resize an image to, preserving its aspect ratio, so that it produces
    /// at most `max_soft_tokens` soft tokens, with both sides multiple of
    /// `patch_size * pooling_kernel_size`
    pub fn target_size(&self, height: usize, width: usize) -> Result<(usize, usize), String> {
        let max_patches = self.max_soft_tokens * self.pooling_kernel_size.pow(2);
        let target_pixels = (max_patches * self.patch_size.pow(2)) as f64;
        let factor = (target_pixels / (height * width) as f64).sqrt();
        let side = self.pooling_kernel_size * self.patch_size;
        let max_side_length = (max_patches / self.pooling_kernel_size.pow(2)) * side;

        let target_height = ((factor * height as f64) / side as f64).floor() as usize * side;
        let target_width = ((factor * width as f64) / side as f64).floor() as usize * side;

        match (target_height, target_width) {
            (0, 0) => Err(format!(
                "Image of {width}x{height} cannot be resized to a valid size"
            )),
            (0, _) => Ok((side, (width / height * side).min(max_side_length))),
            (_, 0) => Ok(((height / width * side).min(max_side_length), side)),
            size => Ok(size),
        }
    }

    pub fn num_soft_tokens(&self, image: &BatchImage) -> usize {
        let side = self.pooling_kernel_size * self.patch_size;
        (image.height / side) * (image.width / side)
    }

    /// Decodes an image (applying its EXIF orientation), converts it to RGB and resizes it with
    /// bicubic interpolation, as Transformers does
    pub fn preprocess(&self, image: &str) -> Result<BatchImage, TextEmbeddingsError> {
        let image = decode_image(&decode_base64(image)?)
            .map_err(|err| TextEmbeddingsError::Validation(format!("Invalid image: {err}")))?;

        // NOTE: Any alpha channel is dropped rather than composited, as with `Image.convert("RGB")`
        let image = image.to_rgb8();
        let (width, height) = (image.width() as usize, image.height() as usize);

        let (target_height, target_width) = self
            .target_size(height, width)
            .map_err(TextEmbeddingsError::Validation)?;

        let pixels = image.into_raw();
        let pixels = if (target_height, target_width) == (height, width) {
            pixels
        } else {
            resample::resize_bicubic_rgb(&pixels, height, width, target_height, target_width)
        };

        Ok(BatchImage {
            height: target_height,
            width: target_width,
            pixels,
        })
    }
}

fn decode_image(bytes: &[u8]) -> image::ImageResult<DynamicImage> {
    let mut decoder = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}

/// Uniformly subsamples `max_frames` indices out of `num_frames`, matching
/// `np.linspace(0, num_frames - 1, max_frames, dtype=int)`
pub fn subsample_frames(num_frames: usize, max_frames: usize) -> Vec<usize> {
    if num_frames <= max_frames {
        return (0..num_frames).collect();
    }
    if max_frames == 1 {
        return vec![0];
    }
    let stop = (num_frames - 1) as f64;
    let step = stop / (max_frames - 1) as f64;
    (0..max_frames)
        .map(|i| {
            if i == max_frames - 1 {
                num_frames - 1
            } else {
                (i as f64 * step) as usize
            }
        })
        .collect()
}

impl MultimodalConfig {
    /// Preprocesses the images and video frames, and renders the inputs into a text where each
    /// image (or video frame) is replaced by its placeholder tokens
    pub fn prepare(
        &self,
        parts: Vec<ContentPart>,
        text: &mut String,
    ) -> Result<Vec<BatchImage>, TextEmbeddingsError> {
        let mut images = Vec::new();
        let tokens = &self.tokens;

        let push_placeholders = |text: &mut String, token: &str, num_soft_tokens: usize| {
            text.push_str(&tokens.begin_image);
            for _ in 0..num_soft_tokens {
                text.push_str(token);
            }
            text.push_str(&tokens.end_image);
        };

        for part in parts {
            match part {
                ContentPart::Text(s) => text.push_str(&s),
                ContentPart::Image(image) => {
                    let image = self.image.preprocess(&image)?;
                    push_placeholders(text, &tokens.image, self.image.num_soft_tokens(&image));
                    images.push(image);
                }
                ContentPart::Video(frames) => {
                    let Some(video) = &self.video else {
                        return Err(TextEmbeddingsError::Validation(
                            "Video inputs are not supported by this model".to_string(),
                        ));
                    };
                    if frames.is_empty() {
                        return Err(TextEmbeddingsError::Validation(
                            "Video inputs must contain at least one frame".to_string(),
                        ));
                    }
                    for index in subsample_frames(frames.len(), video.max_frames) {
                        let frame = video.frame.preprocess(&frames[index])?;
                        push_placeholders(text, &tokens.video, video.frame.num_soft_tokens(&frame));
                        images.push(frame);
                    }
                }
            }
        }

        Ok(images)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(max_soft_tokens: usize) -> VisionProcessorConfig {
        VisionProcessorConfig {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens,
        }
    }

    #[test]
    fn target_size() {
        // NOTE: Expected values from `transformers` i.e. `get_resize_output_image_size`
        assert_eq!(config(280).target_size(240, 320), Ok((672, 912)));
        assert_eq!(config(280).target_size(200, 1000), Ok((336, 1776)));
        assert_eq!(config(280).target_size(1715, 2400), Ok((672, 912)));
        assert_eq!(config(140).target_size(240, 320), Ok((480, 624)));
        assert_eq!(config(280).target_size(1, 10000), Ok((48, 13440)));
    }

    #[test]
    fn subsample() {
        assert_eq!(subsample_frames(3, 32), vec![0, 1, 2]);
        assert_eq!(
            subsample_frames(40, 32),
            vec![
                0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27,
                28, 30, 31, 32, 33, 35, 36, 37, 39
            ]
        );
    }
}
