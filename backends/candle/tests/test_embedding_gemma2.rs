use anyhow::Result;

use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, BatchImage, ModelType, Pool};

mod common;
use crate::common::{
    batch, cosine_matcher, download_artifacts, load_tokenizer, sort_embeddings, SnapshotEmbeddings,
};

#[test]
#[serial_test::serial]
fn test_embedding_gemma2() -> Result<()> {
    let (model_root, dense_paths) = download_artifacts("google/embeddinggemma-2", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        dense_paths,
    )?;

    // NOTE: The long input exceeds the sliding window, so that the bidirectional sliding window
    // attention mask is combined with the padding mask
    let long_input = (0..120)
        .map(|i| format!("Sentence number {i} talks about deep learning and embeddings."))
        .collect::<Vec<String>>()
        .join(" ");

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
            tokenizer.encode(long_input.as_str(), true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 1, 2, 3].to_vec(),
        vec![],
    );

    let matcher = cosine_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("embedding_gemma2_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("embedding_gemma2_single", embeddings_single, &matcher);

    let input_long = batch(
        vec![tokenizer.encode(long_input.as_str(), true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_long)?);
    let embeddings_long = SnapshotEmbeddings::from(pooled_embeddings);

    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_long[0]);
    assert_eq!(embeddings_batch[3], embeddings_single[0]);

    Ok(())
}

/// Deterministic high-frequency RGB image, so that any misplaced pixel shows up
fn synthetic_image(height: usize, width: usize) -> BatchImage {
    let mut pixels = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                pixels.push((((x * 3 + y * 5 + c * 70) % 256) ^ ((x * y) % 256)) as u8);
            }
        }
    }
    BatchImage {
        height,
        width,
        pixels,
    }
}

#[test]
#[serial_test::serial]
fn test_embedding_gemma2_image() -> Result<()> {
    let (model_root, dense_paths) = download_artifacts("google/embeddinggemma-2", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        dense_paths,
    )?;

    // NOTE: 672x912 is already a valid size i.e., no resize is needed, producing 14x19 soft tokens
    let image = synthetic_image(672, 912);
    let image_placeholder = format!("<|image>{}<image|>", "<|image|>".repeat(14 * 19));

    let image_encoding = tokenizer.encode(image_placeholder.as_str(), true).unwrap();
    assert_eq!(image_encoding.len(), 270);

    let input_batch = batch(
        vec![
            image_encoding.clone(),
            tokenizer
                .encode(
                    format!("A colorful pattern{image_placeholder}").as_str(),
                    true,
                )
                .unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );
    let input_batch = Batch {
        images: vec![image.clone(), image.clone()],
        ..input_batch
    };

    let matcher = cosine_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("embedding_gemma2_image_batch", embeddings_batch, &matcher);

    let input_single = Batch {
        images: vec![image],
        ..batch(vec![image_encoding], [0].to_vec(), vec![])
    };

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert!(backend.supports_images());
    drop(backend);

    // Without the vision encoder, only text inputs are supported
    let backend = CandleBackend::new_with_vision(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
        false,
    )?;
    assert!(!backend.supports_images());

    let input_image = Batch {
        images: vec![synthetic_image(672, 912)],
        ..batch(
            vec![tokenizer.encode(image_placeholder.as_str(), true).unwrap()],
            [0].to_vec(),
            vec![],
        )
    };
    assert!(backend.embed(input_image).is_err());

    Ok(())
}
