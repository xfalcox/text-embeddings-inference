use anyhow::Result;

use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

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
