<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Image and video inputs

Multimodal embedding models embed images and videos, optionally combined with text, into the same space as text.
TEI currently supports image and video inputs for [google/embeddinggemma-2](https://hf.co/google/embeddinggemma-2)
with the Candle backend, on both the `/embed` and the OpenAI-compatible `/v1/embeddings` routes.

```shell
model=google/embeddinggemma-2

docker run --gpus all -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-embeddings-inference:cuda-latest --model-id $model
```

On start-up, the logs will contain `Image and video inputs are supported`.

## Disabling image and video inputs

Image and video inputs are enabled by default for the models that support those. To serve text inputs only, set
`--disable-vision` (or `DISABLE_VISION=true`), so that the vision encoder is not loaded, which saves around 670MB
of memory for EmbeddingGemma 2, in which case the image and video inputs are rejected.

```shell
docker run --gpus all -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-embeddings-inference:cuda-latest --model-id $model --disable-vision
```

## Images

Images are sent base64-encoded, either as raw base64 or as a data URL, within an object with an `image` key. Any
format among PNG, JPEG, WebP, GIF and BMP is supported, and images are resized to the resolution expected by the
model, so there is no need to resize those beforehand.

```shell
curl 127.0.0.1:8080/embed \
    -X POST \
    -d "{\"inputs\": {\"image\": \"data:image/jpeg;base64,$(base64 -w0 cat.jpg)\"}}" \
    -H 'Content-Type: application/json'
```

Text and images can be embedded together into a single embedding by listing them, in order, under `content`:

```json
{
  "inputs": {
    "content": [
      {"text": "A photo of"},
      {"image": "data:image/jpeg;base64,/9j/4AAQ..."}
    ]
  }
}
```

Images, videos, text and any combination of those can be mixed within the same batch of `inputs`, e.g.
`"inputs": ["What is Deep Learning?", {"image": "..."}]`.

## Videos

Videos are sent as a list of base64-encoded frames under a `video` key, as TEI doesn't decode video files. The frames
should ideally be sampled at 1 frame per second, as done by Transformers. If more frames than supported by the model
are provided (32 for EmbeddingGemma 2), those are uniformly subsampled.

```json
{
  "inputs": {
    "video": [
      "data:image/jpeg;base64,/9j/4AAQ...",
      "data:image/jpeg;base64,/9j/4AAQ..."
    ]
  }
}
```

## Prompts

The prompts work the same way as for text inputs i.e., the `prompt_name` is prepended to the whole input, including
the images and videos, as done by Sentence Transformers. Note that for EmbeddingGemma 2, the task prompts are
recommended for text inputs only.

## Limitations

- Each image produces up to 280 tokens (and each video frame up to 140 tokens), which count towards the
  `--max-batch-tokens` and the maximum input length. Unlike text, inputs with images or videos are never truncated, so
  an error is returned if those exceed the maximum input length.
- Images must be sent base64-encoded, as fetching image URLs is not supported.
- The default `--payload-limit` is raised to 50MB for models supporting image and video inputs.
- Image and video inputs are not supported via gRPC, nor with the Python and ONNX backends.
- JPEG images are decoded with a different decoder than Pillow, which may lead to slightly different embeddings than
  Sentence Transformers (with a cosine similarity typically above 0.9997), whereas PNG images match closely.
