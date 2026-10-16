use crate::layers::{apply_rotary, get_inv_freqs, HiddenAct, Linear, RopeParameters};
use crate::models::Model;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, BatchImage, ModelType, Pool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingGemma2LayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2PerLayerConfig {
    pub head_dim: usize,
    pub num_key_value_heads: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2TextConfig {
    #[serde(default)]
    pub attention_bias: bool,
    pub pad_token_id: Option<u32>,
    pub head_dim: usize,
    pub hidden_activation: HiddenAct,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: Option<HashMap<EmbeddingGemma2LayerType, RopeParameters>>,
    pub sliding_window: usize,
    pub layer_types: Option<Vec<EmbeddingGemma2LayerType>>,
    /// Overrides of `head_dim` and `num_key_value_heads` keyed by the (zero-padded) layer index
    pub per_layer_config: Option<HashMap<String, EmbeddingGemma2PerLayerConfig>>,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2Config {
    pub text_config: EmbeddingGemma2TextConfig,
    pub vision_config: Option<EmbeddingGemma2VisionConfig>,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub hidden_activation: HiddenAct,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeParameters,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub standardize: bool,
}

impl EmbeddingGemma2TextConfig {
    // NOTE: Defaults match `EmbeddingGemma2TextConfig.__post_init__` in Transformers
    const SLIDING_WINDOW_PATTERN: usize = 6;
    const GLOBAL_HEAD_DIM: usize = 512;
    const NUM_GLOBAL_KEY_VALUE_HEADS: usize = 1;

    fn layer_types(&self) -> Vec<EmbeddingGemma2LayerType> {
        let mut layer_types = self.layer_types.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|i| match (i + 1) % Self::SLIDING_WINDOW_PATTERN {
                    0 => EmbeddingGemma2LayerType::FullAttention,
                    _ => EmbeddingGemma2LayerType::SlidingAttention,
                })
                .collect()
        });
        // NOTE: Transformers forces the last layer to be `full_attention`
        if let Some(last) = layer_types.last_mut() {
            *last = EmbeddingGemma2LayerType::FullAttention;
        }
        layer_types
    }

    /// Returns `(head_dim, num_key_value_heads)` for the given layer
    fn layer_config(
        &self,
        layer_idx: usize,
        layer_type: EmbeddingGemma2LayerType,
    ) -> Result<(usize, usize)> {
        match &self.per_layer_config {
            Some(per_layer_config) => {
                let mut layer_config = None;
                for (key, value) in per_layer_config {
                    let idx = key.parse::<usize>().map_err(|_| {
                        candle::Error::Msg(format!("Invalid `per_layer_config` key `{key}`"))
                    })?;
                    if idx == layer_idx {
                        layer_config = Some((value.head_dim, value.num_key_value_heads));
                    }
                }
                Ok(layer_config.unwrap_or((self.head_dim, self.num_key_value_heads)))
            }
            None => Ok(match layer_type {
                EmbeddingGemma2LayerType::FullAttention => {
                    (Self::GLOBAL_HEAD_DIM, Self::NUM_GLOBAL_KEY_VALUE_HEADS)
                }
                EmbeddingGemma2LayerType::SlidingAttention => {
                    (self.head_dim, self.num_key_value_heads)
                }
            }),
        }
    }

    fn rope_theta(&self, layer_type: EmbeddingGemma2LayerType) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rope_parameters| rope_parameters.get(&layer_type))
            .map(|rope_parameters| rope_parameters.rope_theta)
            .unwrap_or(match layer_type {
                EmbeddingGemma2LayerType::FullAttention => 1_000_000.0,
                EmbeddingGemma2LayerType::SlidingAttention => 10_000.0,
            })
    }
}

/// RMSNorm as in Gemma 4 i.e., scaled by `weight` (rather than `1 + weight` as in Gemma 3), and
/// optionally without any scale at all
#[derive(Debug)]
struct EmbeddingGemma2RMSNorm {
    weight: Option<Tensor>,
    epsilon: f64,

    span: tracing::Span,
}

impl EmbeddingGemma2RMSNorm {
    pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: Some(vb.get(hidden_size, "weight")?),
            epsilon: epsilon as f64,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
        })
    }

    pub fn without_scale(epsilon: f32) -> Self {
        Self {
            weight: None,
            epsilon: epsilon as f64,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let mean_squared = hidden_states.sqr()?.mean_keepdim(D::Minus1)?;
        let hidden_states = hidden_states.broadcast_div(&(mean_squared + self.epsilon)?.sqrt()?)?;
        let hidden_states = match &self.weight {
            Some(weight) => hidden_states.broadcast_mul(&weight.to_dtype(DType::F32)?)?,
            None => hidden_states,
        };
        hidden_states.to_dtype(dtype)
    }
}

struct EmbeddingGemma2Attention {
    qkv_proj: Linear,
    o_proj: Linear,

    q_norm: EmbeddingGemma2RMSNorm,
    k_norm: EmbeddingGemma2RMSNorm,
    v_norm: EmbeddingGemma2RMSNorm,

    attention_head_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2Attention {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2TextConfig,
        attention_head_size: usize,
        num_key_value_heads: usize,
    ) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let q_size = num_attention_heads * attention_head_size;
        let kv_size = num_key_value_heads * attention_head_size;

        let query_weight = vb.pp("q_proj").get((q_size, hidden_size), "weight")?;
        let key_weight = vb.pp("k_proj").get((kv_size, hidden_size), "weight")?;
        let value_weight = vb.pp("v_proj").get((kv_size, hidden_size), "weight")?;
        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;

        let qkv_bias = if config.attention_bias {
            let query_bias = vb.pp("q_proj").get(q_size, "bias")?;
            let key_bias = vb.pp("k_proj").get(kv_size, "bias")?;
            let value_bias = vb.pp("v_proj").get(kv_size, "bias")?;
            Some(Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?)
        } else {
            None
        };
        let qkv_proj = Linear::new(qkv_weight, qkv_bias, None);

        let output_weight = vb.pp("o_proj").get((hidden_size, q_size), "weight")?;
        let output_bias = if config.attention_bias {
            Some(vb.pp("o_proj").get(hidden_size, "bias")?)
        } else {
            None
        };
        let o_proj = Linear::new(output_weight, output_bias, None);

        let q_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("q_norm"),
            attention_head_size,
            config.rms_norm_eps,
        )?;
        let k_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("k_norm"),
            attention_head_size,
            config.rms_norm_eps,
        )?;
        let v_norm = EmbeddingGemma2RMSNorm::without_scale(config.rms_norm_eps);

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            attention_head_size,
            num_attention_heads,
            num_key_value_heads,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        if self.num_key_value_heads == self.num_attention_heads {
            return Ok(x);
        }
        let repeat_factor = self.num_attention_heads / self.num_key_value_heads;
        let (b, h, s, d) = x.shape().dims4()?;
        x.unsqueeze(2)?
            .expand((b, h, repeat_factor, s, d))?
            .reshape((b, h * repeat_factor, s, d))
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let qkv = self.qkv_proj.forward(hidden_states)?;

        let q_size = self.num_attention_heads * self.attention_head_size;
        let kv_size = self.num_key_value_heads * self.attention_head_size;

        let q = qkv.narrow(D::Minus1, 0, q_size)?.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let k = qkv.narrow(D::Minus1, q_size, kv_size)?.reshape((
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.attention_head_size,
        ))?;
        let v = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?.reshape((
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.attention_head_size,
        ))?;

        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = self.v_norm.forward(&v)?.transpose(1, 2)?;

        let q = apply_rotary(&q, cos, sin, self.attention_head_size)?;
        let k = apply_rotary(&k, cos, sin, self.attention_head_size)?;

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // NOTE: EmbeddingGemma 2 uses an attention scaling of 1.0, as the queries and keys are
        // already RMS-normalized
        let mut attn_weights = q.contiguous()?.matmul(&k.t()?.contiguous()?)?;
        if let Some(attention_bias) = attention_bias {
            attn_weights = attn_weights.broadcast_add(attention_bias)?;
        }
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let context_layer = attn_weights.matmul(&v.contiguous()?)?;

        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;
        self.o_proj.forward(&context_layer)
    }
}

struct EmbeddingGemma2MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    hidden_activation: HiddenAct,

    intermediate_size: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2MLP {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let gate_proj_weight = vb
            .pp("gate_proj")
            .get((config.intermediate_size, config.hidden_size), "weight")?;
        let up_proj_weight = vb
            .pp("up_proj")
            .get((config.intermediate_size, config.hidden_size), "weight")?;
        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = Linear::new(gate_up_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let down_proj = Linear::new(down_proj_weight, None, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            hidden_activation: config.hidden_activation.clone(),
            intermediate_size: config.intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;
        let gate_states = gate_up_states.narrow(D::Minus1, 0, self.intermediate_size)?;
        let gate_states = self.hidden_activation.forward(&gate_states)?;
        let up_states =
            gate_up_states.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;

        self.down_proj.forward(&(gate_states * up_states)?)
    }
}

/// Mixes one layer's slice of the per-layer embeddings (PLE) into the residual stream
struct EmbeddingGemma2PLEBlock {
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: EmbeddingGemma2RMSNorm,
    hidden_activation: HiddenAct,

    span: tracing::Span,
}

impl EmbeddingGemma2PLEBlock {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let per_layer_input_gate = Linear::new(
            vb.pp("per_layer_input_gate").get(
                (config.hidden_size_per_layer_input, config.hidden_size),
                "weight",
            )?,
            None,
            None,
        );
        let per_layer_projection = Linear::new(
            vb.pp("per_layer_projection").get(
                (config.hidden_size, config.hidden_size_per_layer_input),
                "weight",
            )?,
            None,
            None,
        );
        let post_per_layer_input_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("post_per_layer_input_norm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            hidden_activation: config.hidden_activation.clone(),
            span: tracing::span!(tracing::Level::TRACE, "ple_block"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate = self.per_layer_input_gate.forward(hidden_states)?;
        let gate = self.hidden_activation.forward(&gate)?;
        let projected = self
            .per_layer_projection
            .forward(&(gate * per_layer_input)?)?;
        let projected = self.post_per_layer_input_norm.forward(&projected)?;
        hidden_states + projected
    }
}

struct EmbeddingGemma2Layer {
    layer_type: EmbeddingGemma2LayerType,

    input_layernorm: EmbeddingGemma2RMSNorm,
    self_attn: EmbeddingGemma2Attention,
    post_attention_layernorm: EmbeddingGemma2RMSNorm,

    pre_feedforward_layernorm: EmbeddingGemma2RMSNorm,
    mlp: EmbeddingGemma2MLP,
    post_feedforward_layernorm: EmbeddingGemma2RMSNorm,

    ple_block: EmbeddingGemma2PLEBlock,
    layer_scalar: f64,

    span: tracing::Span,
}

impl EmbeddingGemma2Layer {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2TextConfig,
        layer_type: EmbeddingGemma2LayerType,
        attention_head_size: usize,
        num_key_value_heads: usize,
    ) -> Result<Self> {
        let norm = |name: &str| {
            EmbeddingGemma2RMSNorm::load(vb.pp(name), config.hidden_size, config.rms_norm_eps)
        };

        let layer_scalar = vb
            .get(1, "layer_scalar")?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?[0] as f64;

        Ok(Self {
            layer_type,
            input_layernorm: norm("input_layernorm")?,
            self_attn: EmbeddingGemma2Attention::load(
                vb.pp("self_attn"),
                config,
                attention_head_size,
                num_key_value_heads,
            )?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            mlp: EmbeddingGemma2MLP::load(vb.pp("mlp"), config)?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
            ple_block: EmbeddingGemma2PLEBlock::load(vb.pp("ple_block"), config)?,
            layer_scalar,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        per_layer_input: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, attention_bias, cos, sin)?;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let mlp_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_states = self.mlp.forward(&mlp_states)?;
        let mlp_states = self.post_feedforward_layernorm.forward(&mlp_states)?;
        let hidden_states = (residual + mlp_states)?;

        let hidden_states = self.ple_block.forward(&hidden_states, per_layer_input)?;
        hidden_states * self.layer_scalar
    }
}

/// Produces the projection-only per-layer embeddings (PLE) for every layer at once
struct EmbeddingGemma2PLE {
    per_layer_model_projection: Linear,
    per_layer_model_projection_scale: f64,
    per_layer_projection_norm: EmbeddingGemma2RMSNorm,

    num_hidden_layers: usize,
    hidden_size_per_layer_input: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2PLE {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let per_layer_model_projection = Linear::new(
            vb.pp("per_layer_model_projection").get(
                (
                    config.num_hidden_layers * config.hidden_size_per_layer_input,
                    config.hidden_size,
                ),
                "weight",
            )?,
            None,
            None,
        );
        let per_layer_projection_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("per_layer_projection_norm"),
            config.hidden_size_per_layer_input,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            per_layer_model_projection,
            per_layer_model_projection_scale: (config.hidden_size as f64).powf(-0.5),
            per_layer_projection_norm,
            num_hidden_layers: config.num_hidden_layers,
            hidden_size_per_layer_input: config.hidden_size_per_layer_input,
            span: tracing::span!(tracing::Level::TRACE, "ple"),
        })
    }

    /// Returns one `[batch_size, seq_len, hidden_size_per_layer_input]` tensor per layer
    pub fn forward(&self, inputs_embeds: &Tensor) -> Result<Vec<Tensor>> {
        let _enter = self.span.enter();

        let (batch_size, seq_len, _) = inputs_embeds.dims3()?;
        let per_layer_projection = (self.per_layer_model_projection.forward(inputs_embeds)?
            * self.per_layer_model_projection_scale)?;
        let per_layer_projection = per_layer_projection.reshape((
            batch_size,
            seq_len,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        ))?;
        let per_layer_inputs = self
            .per_layer_projection_norm
            .forward(&per_layer_projection)?;

        (0..self.num_hidden_layers)
            .map(|i| per_layer_inputs.i((.., .., i))?.contiguous())
            .collect()
    }
}

struct EmbeddingGemma2Rotary {
    inv_freqs: Tensor,
    rotary_dim: usize,
}

impl EmbeddingGemma2Rotary {
    /// Computes the `cos` and `sin` with shape `[batch_size, 1, seq_len, rotary_dim]` on the fly,
    /// as caching those for all the `max_position_embeddings` would be too memory-expensive
    fn cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len) = position_ids.dims2()?;
        let positions = position_ids
            .flatten_all()?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let freqs = positions.matmul(&self.inv_freqs)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        let shape = (batch_size, 1, seq_len, self.rotary_dim);
        Ok((
            freqs.cos()?.to_dtype(dtype)?.reshape(shape)?,
            freqs.sin()?.to_dtype(dtype)?.reshape(shape)?,
        ))
    }
}

/// Loads a vision linear weight, which may be nested under `linear` as the Gemma 4 vision tower
/// wraps those to optionally clip their inputs and outputs
fn vision_linear_weight(vb: VarBuilder, out_dim: usize, in_dim: usize) -> Result<Tensor> {
    let vb = if vb.contains_tensor("linear.weight") {
        vb.pp("linear")
    } else {
        vb
    };
    vb.get((out_dim, in_dim), "weight")
}

fn load_vision_linear(vb: VarBuilder, out_dim: usize, in_dim: usize) -> Result<Linear> {
    Ok(Linear::new(
        vision_linear_weight(vb, out_dim, in_dim)?,
        None,
        None,
    ))
}

struct EmbeddingGemma2VisionAttention {
    qkv_proj: Linear,
    o_proj: Linear,

    q_norm: EmbeddingGemma2RMSNorm,
    k_norm: EmbeddingGemma2RMSNorm,
    v_norm: EmbeddingGemma2RMSNorm,

    num_attention_heads: usize,
    attention_head_size: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2VisionAttention {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2VisionConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let all_head_size = config.num_attention_heads * config.head_dim;

        let qkv_weight = Tensor::cat(
            &[
                vision_linear_weight(vb.pp("q_proj"), all_head_size, hidden_size)?,
                vision_linear_weight(vb.pp("k_proj"), all_head_size, hidden_size)?,
                vision_linear_weight(vb.pp("v_proj"), all_head_size, hidden_size)?,
            ],
            0,
        )?;

        Ok(Self {
            qkv_proj: Linear::new(qkv_weight, None, None),
            o_proj: load_vision_linear(vb.pp("o_proj"), hidden_size, all_head_size)?,
            q_norm: EmbeddingGemma2RMSNorm::load(
                vb.pp("q_norm"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            k_norm: EmbeddingGemma2RMSNorm::load(
                vb.pp("k_norm"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            v_norm: EmbeddingGemma2RMSNorm::without_scale(config.rms_norm_eps),
            num_attention_heads: config.num_attention_heads,
            attention_head_size: config.head_dim,
            span: tracing::span!(tracing::Level::TRACE, "vision_attention"),
        })
    }

    /// Applies the axial 2D RoPE, rotating the first half of each head by the column (`x`) and
    /// the second half by the row (`y`) of each patch
    fn apply_axial_rotary(
        &self,
        x: &Tensor,
        cos_sin: &EmbeddingGemma2VisionCosSin,
    ) -> Result<Tensor> {
        let half = self.attention_head_size / 2;
        let x_rot = apply_rotary(
            &x.narrow(D::Minus1, 0, half)?,
            &cos_sin.cos_x,
            &cos_sin.sin_x,
            half,
        )?;
        let y_rot = apply_rotary(
            &x.narrow(D::Minus1, half, half)?,
            &cos_sin.cos_y,
            &cos_sin.sin_y,
            half,
        )?;
        Tensor::cat(&[&x_rot, &y_rot], D::Minus1)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos_sin: &EmbeddingGemma2VisionCosSin,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (num_patches, _) = hidden_states.dims2()?;
        let all_head_size = self.num_attention_heads * self.attention_head_size;
        let shape = (
            num_patches,
            self.num_attention_heads,
            self.attention_head_size,
        );

        let qkv = self.qkv_proj.forward(hidden_states)?;
        let q = qkv.narrow(D::Minus1, 0, all_head_size)?.reshape(shape)?;
        let k = qkv
            .narrow(D::Minus1, all_head_size, all_head_size)?
            .reshape(shape)?;
        let v = qkv
            .narrow(D::Minus1, 2 * all_head_size, all_head_size)?
            .reshape(shape)?;

        // [num_heads, num_patches, head_dim]
        let q = self.q_norm.forward(&q)?.transpose(0, 1)?;
        let k = self.k_norm.forward(&k)?.transpose(0, 1)?;
        let v = self.v_norm.forward(&v)?.transpose(0, 1)?.contiguous()?;

        let q = self.apply_axial_rotary(&q, cos_sin)?;
        let k = self.apply_axial_rotary(&k, cos_sin)?;

        // NOTE: The attention scaling is 1.0, as the queries and keys are already RMS-normalized,
        // and the attention is bidirectional with no masking, as images are encoded one at a time
        let attn_weights = q.matmul(&k.t()?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let context_layer = attn_weights.matmul(&v)?;

        let context_layer = context_layer.transpose(0, 1)?.flatten_from(D::Minus2)?;
        self.o_proj.forward(&context_layer)
    }
}

struct EmbeddingGemma2VisionMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    hidden_activation: HiddenAct,

    span: tracing::Span,
}

impl EmbeddingGemma2VisionMLP {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2VisionConfig) -> Result<Self> {
        Ok(Self {
            gate_proj: load_vision_linear(
                vb.pp("gate_proj"),
                config.intermediate_size,
                config.hidden_size,
            )?,
            up_proj: load_vision_linear(
                vb.pp("up_proj"),
                config.intermediate_size,
                config.hidden_size,
            )?,
            down_proj: load_vision_linear(
                vb.pp("down_proj"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            hidden_activation: config.hidden_activation.clone(),
            span: tracing::span!(tracing::Level::TRACE, "vision_mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_states = self
            .hidden_activation
            .forward(&self.gate_proj.forward(hidden_states)?)?;
        let up_states = self.up_proj.forward(hidden_states)?;
        self.down_proj.forward(&(gate_states * up_states)?)
    }
}

struct EmbeddingGemma2VisionLayer {
    input_layernorm: EmbeddingGemma2RMSNorm,
    self_attn: EmbeddingGemma2VisionAttention,
    post_attention_layernorm: EmbeddingGemma2RMSNorm,

    pre_feedforward_layernorm: EmbeddingGemma2RMSNorm,
    mlp: EmbeddingGemma2VisionMLP,
    post_feedforward_layernorm: EmbeddingGemma2RMSNorm,

    span: tracing::Span,
}

impl EmbeddingGemma2VisionLayer {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2VisionConfig) -> Result<Self> {
        let norm = |name: &str| {
            EmbeddingGemma2RMSNorm::load(vb.pp(name), config.hidden_size, config.rms_norm_eps)
        };

        Ok(Self {
            input_layernorm: norm("input_layernorm")?,
            self_attn: EmbeddingGemma2VisionAttention::load(vb.pp("self_attn"), config)?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            mlp: EmbeddingGemma2VisionMLP::load(vb.pp("mlp"), config)?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
            span: tracing::span!(tracing::Level::TRACE, "vision_layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos_sin: &EmbeddingGemma2VisionCosSin,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let attn_states = self.input_layernorm.forward(hidden_states)?;
        let attn_states = self.self_attn.forward(&attn_states, cos_sin)?;
        let attn_states = self.post_attention_layernorm.forward(&attn_states)?;
        let hidden_states = (hidden_states + attn_states)?;

        let mlp_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_states = self.mlp.forward(&mlp_states)?;
        let mlp_states = self.post_feedforward_layernorm.forward(&mlp_states)?;
        hidden_states + mlp_states
    }
}

struct EmbeddingGemma2VisionCosSin {
    cos_x: Tensor,
    sin_x: Tensor,
    cos_y: Tensor,
    sin_y: Tensor,
}

/// The Gemma 4 vision tower, followed by the multimodal embedder that projects the soft tokens
/// into the text model space
struct EmbeddingGemma2Vision {
    input_proj: Linear,
    position_embedding_x: Tensor,
    position_embedding_y: Tensor,
    layers: Vec<EmbeddingGemma2VisionLayer>,
    inv_freqs: Tensor,

    embedding_pre_projection_norm: EmbeddingGemma2RMSNorm,
    embedding_projection: Linear,

    hidden_size: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
    position_embedding_size: usize,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl EmbeddingGemma2Vision {
    pub fn load(
        vision_tower_vb: VarBuilder,
        embed_vision_vb: VarBuilder,
        config: &EmbeddingGemma2VisionConfig,
        text_hidden_size: usize,
    ) -> Result<Self> {
        if config.use_clipped_linears {
            candle::bail!("`use_clipped_linears=true` is not supported for EmbeddingGemma2");
        }
        if config.standardize {
            candle::bail!("`standardize=true` is not supported for EmbeddingGemma2");
        }
        if config.num_key_value_heads != config.num_attention_heads {
            candle::bail!(
                "Grouped-query attention is not supported in the EmbeddingGemma2 vision tower"
            );
        }

        let patch_dim = config.patch_size * config.patch_size * 3;
        let vb = vision_tower_vb.pp("patch_embedder");
        let input_proj = load_vision_linear(vb.pp("input_proj"), config.hidden_size, patch_dim)?;
        let position_embedding_table = vb.get(
            (2, config.position_embedding_size, config.hidden_size),
            "position_embedding_table",
        )?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                EmbeddingGemma2VisionLayer::load(
                    vision_tower_vb.pp(format!("encoder.layers.{i}")),
                    config,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // NOTE: Each half of the head dimension is rotated along one of the two spatial axes
        let inv_freqs = get_inv_freqs(
            config.head_dim / 2,
            config.rope_parameters.rope_theta,
            vision_tower_vb.device(),
            None,
        )?;

        let embedding_projection = Linear::new(
            embed_vision_vb
                .pp("embedding_projection")
                .get((text_hidden_size, config.hidden_size), "weight")?,
            None,
            None,
        );

        Ok(Self {
            input_proj,
            position_embedding_x: position_embedding_table.get(0)?,
            position_embedding_y: position_embedding_table.get(1)?,
            layers,
            inv_freqs,
            embedding_pre_projection_norm: EmbeddingGemma2RMSNorm::without_scale(
                config.rms_norm_eps,
            ),
            embedding_projection,
            hidden_size: config.hidden_size,
            patch_size: config.patch_size,
            pooling_kernel_size: config.pooling_kernel_size,
            position_embedding_size: config.position_embedding_size,
            device: vision_tower_vb.device().clone(),
            dtype: vision_tower_vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "vision"),
        })
    }

    fn cos_sin(&self, positions: Vec<u32>) -> Result<(Tensor, Tensor)> {
        let num_positions = positions.len();
        let positions =
            Tensor::from_vec(positions, (num_positions, 1), &self.device)?.to_dtype(DType::F32)?;
        let freqs = positions.matmul(&self.inv_freqs)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        Ok((
            freqs.cos()?.to_dtype(self.dtype)?,
            freqs.sin()?.to_dtype(self.dtype)?,
        ))
    }

    /// Encodes a single image into `[num_soft_tokens, text_hidden_size]` soft tokens
    pub fn forward(&self, image: &BatchImage) -> Result<Tensor> {
        let _enter = self.span.enter();

        let side = self.patch_size * self.pooling_kernel_size;
        if image.height == 0
            || image.width == 0
            || image.height % side != 0
            || image.width % side != 0
            || image.pixels.len() != image.height * image.width * 3
        {
            candle::bail!(
                "Images are expected to be RGB with a non-zero height and width multiple of {side}, but got {}x{} with {} values",
                image.height,
                image.width,
                image.pixels.len()
            );
        }

        let rows = image.height / self.patch_size;
        let cols = image.width / self.patch_size;
        if rows.max(cols) > self.position_embedding_size {
            candle::bail!(
                "Image of {}x{} exceeds the maximum number of positions",
                image.height,
                image.width
            );
        }
        let num_patches = rows * cols;

        // Patchify in row-major order, flattening each patch in height-width-channel order
        let pixel_values =
            Tensor::from_slice(&image.pixels, image.height * image.width * 3, &self.device)?
                .to_dtype(DType::F32)?;
        let pixel_values = (pixel_values * (1.0 / 255.0))?
            .reshape((rows, self.patch_size, cols, self.patch_size, 3))?
            .permute((0, 2, 1, 3, 4))?
            .reshape((num_patches, self.patch_size * self.patch_size * 3))?;
        // NOTE: Pixels are rescaled to [-1, 1] within the patch embedder
        let pixel_values = ((pixel_values - 0.5)? * 2.0)?.to_dtype(self.dtype)?;

        let xs = (0..num_patches)
            .map(|i| (i % cols) as u32)
            .collect::<Vec<_>>();
        let ys = (0..num_patches)
            .map(|i| (i / cols) as u32)
            .collect::<Vec<_>>();
        let xs_tensor = Tensor::from_slice(&xs, num_patches, &self.device)?;
        let ys_tensor = Tensor::from_slice(&ys, num_patches, &self.device)?;

        let mut hidden_states = self.input_proj.forward(&pixel_values)?;
        hidden_states = (hidden_states + self.position_embedding_x.index_select(&xs_tensor, 0)?)?;
        hidden_states = (hidden_states + self.position_embedding_y.index_select(&ys_tensor, 0)?)?;

        let (cos_x, sin_x) = self.cos_sin(xs)?;
        let (cos_y, sin_y) = self.cos_sin(ys)?;
        let cos_sin = EmbeddingGemma2VisionCosSin {
            cos_x,
            sin_x,
            cos_y,
            sin_y,
        };

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos_sin)?;
        }

        // Average pool every `pooling_kernel_size x pooling_kernel_size` block of patches, in
        // row-major order over the pooled grid, then scale by `sqrt(hidden_size)`
        let k = self.pooling_kernel_size;
        let soft_tokens = hidden_states
            .to_dtype(DType::F32)?
            .reshape((rows / k, k, cols / k, k, self.hidden_size))?
            .mean(3)?
            .mean(1)?
            .reshape(((rows / k) * (cols / k), self.hidden_size))?;
        let soft_tokens = (soft_tokens * (self.hidden_size as f64).sqrt())?;

        let soft_tokens = self
            .embedding_pre_projection_norm
            .forward(&soft_tokens)?
            .to_dtype(self.dtype)?;
        self.embedding_projection.forward(&soft_tokens)
    }
}

pub struct EmbeddingGemma2Model {
    embed_tokens: Embedding,
    embed_scale: f64,
    ple: EmbeddingGemma2PLE,
    layers: Vec<EmbeddingGemma2Layer>,
    norm: EmbeddingGemma2RMSNorm,
    embedding_projection: Linear,

    vision: Option<EmbeddingGemma2Vision>,
    /// Token ids whose positions are replaced by the soft tokens of the images or video frames
    multimodal_token_ids: Vec<u32>,

    rotary: HashMap<EmbeddingGemma2LayerType, EmbeddingGemma2Rotary>,
    sliding_window: usize,

    pad_token_id: u32,
    pool: Pool,

    dtype: DType,
    device: Device,

    span: tracing::Span,
}

impl EmbeddingGemma2Model {
    /// Loads the model, including the vision tower if `load_vision` is set and the model has one
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2Config,
        model_type: ModelType,
        load_vision: bool,
    ) -> Result<Self> {
        let multimodal_config = config;
        let config = &config.text_config;

        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for EmbeddingGemma2")
            }
            ModelType::Embedding(Pool::Splade) => {
                candle::bail!("`splade` pooling is not supported for EmbeddingGemma2")
            }
            ModelType::Embedding(pool) => pool,
        };

        let (vb, vision) = if vb.contains_tensor("language_model.embed_tokens.weight") {
            let vision = match load_vision {
                true => Self::load_vision(&vb, multimodal_config)?,
                false => None,
            };
            (vb.pp("language_model"), vision)
        } else if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            let vision = match load_vision {
                true => Self::load_vision(&vb.pp("model"), multimodal_config)?,
                false => None,
            };
            (vb.pp("model.language_model"), vision)
        } else {
            (vb, None)
        };
        let multimodal_token_ids = match vision {
            Some(_) => [
                multimodal_config.image_token_id,
                multimodal_config.video_token_id,
            ]
            .into_iter()
            .flatten()
            .collect(),
            None => vec![],
        };

        let embed_tokens = Embedding::new(
            vb.pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );
        let embed_scale = (config.hidden_size as f64).sqrt();

        let ple = EmbeddingGemma2PLE::load(vb.pp("ple"), config)?;

        let layer_types = config.layer_types();
        if layer_types.len() != config.num_hidden_layers {
            candle::bail!(
                "`layer_types` has {} entries but `num_hidden_layers` is {}",
                layer_types.len(),
                config.num_hidden_layers
            );
        }

        let mut rotary = HashMap::new();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (layer_idx, &layer_type) in layer_types.iter().enumerate() {
            let (head_dim, num_key_value_heads) = config.layer_config(layer_idx, layer_type)?;

            match rotary.get(&layer_type) {
                Some(EmbeddingGemma2Rotary { rotary_dim, .. }) if *rotary_dim != head_dim => {
                    candle::bail!(
                        "All the layers with the same layer type are expected to share the same `head_dim`"
                    )
                }
                Some(_) => {}
                None => {
                    let inv_freqs =
                        get_inv_freqs(head_dim, config.rope_theta(layer_type), vb.device(), None)?;
                    rotary.insert(
                        layer_type,
                        EmbeddingGemma2Rotary {
                            inv_freqs,
                            rotary_dim: head_dim,
                        },
                    );
                }
            }

            layers.push(EmbeddingGemma2Layer::load(
                vb.pp(format!("layers.{layer_idx}")),
                config,
                layer_type,
                head_dim,
                num_key_value_heads,
            )?);
        }

        let norm =
            EmbeddingGemma2RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let embedding_projection = Linear::new(
            vb.pp("embedding_projection")
                .get((config.embedding_dim, config.hidden_size), "weight")?,
            None,
            None,
        );

        Ok(Self {
            embed_tokens,
            embed_scale,
            ple,
            layers,
            norm,
            embedding_projection,
            vision,
            multimodal_token_ids,
            rotary,
            sliding_window: config.sliding_window,
            pad_token_id: config.pad_token_id.unwrap_or(0),
            pool,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    /// Loads the vision tower if the `vision_config` is defined and its weights are available, as
    /// those are optional e.g. the model may have been saved without them
    fn load_vision(
        vb: &VarBuilder,
        config: &EmbeddingGemma2Config,
    ) -> Result<Option<EmbeddingGemma2Vision>> {
        let Some(vision_config) = &config.vision_config else {
            return Ok(None);
        };
        if !vb.contains_tensor("vision_tower.patch_embedder.input_proj.weight")
            && !vb.contains_tensor("vision_tower.patch_embedder.input_proj.linear.weight")
        {
            tracing::warn!("`vision_config` is defined but the vision tower weights were not found, so only text inputs will be supported");
            return Ok(None);
        }
        if config.image_token_id.is_none() && config.video_token_id.is_none() {
            candle::bail!(
                "`vision_config` is defined but neither `image_token_id` nor `video_token_id` are"
            );
        }

        EmbeddingGemma2Vision::load(
            vb.pp("vision_tower"),
            vb.pp("embed_vision"),
            vision_config,
            config.text_config.hidden_size,
        )
        .map(Some)
    }

    /// Encodes all the images and video frames, returning their soft tokens concatenated in order
    fn embed_images(&self, images: &[BatchImage]) -> Result<Option<Tensor>> {
        if images.is_empty() {
            return Ok(None);
        }
        let Some(vision) = &self.vision else {
            candle::bail!("Image inputs are not supported as the vision tower is not loaded");
        };
        let soft_tokens = images
            .iter()
            .map(|image| vision.forward(image))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&soft_tokens, 0).map(Some)
    }

    /// Creates the additive attention bias with shape `[batch_size, 1, seq_len, seq_len]`, or
    /// `None` if no position needs to be masked out
    fn attention_bias(
        &self,
        input_lengths: &[usize],
        seq_len: usize,
        layer_type: EmbeddingGemma2LayerType,
    ) -> Result<Option<Tensor>> {
        let sliding_window = match layer_type {
            EmbeddingGemma2LayerType::FullAttention => None,
            // NOTE: The bidirectional sliding window is an inclusive radius, so that a token
            // attends to any other token if `abs(q_idx - kv_idx) <= sliding_window`
            EmbeddingGemma2LayerType::SlidingAttention => Some(self.sliding_window),
        };

        let has_padding = input_lengths.iter().any(|&length| length != seq_len);
        let has_window = sliding_window.is_some_and(|window| seq_len > window + 1);
        if !has_padding && !has_window {
            return Ok(None);
        }

        // NOTE: A finite minimum is used rather than `-inf` so that fully masked rows (only
        // possible for padding queries) don't produce NaNs, which would then propagate
        let min_value = match self.dtype {
            DType::F32 => f32::MIN,
            _ => -65504.0,
        };

        let mut bias = Vec::with_capacity(input_lengths.len() * seq_len * seq_len);
        for &length in input_lengths {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let allowed =
                        j < length && sliding_window.is_none_or(|window| i.abs_diff(j) <= window);
                    bias.push(if allowed { 0.0 } else { min_value });
                }
            }
        }

        Tensor::from_vec(
            bias,
            (input_lengths.len(), 1, seq_len, seq_len),
            &self.device,
        )?
        .to_dtype(self.dtype)
        .map(Some)
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let mut input_ids = Vec::with_capacity(batch_size * max_length);
        let mut position_ids = Vec::with_capacity(batch_size * max_length);
        let mut input_lengths = Vec::with_capacity(batch_size);

        // Index into the token embeddings followed by the soft tokens, so that the placeholder
        // tokens are replaced by the soft tokens, in order
        let num_positions = batch_size * max_length;
        let mut gather_indices = Vec::with_capacity(num_positions);
        let mut num_placeholders = 0;

        for i in 0..batch_size {
            let start = batch.cumulative_seq_lengths[i] as usize;
            let end = batch.cumulative_seq_lengths[i + 1] as usize;
            input_lengths.push(end - start);

            for &input_id in &batch.input_ids[start..end] {
                if self.multimodal_token_ids.contains(&input_id) {
                    // NOTE: Placeholders are embedded as padding as those are overwritten anyway
                    input_ids.push(self.pad_token_id);
                    gather_indices.push((num_positions + num_placeholders) as u32);
                    num_placeholders += 1;
                } else {
                    input_ids.push(input_id);
                    gather_indices.push(gather_indices.len() as u32);
                }
            }
            position_ids.extend_from_slice(&batch.position_ids[start..end]);

            for _ in (end - start)..max_length {
                input_ids.push(self.pad_token_id);
                position_ids.push(0);
                gather_indices.push(gather_indices.len() as u32);
            }
        }

        let shape = (batch_size, max_length);
        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;

        let mut hidden_states = (self.embed_tokens.forward(&input_ids)? * self.embed_scale)?;

        if let Some(soft_tokens) = self.embed_images(&batch.images)? {
            let num_soft_tokens = soft_tokens.dim(0)?;
            if num_soft_tokens != num_placeholders {
                candle::bail!("The images produce {num_soft_tokens} soft tokens but the inputs contain {num_placeholders} placeholder tokens");
            }
            // NOTE: Unlike the token embeddings, the soft tokens are not scaled by `embed_scale`
            let hidden_size = hidden_states.dim(D::Minus1)?;
            let gather_indices = Tensor::from_vec(gather_indices, num_positions, &self.device)?;
            hidden_states = Tensor::cat(
                &[
                    &hidden_states.reshape((num_positions, hidden_size))?,
                    &soft_tokens,
                ],
                0,
            )?
            .index_select(&gather_indices, 0)?
            .reshape((batch_size, max_length, hidden_size))?;
        } else if num_placeholders > 0 {
            candle::bail!(
                "The inputs contain {num_placeholders} image placeholder tokens but no images"
            );
        }

        // NOTE: The per-layer inputs are computed from the embeddings including the soft tokens
        let per_layer_inputs = self.ple.forward(&hidden_states)?;

        let mut cos_sin = HashMap::new();
        let mut attention_biases = HashMap::new();
        for (&layer_type, rotary) in &self.rotary {
            cos_sin.insert(layer_type, rotary.cos_sin(&position_ids, self.dtype)?);
            attention_biases.insert(
                layer_type,
                self.attention_bias(&input_lengths, max_length, layer_type)?,
            );
        }

        let mut hidden_states = hidden_states;
        for (layer, per_layer_input) in self.layers.iter().zip(per_layer_inputs.iter()) {
            let (cos, sin) = &cos_sin[&layer.layer_type];
            hidden_states = layer.forward(
                &hidden_states,
                per_layer_input,
                attention_biases[&layer.layer_type].as_ref(),
                cos,
                sin,
            )?;
        }

        let outputs = self.norm.forward(&hidden_states)?;
        // NOTE: The projection is applied per token, which is equivalent to applying it after the
        // mean pooling, but it's required for the raw embeddings to match too
        let outputs = self.embedding_projection.forward(&outputs)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let results = batch
                .pooled_indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    let length = input_lengths[i];
                    match self.pool {
                        Pool::Cls => outputs.i((i..i + 1, 0)),
                        Pool::LastToken => outputs.i((i..i + 1, length - 1)),
                        Pool::Mean => outputs.i((i, ..length))?.sum_keepdim(0)? / (length as f64),
                        Pool::Splade => unreachable!(),
                    }
                })
                .collect::<Result<Vec<Tensor>>>()?;
            Some(Tensor::cat(&results, 0)?)
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            let results = batch
                .raw_indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    outputs.i((i, ..input_lengths[i]))
                })
                .collect::<Result<Vec<Tensor>>>()?;
            Some(Tensor::cat(&results, 0)?)
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for EmbeddingGemma2Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
