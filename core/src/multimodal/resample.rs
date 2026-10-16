//! Port of the 8-bit bicubic resampling from Pillow (`src/libImaging/Resample.c`), which matches
//! the antialiased bicubic interpolation of `torchvision` for `uint8` images within +/-2 values

/// Bits of precision of the fixed-point coefficients
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Bicubic (Catmull-Rom like, with `a=-0.5`) filter with a support of 2
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Coefficients of a single resampling pass
struct Coefficients {
    /// Number of coefficients per output pixel
    ksize: usize,
    /// Pairs of `(first input pixel, number of input pixels)` per output pixel
    bounds: Vec<(usize, usize)>,
    /// Fixed-point coefficients, `ksize` per output pixel
    weights: Vec<i32>,
}

fn precompute_coefficients(in_size: usize, out_size: usize) -> Coefficients {
    const SUPPORT: f64 = 2.0;

    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = SUPPORT * filter_scale;
    let ksize = support.ceil() as usize * 2 + 1;

    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];

    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filter_scale;
        // NOTE: Truncating casts are intentional, as those round the (positive) values
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;

        let mut ww = 0.0;
        for (x, kx) in k.iter_mut().enumerate().take(xmax) {
            let w = bicubic_filter((x as f64 + xmin as f64 - center + 0.5) * ss);
            *kx = w;
            ww += w;
        }
        for x in 0..xmax {
            let w = if ww != 0.0 { k[x] / ww } else { k[x] };
            let w = w * (1u32 << PRECISION_BITS) as f64;
            weights[xx * ksize + x] = if w < 0.0 {
                (-0.5 + w) as i32
            } else {
                (0.5 + w) as i32
            };
        }
        bounds.push((xmin, xmax));
    }

    Coefficients {
        ksize,
        bounds,
        weights,
    }
}

#[inline]
fn clip8(value: i32) -> u8 {
    if value >= (1 << PRECISION_BITS << 8) {
        255
    } else if value <= 0 {
        0
    } else {
        (value >> PRECISION_BITS) as u8
    }
}

/// Resizes an RGB image stored in height-width-channel layout with bicubic interpolation,
/// antialiased when downscaling
pub fn resize_bicubic_rgb(
    pixels: &[u8],
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<u8> {
    assert_eq!(pixels.len(), height * width * 3);

    let horizontal = precompute_coefficients(width, out_width);
    let mut vertical = precompute_coefficients(height, out_height);

    // NOTE: The horizontal pass only processes the rows used by the vertical pass
    let (first_row, last_row) = if out_height != height {
        let (first, _) = vertical.bounds[0];
        let (last_start, last_len) = vertical.bounds[out_height - 1];
        (first, last_start + last_len)
    } else {
        (0, height)
    };

    let (pixels, height, width) = if out_width != width {
        for bound in vertical.bounds.iter_mut() {
            bound.0 -= first_row;
        }
        let rows = last_row - first_row;
        let mut out = vec![0u8; rows * out_width * 3];
        for yy in 0..rows {
            let row = &pixels[(yy + first_row) * width * 3..(yy + first_row + 1) * width * 3];
            for xx in 0..out_width {
                let (xmin, xmax) = horizontal.bounds[xx];
                let k = &horizontal.weights[xx * horizontal.ksize..][..xmax];
                let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
                for (x, &w) in k.iter().enumerate() {
                    let p = &row[(x + xmin) * 3..(x + xmin) * 3 + 3];
                    for c in 0..3 {
                        ss[c] += p[c] as i32 * w;
                    }
                }
                let o = &mut out[(yy * out_width + xx) * 3..][..3];
                for c in 0..3 {
                    o[c] = clip8(ss[c]);
                }
            }
        }
        (std::borrow::Cow::Owned(out), rows, out_width)
    } else {
        (std::borrow::Cow::Borrowed(pixels), height, width)
    };

    if out_height == height {
        return pixels.into_owned();
    }

    let mut out = vec![0u8; out_height * width * 3];
    for yy in 0..out_height {
        let (ymin, ymax) = vertical.bounds[yy];
        let k = &vertical.weights[yy * vertical.ksize..][..ymax];
        for xx in 0..width {
            let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
            for (y, &w) in k.iter().enumerate() {
                let p = &pixels[((y + ymin) * width + xx) * 3..][..3];
                for c in 0..3 {
                    ss[c] += p[c] as i32 * w;
                }
            }
            let o = &mut out[(yy * width + xx) * 3..][..3];
            for c in 0..3 {
                o[c] = clip8(ss[c]);
            }
        }
    }
    out
}
