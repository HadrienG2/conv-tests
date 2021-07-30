#![feature(array_chunks)]
#![feature(array_windows)]
#![feature(portable_simd)]

use core_simd::{LaneCount, Mask32, SimdF32, SupportedLaneCount};
use more_asserts::*;

// SIMD processing parameters
pub type Scalar = f32;
pub type Simd<const LANES: usize> = SimdF32<LANES>;
pub type Mask<const LANES: usize> = Mask32<LANES>;

const fn simd_lanes(simd_bits: usize) -> usize {
    // assert_eq!(simd_bits % 8, 0);
    let simd_bytes = simd_bits / 8;
    let scalar_bytes = std::mem::size_of::<Scalar>();
    // assert_gt!(simd_bytes, scalar_bytes);
    simd_bytes / scalar_bytes
}
pub const SSE: usize = simd_lanes(128);
pub const AVX: usize = simd_lanes(256);
// pub const AVX512: usize = simd_lanes(512);

// Divide X by Y, rounding upwards
const fn div_round_up(num: usize, denom: usize) -> usize {
    num / denom + (num % denom != 0) as usize
}

// Set up a storage buffer in a SIMD-friendly layout
pub fn allocate_simd<const LANES: usize>(min_size: usize) -> Vec<Simd<LANES>>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let num_vecs = div_round_up(min_size, LANES);
    vec![[0.0; LANES].into(); num_vecs]
}

// Convert existing scalar data into a SIMD-friendly layout
pub fn simdize<const LANES: usize>(input: &[Scalar]) -> Vec<Simd<LANES>>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // Collect all complete SIMD chunks from input
    let mut chunks = input
        .array_chunks::<LANES>()
        .copied()
        .map(Simd::<LANES>::from_array)
        .collect::<Vec<_>>();

    // If there is a remaining incomplete chunk, zero-pad it
    debug_assert_ge!(input.len(), chunks.len() * LANES);
    if input.len() > chunks.len() * LANES {
        let remainder = &input[chunks.len() * LANES..];
        let mut last_chunk = [0.0; LANES];
        last_chunk[..input.len() % LANES].copy_from_slice(remainder);
        chunks.push(Simd::<LANES>::from_array(last_chunk));
    }
    chunks
}

// Degrade SIMD data into scalar data
pub fn scalarize<const LANES: usize>(slice: &[Simd<LANES>]) -> &[Scalar]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Scalar, LANES * slice.len()) }
}

pub fn scalarize_mut<const LANES: usize>(slice: &mut [Simd<LANES>]) -> &mut [Scalar]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut Scalar, LANES * slice.len())
    }
}

// Perform convolution using a scalar algorithm, let compiler autovectorize it
//
// Inlining is very important, and therefore forced, as the compiler can produce
// much better code when it knows the convolution kernel.
//
#[inline(always)]
pub fn convolve_autovec<const LANES: usize, const KERNEL_LEN: usize>(
    input: &[Simd<LANES>],
    kernel: &[Scalar; KERNEL_LEN],
    output: &mut [Simd<LANES>],
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    let input = scalarize(input);
    let output = scalarize_mut(output);
    assert_ge!(input.len(), kernel.len());
    assert_ge!(output.len(), input.len() - kernel.len() + 1);
    for (data, output) in input
        .array_windows::<{ KERNEL_LEN }>()
        .zip(output.iter_mut())
    {
        *output = data.iter().zip(kernel.iter()).map(|(&x, &k)| x * k).sum()
    }
}

// TODO: Add first explicit SIMD convolution that uses the same algorithm as
//       autovectorization (splat kernel coefficients, load all input sliding
//       windows for each output vector), but leverages FMAs. Test it.
// TODO: Start building comparative performance benchmarks
// TODO: Try to optimize above version by enabling more instruction-level
//       parallelism through the use of multiple summation streams.
// TODO: Try to optimize further by reducing the number of loads, through
//       keeping a ring buffer of last loaded inputs.

// Examples of usage (use cargo asm to show assembly)
const FINITE_DIFF: [Scalar; 2] = [-1.0, 1.0];
const SHARPEN3: [Scalar; 3] = [-0.5, 2.0, -0.5];
const SMOOTH5: [Scalar; 5] = [0.1, 0.2, 0.4, 0.2, 0.1];

// TODO: Automate generation of the batch of functions below so that we can
//       concisely do it for other convolution implementations. Since Rust
//       doesn't have GATs yet, this will require macros.

pub fn finite_diff_autovec_sse(input: &[Simd<SSE>], output: &mut [Simd<SSE>]) {
    convolve_autovec(input, &FINITE_DIFF, output);
}

pub fn finite_diff_autovec_avx(input: &[Simd<AVX>], output: &mut [Simd<AVX>]) {
    convolve_autovec(input, &FINITE_DIFF, output);
}

pub fn sharpen3_autovec_sse(input: &[Simd<SSE>], output: &mut [Simd<SSE>]) {
    convolve_autovec(input, &SHARPEN3, output);
}

pub fn sharpen3_autovec_avx(input: &[Simd<AVX>], output: &mut [Simd<AVX>]) {
    convolve_autovec(input, &SHARPEN3, output);
}

pub fn smooth5_autovec_sse(input: &[Simd<SSE>], output: &mut [Simd<SSE>]) {
    convolve_autovec(input, &SMOOTH5, output);
}

pub fn smooth5_autovec_avx(input: &[Simd<AVX>], output: &mut [Simd<AVX>]) {
    convolve_autovec(input, &SMOOTH5, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // Test division with upwards rounding
    #[quickcheck]
    fn div_round_up(num: usize, denom: usize) -> TestResult {
        if denom == 0 {
            return TestResult::discard();
        }
        let result = super::div_round_up(num, denom);
        if num % denom == 0 {
            TestResult::from_bool(result == num / denom)
        } else {
            TestResult::from_bool(result == num / denom + 1)
        }
    }

    // Test SIMD-friendly allocation
    fn allocate_simd<const LANES: usize>(size: u16) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let size = size as usize;
        let vec = super::allocate_simd::<LANES>(size);
        vec.len() == super::div_round_up(size, LANES)
    }

    #[quickcheck]
    fn allocate_simd_sse(size: u16) -> bool {
        allocate_simd::<SSE>(size)
    }

    #[quickcheck]
    fn allocate_simd_avx(size: u16) -> bool {
        allocate_simd::<AVX>(size)
    }

    // Test conversion of scalar data to SIMD data and back
    fn simdize_scalarize<const LANES: usize>(input: &[Scalar]) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let output_simd = super::simdize::<LANES>(input);
        assert_eq!(output_simd.len(), super::div_round_up(input.len(), LANES));
        let output_scalar = super::scalarize(&output_simd);
        for (input_elem, output_elem) in input.iter().zip(output_scalar.iter()) {
            if input_elem.is_nan() {
                assert!(output_elem.is_nan());
            } else {
                assert_eq!(input_elem, output_elem);
            }
        }
        for tail_elem in &output_scalar[input.len()..] {
            assert_eq!(*tail_elem, 0.0);
        }
        true
    }

    #[quickcheck]
    fn simdize_scalarize_sse(input: Vec<Scalar>) -> bool {
        simdize_scalarize::<SSE>(&input)
    }

    #[quickcheck]
    fn simdize_scalarize_avx(input: Vec<Scalar>) -> bool {
        simdize_scalarize::<AVX>(&input)
    }

    // Generic test of convolution implementations
    #[inline(always)]
    fn convolve<
        Convolution: FnOnce(&[Simd<LANES>], &[Scalar; KERNEL_LEN], &mut [Simd<LANES>]),
        const LANES: usize,
        const KERNEL_LEN: usize,
    >(
        convolution: Convolution,
        input: Vec<Scalar>,
        kernel: &[Scalar; KERNEL_LEN],
    ) -> TestResult
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        if input.len() < KERNEL_LEN {
            return TestResult::discard();
        }
        let input_simd = super::simdize::<LANES>(&input);
        let output_len = input_simd.len() * LANES - KERNEL_LEN + 1;
        let mut output_simd = super::allocate_simd::<LANES>(output_len);

        convolution(&input_simd, &kernel, &mut output_simd);

        let output = scalarize(&output_simd);
        for (out, ins) in output.into_iter().zip(input.array_windows::<KERNEL_LEN>()) {
            let expected = ins
                .iter()
                .zip(kernel.iter())
                .map(|(&x, &k)| x * k)
                .sum::<Scalar>();
            if expected.is_nan() {
                assert!(out.is_nan());
            } else {
                assert_eq!(*out, expected);
            }
        }
        TestResult::passed()
    }

    // TODO: Automate generation of the batch of tests below so that we can
    //       concisely do it for other convolution implementations. Since Rust
    //       doesn't have GATs yet, this will require macros.

    #[quickcheck]
    fn finite_diff_autovec_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { FINITE_DIFF.len() }>,
            input,
            &FINITE_DIFF,
        )
    }

    #[quickcheck]
    fn finite_diff_autovec_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { FINITE_DIFF.len() }>,
            input,
            &FINITE_DIFF,
        )
    }

    #[quickcheck]
    fn sharpen3_autovec_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { SHARPEN3.len() }>,
            input,
            &SHARPEN3,
        )
    }

    #[quickcheck]
    fn sharpen3_autovec_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { SHARPEN3.len() }>,
            input,
            &SHARPEN3,
        )
    }

    #[quickcheck]
    fn smooth5_autovec_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { SMOOTH5.len() }>,
            input,
            &SMOOTH5,
        )
    }

    #[quickcheck]
    fn smooth5_autovec_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { SMOOTH5.len() }>,
            input,
            &SMOOTH5,
        )
    }
}
