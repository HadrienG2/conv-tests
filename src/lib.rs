#![feature(array_chunks)]
#![feature(array_windows)]
#![feature(portable_simd)]
#![feature(trusted_len)]

use core_simd::{LaneCount, Mask32, SimdF32, SupportedLaneCount};
use more_asserts::*;
use std::{iter::TrustedLen, ops::AddAssign};

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
pub const WIDEST: usize = AVX; /* AVX512 */

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

// Iterator summation algorithm that minimizes dependency chain length
#[inline(always)]
fn smart_sum<
    T: Copy + Default + AddAssign<T>,
    I: Iterator<Item = T> + TrustedLen,
    const KERNEL_LEN: usize,
>(
    products: I,
) -> T {
    assert_eq!(products.size_hint().0, KERNEL_LEN);
    let mut target = [T::default(); KERNEL_LEN];
    for (dest, src) in target.iter_mut().zip(products) {
        *dest = src;
    }
    let mut stride = KERNEL_LEN / 2;
    while stride > 0 {
        for i in 0..stride {
            target[i] += target[i + stride];
        }
        stride /= 2;
    }
    target[0]
}

// Perform convolution using a scalar algorithm, let compiler autovectorize it
//
// Inlining is very important, and therefore forced, as the compiler can produce
// much better code when it knows the convolution kernel.
//
#[inline(always)]
pub fn convolve_autovec<const LANES: usize, const KERNEL_LEN: usize, const SMART_SUM: bool>(
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
        let products = data.iter().zip(kernel.iter()).map(|(&x, &k)| x * k);
        *output = if SMART_SUM {
            smart_sum::<_, _, KERNEL_LEN>(products)
        } else {
            products.sum()
        };
    }
}

// TODO: Start building comparative performance benchmarks
// TODO: Try to optimize further by reducing the number of loads, through
//       keeping a ring buffer of last loaded inputs.

// Examples of usage (use cargo asm to show assembly)
const FINITE_DIFF: [Scalar; 2] = [-1.0, 1.0];
const SHARPEN3: [Scalar; 3] = [-0.5, 2.0, -0.5];
const SMOOTH5: [Scalar; 5] = [0.1, 0.2, 0.4, 0.2, 0.1];
const ANTISYM8: [Scalar; 8] = [0.1, -0.2, 0.4, -0.8, 0.8, -0.4, 0.2, -0.1];
// TODO: For AVX-512, could also test a 16-wide kernel

// TODO: Automate generation of the batch of functions below so that we can
//       concisely do it for other convolution implementations. Since Rust
//       doesn't have GATs yet, this will require macros.

pub fn finite_diff_autovec_basic(input: &[Simd<WIDEST>], output: &mut [Simd<WIDEST>]) {
    convolve_autovec::<WIDEST, { FINITE_DIFF.len() }, false>(input, &FINITE_DIFF, output);
}

pub fn sharpen3_autovec_basic(input: &[Simd<WIDEST>], output: &mut [Simd<WIDEST>]) {
    convolve_autovec::<WIDEST, { SHARPEN3.len() }, false>(input, &SHARPEN3, output);
}

pub fn smooth5_autovec_basic(input: &[Simd<WIDEST>], output: &mut [Simd<WIDEST>]) {
    convolve_autovec::<WIDEST, { SMOOTH5.len() }, false>(input, &SMOOTH5, output);
}

pub fn antisym8_autovec_basic(input: &[Simd<WIDEST>], output: &mut [Simd<WIDEST>]) {
    convolve_autovec::<WIDEST, { ANTISYM8.len() }, false>(input, &ANTISYM8, output);
}

pub fn finite_diff_autovec_smart(input: &[Simd<WIDEST>], output: &mut [Simd<WIDEST>]) {
    convolve_autovec::<WIDEST, { FINITE_DIFF.len() }, true>(input, &FINITE_DIFF, output);
}

pub fn sharpen3_autovec_smart(input: &[Simd<WIDEST>], output: &mut [Simd<WIDEST>]) {
    convolve_autovec::<WIDEST, { SHARPEN3.len() }, true>(input, &SHARPEN3, output);
}

pub fn smooth5_autovec_smart(input: &[Simd<WIDEST>], output: &mut [Simd<WIDEST>]) {
    convolve_autovec::<WIDEST, { SMOOTH5.len() }, true>(input, &SMOOTH5, output);
}

pub fn antisym8_autovec_smart(input: &[Simd<WIDEST>], output: &mut [Simd<WIDEST>]) {
    convolve_autovec::<WIDEST, { ANTISYM8.len() }, true>(input, &ANTISYM8, output);
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
    fn finite_diff_autovec_basic_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { FINITE_DIFF.len() }, false>,
            input,
            &FINITE_DIFF,
        )
    }

    #[quickcheck]
    fn finite_diff_autovec_basic_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { FINITE_DIFF.len() }, false>,
            input,
            &FINITE_DIFF,
        )
    }

    #[quickcheck]
    fn sharpen3_autovec_basic_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { SHARPEN3.len() }, false>,
            input,
            &SHARPEN3,
        )
    }

    #[quickcheck]
    fn sharpen3_autovec_basic_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { SHARPEN3.len() }, false>,
            input,
            &SHARPEN3,
        )
    }

    #[quickcheck]
    fn smooth5_autovec_basic_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { SMOOTH5.len() }, false>,
            input,
            &SMOOTH5,
        )
    }

    #[quickcheck]
    fn smooth5_autovec_basic_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { SMOOTH5.len() }, false>,
            input,
            &SMOOTH5,
        )
    }

    #[quickcheck]
    fn antisym8_autovec_basic_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { ANTISYM8.len() }, false>,
            input,
            &ANTISYM8,
        )
    }

    #[quickcheck]
    fn antisym8_autovec_basic_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { ANTISYM8.len() }, false>,
            input,
            &ANTISYM8,
        )
    }

    #[quickcheck]
    fn finite_diff_autovec_smart_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { FINITE_DIFF.len() }, true>,
            input,
            &FINITE_DIFF,
        )
    }

    #[quickcheck]
    fn finite_diff_autovec_smart_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { FINITE_DIFF.len() }, true>,
            input,
            &FINITE_DIFF,
        )
    }

    #[quickcheck]
    fn sharpen3_autovec_smart_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { SHARPEN3.len() }, true>,
            input,
            &SHARPEN3,
        )
    }

    #[quickcheck]
    fn sharpen3_autovec_smart_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { SHARPEN3.len() }, true>,
            input,
            &SHARPEN3,
        )
    }

    #[quickcheck]
    fn smooth5_autovec_smart_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { SMOOTH5.len() }, true>,
            input,
            &SMOOTH5,
        )
    }

    #[quickcheck]
    fn smooth5_autovec_smart_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { SMOOTH5.len() }, true>,
            input,
            &SMOOTH5,
        )
    }

    #[quickcheck]
    fn antisym8_autovec_smart_sse(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<SSE, { ANTISYM8.len() }, true>,
            input,
            &ANTISYM8,
        )
    }

    #[quickcheck]
    fn antisym8_autovec_smart_avx(input: Vec<Scalar>) -> TestResult {
        convolve(
            super::convolve_autovec::<AVX, { ANTISYM8.len() }, true>,
            input,
            &ANTISYM8,
        )
    }
}
