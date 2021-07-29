#![feature(array_chunks)]
#![feature(array_windows)]
#![feature(portable_simd)]

use core_simd::{LaneCount, Mask32, SimdF32, SupportedLaneCount};
use more_asserts::*;

// SIMD processing parameters
type Scalar = f32;
type Simd<const LANES: usize> = SimdF32<LANES>;
type Mask<const LANES: usize> = Mask32<LANES>;

const fn simd_lanes(simd_bits: usize) -> usize {
    // assert_eq!(simd_bits % 8, 0);
    let simd_bytes = simd_bits / 8;
    let scalar_bytes = std::mem::size_of::<Scalar>();
    // assert_gt!(simd_bytes, scalar_bytes);
    simd_bytes / scalar_bytes
}
const SSE: usize = simd_lanes(128);
const AVX: usize = simd_lanes(256);
// const AVX512: usize = simd_lanes(512);

// Allocate storage in a SIMD-friendly layout
fn allocate_simd<const LANES: usize>(min_size: usize) -> Vec<Simd<LANES>>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let num_vecs = min_size / LANES + (min_size % LANES != 0) as usize;
    vec![[0.0; LANES].into(); num_vecs]
}

// Degrade SIMD data into scalar data
fn scalarize<const LANES: usize>(slice: &[Simd<LANES>]) -> &[Scalar]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Scalar, LANES * slice.len()) }
}

fn scalarize_mut<const LANES: usize>(slice: &mut [Simd<LANES>]) -> &mut [Scalar]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut Scalar, LANES * slice.len())
    }
}

// Perform convolution using a scalar algorithm (that may be autovectorized)
pub fn convolve_scalar<const LANES: usize, const KERNEL_LEN: usize>(
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
        *output = data
            .iter()
            .copied()
            .zip(kernel.iter().copied())
            .map(|(x, k)| x * k)
            .sum()
    }
}

// Examples of usage
const FINITE_DIFF: [Scalar; 2] = [-1.0, 1.0];
const SHARPEN3: [Scalar; 3] = [-0.5, 2.0, -0.5];
const SMOOTH5: [Scalar; 5] = [0.1, 0.2, 0.4, 0.2, 0.1];

pub fn finite_diff_scalar<const LANES: usize>(input: &[Simd<LANES>], output: &mut [Simd<LANES>])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    convolve_scalar(input, &FINITE_DIFF, output);
}

pub fn sharpen3_scalar<const LANES: usize>(input: &[Simd<LANES>], output: &mut [Simd<LANES>])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    convolve_scalar(input, &SHARPEN3, output);
}

pub fn smooth5_scalar<const LANES: usize>(input: &[Simd<LANES>], output: &mut [Simd<LANES>])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    convolve_scalar(input, &SMOOTH5, output);
}

// Main function
fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    // Test of SIMD allocation
    fn allocate_simd<const LANES: usize>(size: u16) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let size = size as usize;
        let vec = super::allocate_simd::<LANES>(size);
        vec.len() == size / LANES + (size % LANES != 0) as usize
    }

    #[quickcheck]
    fn allocate_simd_sse(size: u16) -> bool {
        allocate_simd::<SSE>(size)
    }

    #[quickcheck]
    fn allocate_simd_avx(size: u16) -> bool {
        allocate_simd::<AVX>(size)
    }

    // TODO: Extract into a top-level simdize utility and test that + scalarize
    fn into_simd<const LANES: usize>(input: &[Scalar]) -> Vec<Simd<LANES>>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // FIMXE: zero-pad with remainder, if any
        input
            .array_chunks::<LANES>()
            .copied()
            .map(Simd::<LANES>::from_array)
            .collect()
    }

    // TODO: Also test AVX using same approach as above
    #[quickcheck]
    fn finite_diff_scalar_sse(input: Vec<Scalar>) -> TestResult {
        if input.len() < FINITE_DIFF.len() {
            return TestResult::discard();
        }
        let input_simd = into_simd::<SSE>(&input);

        let output_len = input.len() - FINITE_DIFF.len() + 1;
        let mut output_simd = super::allocate_simd::<SSE>(output_len);

        finite_diff_scalar(&input_simd[..], &mut output_simd[..]);

        let output = &scalarize(&output_simd)[..output_len];

        for (out, ins) in output
            .into_iter()
            .zip(input.array_windows::<{ FINITE_DIFF.len() }>())
        {
            assert_eq!(
                *out,
                ins.iter()
                    .zip(FINITE_DIFF.into_iter())
                    .map(|(x, k)| x * k)
                    .sum()
            );
        }

        TestResult::passed()
    }
}
