#![feature(array_chunks)]
#![feature(array_windows)]
#![feature(portable_simd)]
#![feature(trusted_len)]

use core_simd::{LaneCount, Mask32, SimdF32, SupportedLaneCount};
use more_asserts::*;
use paste::paste;
use std::ops::AddAssign;

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
#[cfg(target_feature = "avx")]
pub const WIDEST: usize = AVX; /* AVX512 */
#[cfg(not(target_feature = "avx"))]
pub const WIDEST: usize = SSE; /* AVX512 */

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
//
// Inlining is very important, and therefore forced, as we want the summation to
// be carried out in registers whenever possible.
//
#[inline(always)]
fn smart_sum<T: Copy + Default + AddAssign<T>, I: Iterator<Item = T>, const KERNEL_LEN: usize>(
    products: I,
) -> T {
    // Collect the inputs
    assert_eq!(products.size_hint().0, KERNEL_LEN);
    let mut buffer = [T::default(); KERNEL_LEN];
    for (dest, src) in buffer.iter_mut().zip(products) {
        *dest = src;
    }

    // Perform the summation using a binary tree algorithm
    let mut stride = KERNEL_LEN.next_power_of_two() / 2;
    while stride > 0 {
        for i in 0..stride.min(KERNEL_LEN - stride) {
            buffer[i] += buffer[i + stride];
        }
        stride /= 2;
    }
    buffer[0]
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
    for (data, output) in input.array_windows::<KERNEL_LEN>().zip(output.iter_mut()) {
        let products = data.iter().zip(kernel.iter()).map(|(&x, &k)| x * k);
        *output = if SMART_SUM {
            smart_sum::<_, _, KERNEL_LEN>(products)
        } else {
            products.sum()
        };
    }
}

// Manually vectorized version of the autovectorized code
//
// Slightly faster because the autovectorizer failed to take advantage of data
// alignment, which the manually vectorized version does leverage.
//
struct Manual<const LANES: usize>;

macro_rules! gen_manual_convolve {
    ($($lanes:expr),*) => {
        $( paste!{
            impl Manual<$lanes> {
                #[inline(always)]
                fn convolve<const KERNEL_LEN: usize, const SMART_SUM: bool>(
                    input: &[Simd<$lanes>],
                    kernel: &[Scalar; KERNEL_LEN],
                    output: &mut [Simd<$lanes>]
                ) {
                    // Validate inputs
                    assert_ge!(input.len()*$lanes, KERNEL_LEN, "Convolution input is smaller than convolution kernel length");
                    assert_ge!(output.len()*$lanes, input.len()*$lanes - KERNEL_LEN + 1, "Convolution output buffer is smaller than output length");

                    // Perform vectorized convolution
                    let kernel_len_vecs = div_round_up(KERNEL_LEN-1, $lanes) + 1;
                    for (out_vec, in_aligned_vecs) in output.iter_mut().zip(input.windows(kernel_len_vecs)) {
                        // Collect unaligned convolution inputs
                        let in_vecs_iter = scalarize(in_aligned_vecs)
                            .array_windows::<$lanes>()
                            .copied()
                            .map(Simd::<$lanes>::from_array);

                        // Compute products with kernel
                        let products = kernel.iter().zip(in_vecs_iter).map(|(&kernel_elem, in_vec)| {
                            in_vec * Simd::<$lanes>::splat(kernel_elem)
                        });

                        // Sum inputs using an algorithm of variable cleverness
                        *out_vec = if SMART_SUM {
                            smart_sum::<_, _, KERNEL_LEN>(products)
                        } else {
                            products.sum()
                        };
                    }

                    // Compute tail elements (if any) using scalar algorithm
                    let num_out_vecs = output.len();
                    let num_ins_windows = input.windows(kernel_len_vecs).count();
                    debug_assert_ge!(num_out_vecs, num_ins_windows);
                    if num_out_vecs > num_ins_windows {
                        convolve_autovec::<$lanes, KERNEL_LEN, false>(&input[num_ins_windows..], kernel, &mut output[num_ins_windows..]);
                    }
                }
            }
        } )*
    };
}

gen_manual_convolve!(4, 8 /*, 16*/);

// Element-shifting interpolation between two consecutive vectors
// TODO: Should codegen this using macros

#[allow(unused)]
fn simd_shift_2(left: Simd<2>, right: Simd<2>, shift: usize) -> Simd<2> {
    match shift {
        0 => left,
        1 => left.shuffle::<{ [1, 2] }>(right),
        2 => right,
        _ => unreachable!("Bad shift value"),
    }
}

#[allow(unused)]
fn simd_shift_4(left: Simd<4>, right: Simd<4>, shift: usize) -> Simd<4> {
    match shift {
        0 => left,
        1 => left.shuffle::<{ [1, 2, 3, 4] }>(right),
        2 => left.shuffle::<{ [2, 3, 4, 5] }>(right),
        3 => left.shuffle::<{ [3, 4, 5, 6] }>(right),
        4 => right,
        _ => unreachable!("Bad shift value"),
    }
}

fn simd_shift_8(left: Simd<8>, right: Simd<8>, shift: usize) -> Simd<8> {
    match shift {
        0 => left,
        1 => left.shuffle::<{ [1, 2, 3, 4, 5, 6, 7, 8] }>(right),
        2 => left.shuffle::<{ [2, 3, 4, 5, 6, 7, 8, 9] }>(right),
        3 => left.shuffle::<{ [3, 4, 5, 6, 7, 8, 9, 10] }>(right),
        4 => left.shuffle::<{ [4, 5, 6, 7, 8, 9, 10, 11] }>(right),
        5 => left.shuffle::<{ [5, 6, 7, 8, 9, 10, 11, 12] }>(right),
        6 => left.shuffle::<{ [6, 7, 8, 9, 10, 11, 12, 13] }>(right),
        7 => left.shuffle::<{ [7, 8, 9, 10, 11, 12, 13, 14] }>(right),
        8 => right,
        _ => unreachable!("Bad shift value"),
    }
}

// TODO: Should add a _16 version for AVX-512

// Variant of the convolution algorithm that minimizes the number of loads and
// makes the loads as efficient as possible, at the cost of using a lot more
// shuffles. On Zen 2, this is not a good tradeoff.
struct MinimalLoads<const LANES: usize>;

macro_rules! gen_minimal_loads_convolve {
    ($($lanes:expr),*) => {
        $( paste!{
            impl MinimalLoads<$lanes> {
                #[inline(always)]
                fn convolve<const KERNEL_LEN: usize, const SMART_SUM: bool>(
                    input: &[Simd<$lanes>],
                    kernel: &[Scalar; KERNEL_LEN],
                    output: &mut [Simd<$lanes>]
                ) {
                    // Validate inputs
                    assert_ge!(input.len()*$lanes, KERNEL_LEN, "Convolution input is smaller than convolution kernel length");
                    assert_ge!(output.len()*$lanes, input.len()*$lanes - KERNEL_LEN + 1, "Convolution output buffer is smaller than output length");

                    // Perform vectorized convolution
                    let kernel_len_vecs = div_round_up(KERNEL_LEN-1, $lanes) + 1;
                    for (out_vec, in_aligned_vecs) in output.iter_mut().zip(input.windows(kernel_len_vecs)) {
                        let products = kernel.iter().enumerate().map(|(shift, &kernel_elem)| {
                            let vec_shift = shift / $lanes;
                            let elem_shift = shift % $lanes;
                            let left_vec = in_aligned_vecs[vec_shift];
                            let in_vec = if elem_shift == 0 {
                                left_vec
                            } else {
                                let right_vec = in_aligned_vecs[vec_shift + 1];
                                [< simd_shift_ $lanes >](left_vec, right_vec, elem_shift)
                            };
                            in_vec * Simd::<$lanes>::splat(kernel_elem)
                        });
                        *out_vec = if SMART_SUM {
                            smart_sum::<_, _, KERNEL_LEN>(products)
                        } else {
                            products.sum()
                        };
                    }

                    // Compute tail elements (if any) using scalar algorithm
                    let num_out_vecs = output.len();
                    let num_ins_windows = input.windows(kernel_len_vecs).count();
                    debug_assert_ge!(num_out_vecs, num_ins_windows);
                    if num_out_vecs > num_ins_windows {
                        convolve_autovec::<$lanes, KERNEL_LEN, false>(&input[num_ins_windows..], kernel, &mut output[num_ins_windows..]);
                    }
                }
            }
        } )*
    };
}

gen_minimal_loads_convolve!(4, 8 /*, 16*/);

// A shuffle pattern that is efficient on x86 CPUs with SSE/AVX
// TODO: Should codegen this using macros

#[allow(unused)]
fn simd_shuf2_4(base: Simd<4>, shift4: Simd<4>) -> Simd<4> {
    // base   is [ 0 1 2 3 ]
    // shift4 is [ 4 5 6 7 ]
    // output is [ 2 3 4 5 ]
    base.shuffle::<{ [2, 3, 4, 5] }>(shift4)
}

fn simd_shuf2_8(base: Simd<8>, shift4: Simd<8>) -> Simd<8> {
    // base   is [ 0 1 2 3 | 4 5 6 7 ]
    // shift4 is [ 4 5 6 7 | 8 9 10 11 ]
    // output is [ 2 3 4 5 | 6 7 8 9 ]
    base.shuffle::<{ [2, 3, 8, 9, 6, 7, 12, 13] }>(shift4)
}

// TODO: Should add a _16 version for AVX-512

// Variant of the convolution algorithm that tries to strike a balance between
// reducing the number of unaligned loads (which bound the performance of the
// naive algorithm) and adding as few shuffles as possible (since these compete
// with adds an muls for SIMD execution ports on Zen2).
//
// Interestingly enough, this is still not beneficial, because it results in
// turning memory operands into MOVs, which also has a cost.
//
struct Shuf2Loadu<const LANES: usize>;

macro_rules! gen_shuf2_loadu_convolve {
    ($($lanes:expr),*) => {
        $( paste!{
            impl Shuf2Loadu<$lanes> {
                #[inline(always)]
                fn convolve<const KERNEL_LEN: usize, const SMART_SUM: bool>(
                    input: &[Simd<$lanes>],
                    kernel: &[Scalar; KERNEL_LEN],
                    output: &mut [Simd<$lanes>]
                ) {
                    // Validate inputs
                    assert_ge!(input.len()*$lanes, KERNEL_LEN, "Convolution input is smaller than convolution kernel length");
                    assert_ge!(output.len()*$lanes, input.len()*$lanes - KERNEL_LEN + 1, "Convolution output buffer is smaller than output length");

                    // Perform vectorized convolution
                    let kernel_len_vecs = div_round_up(KERNEL_LEN-1, $lanes) + 1;
                    for (out_vec, in_aligned_vecs) in output.iter_mut().zip(input.windows(kernel_len_vecs)) {
                        // Collect unaligned convolution inputs
                        let in_vecs_iter = scalarize(in_aligned_vecs)
                            .array_windows::<$lanes>()
                            .copied()
                            .map(Simd::<$lanes>::from_array);
                        let mut in_vecs = [Default::default(); KERNEL_LEN];
                        for (in_vec, data) in in_vecs.iter_mut().zip(in_vecs_iter) {
                            *in_vec = data;
                        }

                        // Compute products with kernel, only using a subset of inputs
                        let products = kernel.iter().enumerate().map(|(idx, &kernel_elem)| {
                            let elem_idx = idx % $lanes;
                            let in_vec = if (elem_idx == 2 || elem_idx == 3) && idx < KERNEL_LEN-2 {
                                // Inputs 2 and 3 are interpolated from inputs (0, 1, 4, 5) using shuffles
                                [< simd_shuf2_ $lanes >](in_vecs[idx-2], in_vecs[idx+2])
                            } else {
                                // Other inputs are accessed using unaligned loads
                                in_vecs[idx]
                            };
                            in_vec * Simd::<$lanes>::splat(kernel_elem)
                        });

                        // Sum inputs using an algorithm of variable cleverness
                        *out_vec = if SMART_SUM {
                            smart_sum::<_, _, KERNEL_LEN>(products)
                        } else {
                            products.sum()
                        };
                    }

                    // Compute tail elements (if any) using scalar algorithm
                    let num_out_vecs = output.len();
                    let num_ins_windows = input.windows(kernel_len_vecs).count();
                    debug_assert_ge!(num_out_vecs, num_ins_windows);
                    if num_out_vecs > num_ins_windows {
                        convolve_autovec::<$lanes, KERNEL_LEN, false>(&input[num_ins_windows..], kernel, &mut output[num_ins_windows..]);
                    }
                }
            }
        } )*
    };
}

gen_shuf2_loadu_convolve!(4, 8 /*, 16*/);

// TODO: On Intel processors, FMA might perform better than mul + add (while the
//       reverse is expected on AMD Zen processors)

// Examples of usage (use cargo asm to show assembly)
pub const FINITE_DIFF: [Scalar; 2] = [-1.0, 1.0];
pub const SHARPEN3: [Scalar; 3] = [-0.5, 2.0, -0.5];
pub const SMOOTH5: [Scalar; 5] = [0.1, 0.2, 0.4, 0.2, 0.1];
pub const ANTISYM8: [Scalar; 8] = [0.1, -0.2, 0.4, -0.8, 0.8, -0.4, 0.2, -0.1];
// TODO: For AVX-512, could also test a 16-wide kernel

// Generate example implementations using above kernels
macro_rules! generate_examples {
    ($impl:ident) => {
        generate_examples!($impl, SSE);
        generate_examples!($impl, AVX);
        // generate_examples!($impl, AVX512);
    };
    ($impl:ident, $width:ident) => {
        generate_examples!($impl, $width, FINITE_DIFF);
        generate_examples!($impl, $width, SHARPEN3);
        generate_examples!($impl, $width, SMOOTH5);
        generate_examples!($impl, $width, ANTISYM8);
    };
    ($impl:ident, $width:ident, $kernel:ident) => {
        generate_examples!($impl, $width, $kernel, false, basic);
        generate_examples!($impl, $width, $kernel, true, smart);
    };
    (autovec, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
        paste!{
            #[inline(never)]
            pub fn [<$kernel:lower _autovec_ $suffix _ $width:lower>](input: &[Simd<$width>], output: &mut [Simd<$width>]) {
                convolve_autovec::<$width, { $kernel.len() }, $smart>(input, &$kernel, output);
            }
        }
    };
    ($simd_impl:ident, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
        paste!{
            #[inline(never)]
            pub fn [<$kernel:lower _ $simd_impl:snake:lower _ $suffix _ $width:lower>](input: &[Simd<$width>], output: &mut [Simd<$width>]) {
                $simd_impl::<$width>::convolve::<{ $kernel.len() }, $smart>(input, &$kernel, output);
            }
        }
    }
}

// No need to support multiple vector widths for autovectorized version, it will
// use max-width AVX with unaligned operands anyway (and that's okay)
generate_examples!(autovec, WIDEST);
generate_examples!(Manual, WIDEST);
generate_examples!(Shuf2Loadu, WIDEST);
generate_examples!(MinimalLoads, WIDEST);

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
        // Reject unreasonable test inputs
        if input.len() < KERNEL_LEN || !input.iter().copied().all(Scalar::is_normal) {
            return TestResult::discard();
        }

        // Normalize input magnitude to [eps, 1] range for easier error analysis
        let input_magnitude = input
            .iter()
            .fold(Scalar::MIN_POSITIVE, |acc, x| acc.max(x.abs()));
        let input = input
            .into_iter()
            .map(|x| {
                let mut normalized = x / input_magnitude;
                if normalized.abs() < Scalar::EPSILON {
                    normalized = Scalar::EPSILON.copysign(x);
                }
                normalized
            })
            .collect::<Vec<_>>();

        // Prepare and perform the convolution
        let input_simd = super::simdize::<LANES>(&input);
        let output_len = input_simd.len() * LANES - KERNEL_LEN + 1;
        let mut output_simd = super::allocate_simd::<LANES>(output_len);
        convolution(&input_simd, &kernel, &mut output_simd);

        // Check convolution results against a basic reference implementation
        let output = scalarize(&output_simd);
        for (output_idx, (out, ins)) in output
            .into_iter()
            .zip(input.array_windows::<KERNEL_LEN>())
            .enumerate()
        {
            let expected = ins
                .iter()
                .zip(kernel.iter())
                .map(|(&x, &k)| x * k)
                .sum::<Scalar>();
            assert_le!(
                (*out - expected).abs(),
                2.0 * Scalar::EPSILON,
                "At output index {}/{}, fed inputs {:?} (from input indices {}->{}/{}) into kernel {:?} and got output {} instead of expected {}",
                output_idx,
                output_len-1,
                ins,
                output_idx,
                output_idx + KERNEL_LEN-1,
                input.len()-1,
                kernel,
                out,
                expected
            );
        }
        TestResult::passed()
    }

    // Generate tests for a given convolution implementation, for all SIMD
    // widths, example kernels, and summation algorithms
    macro_rules! generate_tests {
        ($impl:ident) => {
            generate_tests!($impl, SSE);
            generate_tests!($impl, AVX);
            // generate_tests!($impl, AVX512);
        };
        ($impl:ident, $width:ident) => {
            generate_tests!($impl, $width, FINITE_DIFF);
            generate_tests!($impl, $width, SHARPEN3);
            generate_tests!($impl, $width, SMOOTH5);
            generate_tests!($impl, $width, ANTISYM8);
        };
        ($impl:ident, $width:ident, $kernel:ident) => {
            generate_tests!($impl, $width, $kernel, false, basic);
            generate_tests!($impl, $width, $kernel, true, smart);
        };
        (autovec, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
            paste!{
                #[quickcheck]
                fn [<$kernel:lower _autovec_ $suffix _ $width:lower>](input: Vec<Scalar>) -> TestResult {
                    convolve(
                        super::convolve_autovec::<$width, { $kernel.len() }, $smart>,
                        input,
                        &$kernel,
                    )
                }
            }
        };
        ($simd_impl:ident, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
            paste!{
                #[quickcheck]
                fn [<$kernel:lower _ $simd_impl:snake:lower _ $suffix _ $width:lower>](input: Vec<Scalar>) -> TestResult {
                    convolve(
                        super::$simd_impl::<$width>::convolve::<{ $kernel.len() }, $smart>,
                        input,
                        &$kernel,
                    )
                }
            }
        }
    }
    generate_tests!(autovec);
    generate_tests!(Manual);
    generate_tests!(MinimalLoads);
    generate_tests!(Shuf2Loadu);
}
