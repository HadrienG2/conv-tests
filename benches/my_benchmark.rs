use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use paste::paste;
use rand::Fill;
use simd_tests::{Scalar, ANTISYM8, FINITE_DIFF, SHARPEN3, SMOOTH5, WIDEST};

#[derive(Debug)]
enum CacheLevel {
    L1,
    L2,
    LocalL3,
    RemoteL3,
    DRAM,
}

#[derive(Debug)]
enum FigureOfMerit {
    InputBytes,
    OutputBytes,
    Muls,
    Adds,
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    const L1_CAPACITY: usize = 32 * 1024 / std::mem::size_of::<Scalar>();
    const L2_CAPACITY: usize = 512 * 1024 / std::mem::size_of::<Scalar>();
    const L3_CCX_CAPACITY: usize = 4 * 1024 * 1024 / std::mem::size_of::<Scalar>();
    const L3_TOTAL_CAPACITY: usize = 2 * L3_CCX_CAPACITY;
    for figure_of_merit in [
        FigureOfMerit::InputBytes,
        /* FigureOfMerit::OutputBytes, */
        FigureOfMerit::Muls,
        FigureOfMerit::Adds,
    ] {
        let mut group = c.benchmark_group(format!("{:?}", figure_of_merit));
        for cache_level in [
            CacheLevel::L1,
            /*CacheLevel::L2,
            CacheLevel::LocalL3,
            CacheLevel::RemoteL3,
            CacheLevel::DRAM,*/
        ] {
            // Select input/output size to match a certain CPU cache level
            let input_len = match cache_level {
                CacheLevel::L1 => L1_CAPACITY / 4,
                CacheLevel::L2 => (L2_CAPACITY + L1_CAPACITY) / 4,
                CacheLevel::LocalL3 => (L2_CAPACITY + L3_CCX_CAPACITY) / 4,
                CacheLevel::RemoteL3 => (L3_CCX_CAPACITY + L3_TOTAL_CAPACITY) / 4,
                CacheLevel::DRAM => 2 * L3_TOTAL_CAPACITY,
            };

            // Measure throughput as bytes of input processed per second
            macro_rules! generate_benchmarks {
                ($impl:ident) => {
                    generate_benchmarks!($impl, SSE);
                    generate_benchmarks!($impl, AVX);
                    // generate_benchmarks!($impl, AVX512);
                };
                ($impl:ident, $width:ident) => {
                    generate_benchmarks!($impl, $width, FINITE_DIFF);
                    generate_benchmarks!($impl, $width, SHARPEN3);
                    generate_benchmarks!($impl, $width, SMOOTH5);
                    generate_benchmarks!($impl, $width, ANTISYM8);
                };
                ($impl:ident, $width:ident, $kernel:ident) => {
                    generate_benchmarks!($impl, $width, $kernel, false, basic);
                    generate_benchmarks!($impl, $width, $kernel, true, smart);
                };
                ($impl:ident, $width:ident, $kernel:ident, $smart:expr, $suffix:ident) => {
                    let output_elems = input_len - $kernel.len() + 1;
                    let output_bytes = output_elems * std::mem::size_of::<Scalar>();
                    let input_bytes = output_bytes * $kernel.len();
                    let num_muls = output_elems * $kernel.len();
                    let num_adds = num_muls - output_elems;
                    match figure_of_merit {
                        FigureOfMerit::OutputBytes => group.throughput(Throughput::Bytes(output_bytes as _)),
                        FigureOfMerit::InputBytes => group.throughput(Throughput::Bytes(input_bytes as _)),
                        FigureOfMerit::Muls => group.throughput(Throughput::Elements(num_muls as _)),
                        FigureOfMerit::Adds => group.throughput(Throughput::Elements(num_adds as _)),
                    };
                    let mut input = simd_tests::allocate_simd(input_len);
                    let scalar_input = simd_tests::scalarize_mut(&mut input);
                    scalar_input.try_fill(&mut rng).unwrap();
                    let mut output = simd_tests::allocate_simd(output_elems);
                    paste!{
                        group.bench_function(&format!("{:?} cache, {} kernel, {} impl, {} sum, {} vectorization", cache_level, stringify!([<$kernel:lower>]), stringify!($impl), stringify!($suffix), stringify!([<$width:lower>])),
                            |b| {
                                b.iter(|| simd_tests::[<$kernel:lower _ $impl _ $suffix _ $width:lower>](black_box(&input), &mut output));
                                black_box(&mut output);
                            }
                        );
                    }
                }
            }
            generate_benchmarks!(autovec, WIDEST);
            generate_benchmarks!(minimal_loads, WIDEST);
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
