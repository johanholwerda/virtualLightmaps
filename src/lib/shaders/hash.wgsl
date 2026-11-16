// Integer hashes -> uniform f32 in [0, 1).
//
// Naming: hash<inputs><outputs><it> where <it> is the input type,
//   'uf' = u32 seed -> f32 output, 'ff' = f32 seed -> f32 output.
//
// Everything is built from one primitive, pcgHash, a bijective 32-bit finalizer:
//   * N inputs are folded into one word (foldN),
//   * M outputs are produced by chaining pcgHash M times.
// Cost is (N - 1) + 2 * M integer multiplies, which is cheaper than the previous
// implementation for 9 of the 16 combinations and at most 2 multiplies more for
// the rest (those were cheap only because they fabricated the 2nd/3rd/4th
// component as `n * <constant>`, which added no entropy at all).

// ---------------------------------------------------------------------------
// u32 -> f32 in [0, 1)
//
// Splices the top 23 bits into the mantissa of an exponent-127 float, giving a
// value in [1, 2), then subtracts 1. No integer->float conversion and no divide,
// and it keeps the well-mixed high bits rather than masking off the sign bit.
// ---------------------------------------------------------------------------
fn uf1(v: u32) -> f32 {
    return bitcast<f32>((v >> 9u) | 0x3f800000u) - 1.0;
}

fn uf2(v: vec2<u32>) -> vec2<f32> {
    return bitcast<vec2<f32>>((v >> vec2<u32>(9u)) | vec2<u32>(0x3f800000u)) - 1.0;
}

fn uf3(v: vec3<u32>) -> vec3<f32> {
    return bitcast<vec3<f32>>((v >> vec3<u32>(9u)) | vec3<u32>(0x3f800000u)) - 1.0;
}

fn uf4(v: vec4<u32>) -> vec4<f32> {
    return bitcast<vec4<f32>>((v >> vec4<u32>(9u)) | vec4<u32>(0x3f800000u)) - 1.0;
}

// ---------------------------------------------------------------------------
// PCG RXS-M-XS 32-bit finalizer (Melissa O'Neill, pcg-random.org).
// A bijection, so distinct seeds never collide and chaining it cannot shorten
// the output range.
// ---------------------------------------------------------------------------
fn pcgHash(seed: u32) -> u32 {
    let state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcgChain2(seed: u32) -> vec2<u32> {
    let n0 = pcgHash(seed);
    return vec2<u32>(n0, pcgHash(n0));
}

fn pcgChain3(seed: u32) -> vec3<u32> {
    let n0 = pcgHash(seed);
    let n1 = pcgHash(n0);
    return vec3<u32>(n0, n1, pcgHash(n1));
}

fn pcgChain4(seed: u32) -> vec4<u32> {
    let n0 = pcgHash(seed);
    let n1 = pcgHash(n0);
    let n2 = pcgHash(n1);
    return vec4<u32>(n0, n1, n2, pcgHash(n2));
}

// ---------------------------------------------------------------------------
// Fold N seed words into one.
//
// The `v ^ (v >> 16u)` premix is what makes the ff (float seed) entry points
// safe. A float like 0.5 or 3.0/1024.0 bitcasts to a word whose low bits are all
// zero, so multiplying it during the fold pushes every informative bit off the
// top of the register and the seeds collapse onto each other. Folding the high
// bits down first costs no multiplies and is a no-op for small integer seeds.
//
// The multipliers are odd and large (2^32 / golden ratio and friends) so that
// neighbouring lattice coordinates stay far apart after folding.
// ---------------------------------------------------------------------------
fn fold2(p: vec2<u32>) -> u32 {
    let m = p ^ (p >> vec2<u32>(16u));
    return m.x + m.y * 0x9e3779b9u;
}

fn fold3(p: vec3<u32>) -> u32 {
    let m = p ^ (p >> vec3<u32>(16u));
    return m.x + m.y * 0x9e3779b9u + m.z * 0x85ebca6bu;
}

fn fold4(p: vec4<u32>) -> u32 {
    let m = p ^ (p >> vec4<u32>(16u));
    return m.x + m.y * 0x9e3779b9u + m.z * 0x85ebca6bu + m.w * 0xc2b2ae35u;
}

// ---------------------------------------------------------------------------
// 1 input
// ---------------------------------------------------------------------------
fn hash11ff(seed: f32) -> f32 {
	return hash11uf(bitcast<u32>(seed));
}

fn hash11uf(seed: u32) -> f32 {
    return uf1(pcgHash(seed));
}

fn hash12ff(seed: f32) -> vec2<f32> {
	return hash12uf(bitcast<u32>(seed));
}

fn hash12uf(seed: u32) -> vec2<f32> {
    return uf2(pcgChain2(seed));
}

fn hash13ff(seed: f32) -> vec3<f32> {
	return hash13uf(bitcast<u32>(seed));
}

fn hash13uf(seed: u32) -> vec3<f32> {
    return uf3(pcgChain3(seed));
}

fn hash14ff(seed: f32) -> vec4<f32> {
	return hash14uf(bitcast<u32>(seed));
}

fn hash14uf(seed: u32) -> vec4<f32> {
    return uf4(pcgChain4(seed));
}

// ---------------------------------------------------------------------------
// 2 inputs
// ---------------------------------------------------------------------------
fn hash21ff(seed: vec2<f32>) -> f32 {
	return hash21uf((bitcast<vec2<u32>>(seed)));
}

fn hash21uf(seed: vec2<u32>) -> f32 {
    return uf1(pcgHash(fold2(seed)));
}

fn hash22ff(seed: vec2<f32>) -> vec2<f32> {
	return hash22uf((bitcast<vec2<u32>>(seed)));
}

fn hash22uf(seed: vec2<u32>) -> vec2<f32> {
    return uf2(pcgChain2(fold2(seed)));
}

fn hash23ff(seed: vec2<f32>) -> vec3<f32> {
	return hash23uf((bitcast<vec2<u32>>(seed)));
}

fn hash23uf(seed: vec2<u32>) -> vec3<f32> {
    return uf3(pcgChain3(fold2(seed)));
}

fn hash24ff(seed: vec2<f32>) -> vec4<f32> {
	return hash24uf((bitcast<vec2<u32>>(seed)));
}

fn hash24uf(seed: vec2<u32>) -> vec4<f32> {
    return uf4(pcgChain4(fold2(seed)));
}

// ---------------------------------------------------------------------------
// 3 inputs
// ---------------------------------------------------------------------------
fn hash31ff(seed: vec3<f32>) -> f32 {
	return hash31uf((bitcast<vec3<u32>>(seed)));
}

fn hash31uf(seed: vec3<u32>) -> f32 {
    return uf1(pcgHash(fold3(seed)));
}

fn hash32ff(seed: vec3<f32>) -> vec2<f32> {
	return hash32uf((bitcast<vec3<u32>>(seed)));
}

fn hash32uf(seed: vec3<u32>) -> vec2<f32> {
    return uf2(pcgChain2(fold3(seed)));
}

fn hash33ff(seed: vec3<f32>) -> vec3<f32> {
	return hash33uf((bitcast<vec3<u32>>(seed)));
}

fn hash33uf(seed: vec3<u32>) -> vec3<f32> {
    return uf3(pcgChain3(fold3(seed)));
}

fn hash34ff(seed: vec3<f32>) -> vec4<f32> {
	return hash34uf((bitcast<vec3<u32>>(seed)));
}

fn hash34uf(seed: vec3<u32>) -> vec4<f32> {
    return uf4(pcgChain4(fold3(seed)));
}

// ---------------------------------------------------------------------------
// 4 inputs
// ---------------------------------------------------------------------------
fn hash41ff(seed: vec4<f32>) -> f32 {
	return hash41uf((bitcast<vec4<u32>>(seed)));
}

fn hash41uf(seed: vec4<u32>) -> f32 {
    return uf1(pcgHash(fold4(seed)));
}

fn hash42ff(seed: vec4<f32>) -> vec2<f32> {
	return hash42uf((bitcast<vec4<u32>>(seed)));
}

fn hash42uf(seed: vec4<u32>) -> vec2<f32> {
    return uf2(pcgChain2(fold4(seed)));
}

fn hash43ff(seed: vec4<f32>) -> vec3<f32> {
	return hash43uf((bitcast<vec4<u32>>(seed)));
}

fn hash43uf(seed: vec4<u32>) -> vec3<f32> {
    return uf3(pcgChain3(fold4(seed)));
}

fn hash44ff(seed: vec4<f32>) -> vec4<f32> {
	return hash44uf((bitcast<vec4<u32>>(seed)));
}

fn hash44uf(seed: vec4<u32>) -> vec4<f32> {
    return uf4(pcgChain4(fold4(seed)));
}
