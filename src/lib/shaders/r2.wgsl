//The R2 sequence, a 2d low discrepancy sequence: x(i) = fract(x(0) + i * alpha), alpha = (1 / g, 1 / g^2), g the plastic
//number (the real root of x^3 = x + 1). Martin Roberts 2018, "The Unreasonable Effectiveness of Quasirandom Sequences".
//Progressive: any prefix of the sequence covers the square evenly. It is a lattice though, the points lie on a few families
//of parallel lines, which shows as bands after a polar mapping (see blueNoiseHemisphere)

//alpha in 0.32 fixed point: round(alpha * 2^32)
const R2_FIXED = vec2<u32>(3242174889u, 2447445414u);

//Why integers: i * R2_FIXED wraps around at 2^32, which is exactly the fract, so every index gets the full 32 bits of
//fraction, up to i = 2^32 (the period). With floats, i * alpha has to fit the integer part and the fraction in the 24 bits
//of an f32 mantissa: at i = 2^20 (a million samples: almost 5 hours of frames at 60 fps) only 4 bits of fraction are left and
//the points snap to a 16 x 16 grid, from i = 2^24 on the fraction is gone and every index returns the same point.
//It is also cheaper: no fract.
//The top 24 bits only, as an f32 has 24 bits of mantissa: f32 of the full u32 rounds values close to 2^32 up to 2^32,
//which would return 1, outside of [0, 1)
fn getR2(i: u32) -> vec2<f32> {
	return vec2<f32>((i * R2_FIXED) >> vec2(8u)) * (1. / 16777216.);
}

//the float version, for comparison: loses its precision as i grows, see above
const R2_ALPHA = vec2<f32>(0.7548776662466927, 0.5698402909980532);
fn getR2Float(i: f32) -> vec2<f32> {
	return fract(i * R2_ALPHA);
}
