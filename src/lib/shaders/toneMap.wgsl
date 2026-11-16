//https://www.shadertoy.com/view/XsGfWV
fn acesTonemap(color: vec3<f32>)->vec3<f32>{
	let m1 = mat3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
	);
	let m2 = mat3x3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
	);
	let v = m1 * color;
	let a = v * (v + 0.0245786) - 0.000090537;
	let b = v * (0.983729 * v + 0.4329510) + 0.238081;
	return saturate(m2 * (a / b));
}

//curve fitted approximation. Over saturates bright colors
//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn acesTonemapFitted(color: vec3<f32>) -> vec3<f32> {
	let a = 2.51;
	let b = 0.03;
	let c = 2.43;
	let d = 0.59;
	let e = 0.14;
	return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

//unity gamma to linear
fn gammaToLinear (sRGB: vec3<f32>)->vec3<f32> {
	// Approximate version from http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html?m=1
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}

fn linearToGamma (linRGB: vec3<f32>)->vec3<f32> {
	// An almost-perfect approximation from http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html?m=1
	return max(1.055 * pow(linRGB, vec3(0.416666667)) - 0.055, vec3(0));
}

fn toneMap(color: vec3<f32>)->vec3<f32>{
	let c = acesTonemap(color);
	//color = mix(c, smoothstep(0., 1., c), _Contrast);
	return linearToGamma(c);
}
