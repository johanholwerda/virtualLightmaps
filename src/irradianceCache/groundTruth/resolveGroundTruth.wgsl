struct Uniforms {
	resolution : vec2<f32>,
	exposure : f32,
}

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
//per pixel: the sum of the color (rgb) and the frame count (a), see accumulateGroundTruth.wgsl
@group(0) @binding(1) var<storage, read> accumulationBuffer : array<vec4<f32>>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.0);
	output.uv = input.uv;
	return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	let coord = vec2<u32>(input.uv * uniforms.resolution);

	//the color with the albedo and the sky, averaged over the jittered frames, see accumulateGroundTruth.wgsl
	let accumulated = accumulationBuffer[coord.y * u32(uniforms.resolution.x) + coord.x];
	let color = accumulated.rgb / max(accumulated.a, 1.);
	return vec4(toneMap(color * uniforms.exposure), 1.);
}
