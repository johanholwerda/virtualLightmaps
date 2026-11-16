struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) uv : vec2<f32>
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

//the model target of the gBuffer, see packModelTriangle in includes/gBuffer.wgsl
@group(0) @binding(0) var texture: texture_2d<u32>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4<f32>(input.position, 1.);
	output.uv = input.uv;
	return output;
}

//pcg hash to a random color
fn hashColor(value: u32) -> vec3<f32> {
  var state = value * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  let h = (word >> 22u) ^ word;
  return vec3(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let model = unpackModelTriangle(textureLoad(texture, vec2<u32>(input.position.xy), 0).x);
  return vec4(hashColor(model.y + 1u), 1.);
}
