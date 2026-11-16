struct VertexInput {
	@location(0) position : vec3<f32>,
	//the part of the merged mesh, see PivotRenderer
	@location(1) aIndex : f32
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(1) color: vec3<f32>,
}

struct Uniforms {
	viewProjection : mat4x4<f32>,
	rotation : mat4x4<f32>,
	position: vec3<f32>,
	scale : f32,
	//0 hides the parts of an axis
	axisMask: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output: VertexOutput;

	let part = u32(input.aIndex + .5);
	let axis = part % 3u;
	if(part < 6u && uniforms.axisMask[axis] == 0.) {
		//outside the clip volume, the triangle is dropped
		output.position = vec4(2., 2., 2., 1.);
		return output;
	}

	let world = uniforms.position + (uniforms.rotation * vec4(input.position, 0.)).xyz * uniforms.scale;
	output.position = uniforms.viewProjection * vec4(world, 1.);

	//red, green, blue by axis, the center cube yellow
	output.color = select(vec3(f32(axis == 0u), f32(axis == 1u), f32(axis == 2u)), vec3(1., 1., 0.), part > 5u);
	return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	return vec4(input.color, 1.);
}
