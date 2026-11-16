//the camera ray and the world position of a gBuffer pixel. Needs uniforms.inverseProjection, inverseView and cameraPosition.
//uv: -1 to 1, y down like the texture coordinates
fn getRay(uv: vec2<f32>)->vec3<f32>
{
    var dirEye = uniforms.inverseProjection * vec4(uv.x, -uv.y, 1., 1.0);
    dirEye.w = 0.;
    return normalize((uniforms.inverseView * dirEye).xyz);
}

//d: the distance along the camera ray, see packNormalDepth in gBuffer.wgsl
fn depthToPosition(d: f32, uv: vec2<f32>) -> vec3<f32>{
	return uniforms.cameraPosition + getRay(uv) * d;
}
