//Octahedral normal encoding: maps a unit vector to/from the [-1,1] square.
//Both channels use their full range, so an rg16float target keeps ~11 bits per
//axis (well under a tenth of a degree of error) at half the memory of rgba16float.
//For an 8 bit unorm target remap with * .5 + .5 / * 2. - 1. around these.
fn signNotZero(v:vec2<f32>) -> vec2<f32> {
    return vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}

fn encodeNormal(n:vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    if (n.z <= 0.0) {
        return (1.0 - abs(p.yx)) * signNotZero(p);
    }
    return p;
}

fn decodeNormal(e:vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let xy = (1.0 - abs(n.yx)) * signNotZero(n.xy);
        n = vec3<f32>(xy, n.z);
    }
    return normalize(n);
}
