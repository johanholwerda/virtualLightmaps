export const EPSILON = 1e-9;

export function posMod(x: number, m: number): number {
  return ((x % m) + m) % m;
}

export function fract(x: number): number {
  return x - Math.floor(x);
}

export function lerp(a: number, b: number, i: number): number {
  return (1 - i) * a + i * b;
}

//exponential interpolation. Use for camera zooms etc
export function eerp(a: number, b: number, i: number): number {
  return a * Math.pow(b / a, i);
}

export function linearstep(a: number, b: number, i: number): number {
  return clamp01((i - a) / (b - a));
}

export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// use for smoothstep or inverse smoothstep with controllable exponent
// https://www.shadertoy.com/view/ldBfR1
export function gain01(x: number, P: number) {
  if (x > 0.5) {
    return 1.0 - 0.5 * Math.pow(2.0 - 2.0 * x, P);
  } else {
    return 0.5 * Math.pow(2.0 * x, P);
  }
}

export function smootherstep01(x: number): number {
  return x * x * x * (x * (x * 6 - 15) + 10);
}

export function smoothstep01(x: number): number {
  x = clamp01(x);
  return x * x * (3 - 2 * x);
}

export function invSmoothstep01(x: number): number {
  x = clamp01(x);
  return 0.5 - Math.sin(Math.asin(1.0 - 2.0 * x) / 3.0);
}

export function smoothstep(e0: number, e1: number, x: number): number {
  x = clamp01((x - e0) / (e1 - e0));
  return x * x * (3 - 2 * x);
}

export function smootherstep(e0: number, e1: number, x: number): number {
  x = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return smootherstep01(x);
}

export function nearestPowerOfTwo(x: number): number {
  return Math.pow(2, Math.round(Math.log2(x)));
}

export function nextPowerOfTwo(x: number): number {
  return Math.pow(2, Math.ceil(Math.log2(x)));
}

export function isPowerOfTwo(x: number): boolean {
  return (x & (x - 1)) === 0;
}

export function isNumber(n: any) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

export function isVector3(n: any): boolean {
  return (n && isNumber(n.x) && isNumber(n.y) && isNumber(n.z));
}

export function smallerPowerOfTwo(x: number): number {
  return Math.pow(2, Math.floor(Math.log2(x)));
}

export function clamp(val: number, min: number, max: number) {
  return Math.min(Math.max(val, min), max);
}

//a factor something is multiplied by every frame (like a damping of 0.95), tuned at 60 fps, for a frame of dt seconds:
//the same as before at 60 fps, and the same change per second at any other frame rate
export function dampingFactor(factorAt60: number, dt: number): number {
  return Math.pow(factorAt60, dt * 60);
}

//the amount of a lerp towards a target every frame (like lerp(a, target, 0.1)), tuned at 60 fps, for a frame of dt
//seconds: it closes the same part of the distance per second at any frame rate
export function lerpFactor(amountAt60: number, dt: number): number {
  return 1 - Math.pow(1 - amountAt60, dt * 60);
}

export function degToRad(d: number): number {
  return d * (Math.PI / 180);
}

export function radToDeg(r: number): number {
  return r * (180 / Math.PI);
}

export function approximately(a: number, b: number): Boolean {
  return Math.abs(a - b) < 0.000001;
}

export function haltonSequence(index: number, base: number): number {
  let result = 0;
  let f = 1;
  let i = index;
  while (i > 0) {
    f = f / base;
    result = result + f * (i % base);
    i = Math.floor(i / base);
  }
  return result;
}