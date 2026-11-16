import Vector4 from './Vector4';
import Vector3 from './Vector3';

export default class Color extends Vector4 {
  public static get ONE(): Color {
    return new Color(1, 1, 1, 1);
  }

  public static get ZERO(): Color {
    return new Color(0, 0, 0, 0);
  }

  public static random(): Color {
    return new Color(Math.random(), Math.random(), Math.random());
  }

  get r(): number {
    return this.x;
  }

  set r(value: number) {
    this.x = value;
  }

  get g(): number {
    return this.y;
  }

  set g(value: number) {
    this.y = value;
  }

  get b(): number {
    return this.z;
  }

  set b(value: number) {
    this.z = value;
  }

  get a(): number {
    return this.w;
  }

  set a(value: number) {
    this.w = value;
  }

  public get vector3(): Vector3 {
    return new Vector3(this.x, this.y, this.z);
  }

  // Alpha is linear coverage, so both conversions pass it through untouched.

  public get gammaToLinearSpace(): Color {
    // Use Unity conversion

    // Approximate version from http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html?m=1
    // return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);

    const convert = (c: number) => c * (c * (c * 0.305306011 + 0.682171111) + 0.012522878);
    return new Color(convert(this.r), convert(this.g), convert(this.b), this.a);
  }

  public get linearToGammaSpace(): Color {
    // Use Unity conversion

    // linRGB = max(linRGB, vec3(0, 0, 0));
    // An almost-perfect approximation from http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html?m=1
    // return max(1.055 * pow(linRGB, vec3(0.416666667)) - 0.055, vec3(0));

    const convert = (c: number) => Math.max(1.055 * Math.pow(Math.max(c, 0), 0.416666667) - 0.055, 0);
    return new Color(convert(this.r), convert(this.g), convert(this.b), this.a);
  }

  public static fromHex(hex: string): Color {
    return new Color().setHex(hex);
  }

  public static fromRGB255(value: [number,number,number]): Color {
    return new Color().setRGBA255([...value, 255]);
  }

  public static fromHSV(h: number, s: number, v: number): Color {
    return new Color().setHSV(h, s, v);
  }

  public clone(): Color {
    return new Color(this.x, this.y, this.z, this.w);
  }

  public setRGBA255(value: [number, number, number, number]): Color {
    this.r = value[0] / 255;
    this.g = value[1] / 255;
    this.b = value[2] / 255;
    this.a = value[3] / 255;
    return this;
  }

  // accepts rgb, rgba, rrggbb and rrggbbaa, with or without the #
  public setHex(hex: string): Color {
    hex = hex.replace(/[^0-9A-F]/gi, '');
    if (hex.length === 3 || hex.length === 4) hex = hex.replace(/./g, '$&$&');
    // per channel: parsing the whole string at once overflows the 32 bit shifts for rrggbbaa
    const channel = (i: number) => parseInt(hex.substring(i * 2, i * 2 + 2), 16) / 255;
    this.r = channel(0);
    this.g = channel(1);
    this.b = channel(2);
    this.a = hex.length === 8 ? channel(3) : 1;
    return this;
  }

  // #rrggbb, clamped to [0, 1] since HDR and negative values have no hex form
  public getHex() {
    const toHex = (c: number) => Math.round(Math.min(Math.max(c, 0), 1) * 255).toString(16).padStart(2, '0');
    return `#${toHex(this.r)}${toHex(this.g)}${toHex(this.b)}`;
  }

  // provide normalized hsv
  public setHSV(h: number, s: number, v: number): Color {
    let r = 0;
    let g = 0;
    let b = 0;
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    switch (i % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
    this.r = r;
    this.g = g;
    this.b = b;
    return this;
  }
}
