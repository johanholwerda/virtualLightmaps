import type Vector2 from './Vector2';

export type Matrix2x2Type = [number, number, number, number];

export default class Matrix2x2 {
  public m: Matrix2x2Type = [1, 0, 0, 1];

  constructor(...v: number[]) {
    if (v.length === 4) {
      this.m = [...v] as Matrix2x2Type;
    }
  }

  public static clone(a: Matrix2x2) {
    return Matrix2x2.copy(a, new Matrix2x2());
  }

  public static copy(src: Matrix2x2, dest: Matrix2x2) {
    const out = dest.m,
          a   = src.m;
    for (let i = 0; i < 4; i++) out[i] = a[i];
    return dest;
  }

  public static identity(outm: Matrix2x2) {
    const out = outm.m;
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 1;
    return outm;
  }

  public static transpose(am: Matrix2x2, outm: Matrix2x2 | undefined = undefined) {
    outm = outm || new Matrix2x2();
    const out = outm.m,
          a   = am.m;
    // If we are transposing ourselves we can skip a few steps but have to cache
    // some values
    if (out === a) {
      const a1 = a[1];
      out[1] = a[2];
      out[2] = a1;
    } else {
      out[0] = a[0];
      out[1] = a[2];
      out[2] = a[1];
      out[3] = a[3];
    }

    return outm;
  }

  public static invert(am: Matrix2x2, outm: Matrix2x2 | undefined = undefined) {
    outm = outm || new Matrix2x2();
    const out = outm.m,
          a   = am.m;
    const a0 = a[0],
          a1 = a[1],
          a2 = a[2],
          a3 = a[3];

    // Calculate the determinant
    let det = a0 * a3 - a2 * a1;

    // singular: identity rather than leaving whatever outm held before
    if (!det) {
      return Matrix2x2.identity(outm);
    }
    det = 1.0 / det;

    out[0] = a3 * det;
    out[1] = -a1 * det;
    out[2] = -a2 * det;
    out[3] = a0 * det;

    return outm;
  }

  public static determinant(am: Matrix2x2) {
    const a = am.m;
    return a[0] * a[3] - a[2] * a[1];
  }

  public static multiply(am: Matrix2x2, bm: Matrix2x2, outm: Matrix2x2 | undefined = undefined) {
    outm = outm || new Matrix2x2();
    const out = outm.m,
          a   = am.m,
          b   = bm.m;
    const a0 = a[0],
          a1 = a[1],
          a2 = a[2],
          a3 = a[3];
    const b0 = b[0],
          b1 = b[1],
          b2 = b[2],
          b3 = b[3];
    out[0] = a0 * b0 + a2 * b1;
    out[1] = a1 * b0 + a3 * b1;
    out[2] = a0 * b2 + a2 * b3;
    out[3] = a1 * b2 + a3 * b3;
    return outm;
  }

  public static rotate(am: Matrix2x2, rad: number, outm: Matrix2x2 | undefined = undefined) {
    outm = outm || new Matrix2x2();
    const out = outm.m,
          a   = am.m;
    const a0 = a[0],
          a1 = a[1],
          a2 = a[2],
          a3 = a[3];
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    out[0] = a0 * c + a2 * s;
    out[1] = a1 * c + a3 * s;
    out[2] = a0 * -s + a2 * c;
    out[3] = a1 * -s + a3 * c;
    return outm;
  }

  public static scale(am: Matrix2x2, v: Vector2, outm: Matrix2x2 | undefined = undefined) {
    outm = outm || new Matrix2x2();
    const out = outm.m,
          a   = am.m;
    const a0 = a[0],
          a1 = a[1],
          a2 = a[2],
          a3 = a[3];
    const v0 = v.x,
          v1 = v.y;
    out[0] = a0 * v0;
    out[1] = a1 * v0;
    out[2] = a2 * v1;
    out[3] = a3 * v1;
    return outm;
  }

  public static fromRotation(rad: number, outm: Matrix2x2 | undefined = undefined) {
    outm = outm || new Matrix2x2();
    const out = outm.m;
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    out[0] = c;
    out[1] = s;
    out[2] = -s;
    out[3] = c;
    return outm;
  }

  public static fromScaling(v: Vector2, outm: Matrix2x2 | undefined = undefined) {
    outm = outm || new Matrix2x2();
    const out = outm.m;
    out[0] = v.x;
    out[1] = 0;
    out[2] = 0;
    out[3] = v.y;
    return outm;
  }

  public clone() {
    return Matrix2x2.clone(this);
  }
}
