import type Quaternion from './Quaternion';
import type Matrix4x4 from './Matrix4x4';

export type Matrix3x3Type = [number, number, number, number, number, number, number, number, number];

export default class Matrix3x3 {
  public m: Matrix3x3Type = [1, 0, 0, 0, 1, 0, 0, 0, 1];

  constructor(...v: number[]) {
    if (v.length === 9) {
      this.m = [...v] as Matrix3x3Type;
    }
  }

  public static clone(a: Matrix3x3) {
    return Matrix3x3.copy(a, new Matrix3x3());
  }

  public static copy(src: Matrix3x3, dest: Matrix3x3) {
    const out = dest.m,
          a   = src.m;
    for (let i = 0; i < 9; i++) out[i] = a[i];
    return dest;
  }

  public static setValues(
    outm: Matrix3x3,
    m00: number,
    m01: number,
    m02: number,
    m10: number,
    m11: number,
    m12: number,
    m20: number,
    m21: number,
    m22: number,
  ) {
    const out = outm.m;
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m10;
    out[4] = m11;
    out[5] = m12;
    out[6] = m20;
    out[7] = m21;
    out[8] = m22;
    return outm;
  }

  public static identity(outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    return Matrix3x3.setValues(outm, 1, 0, 0, 0, 1, 0, 0, 0, 1);
  }

  public static transpose(am: Matrix3x3, outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    const out = outm.m,
          a   = am.m;
    // If we are transposing ourselves we can skip a few steps but have to cache some values
    if (out === a) {
      const a01 = a[1],
            a02 = a[2],
            a12 = a[5];
      out[1] = a[3];
      out[2] = a[6];
      out[3] = a01;
      out[5] = a[7];
      out[6] = a02;
      out[7] = a12;
    } else {
      out[0] = a[0];
      out[1] = a[3];
      out[2] = a[6];
      out[3] = a[1];
      out[4] = a[4];
      out[5] = a[7];
      out[6] = a[2];
      out[7] = a[5];
      out[8] = a[8];
    }

    return outm;
  }

  public static invert(am: Matrix3x3, outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    const out = outm.m,
          a   = am.m;
    const a00 = a[0],
          a01 = a[1],
          a02 = a[2];
    const a10 = a[3],
          a11 = a[4],
          a12 = a[5];
    const a20 = a[6],
          a21 = a[7],
          a22 = a[8];

    const b01 = a22 * a11 - a12 * a21;
    const b11 = -a22 * a10 + a12 * a20;
    const b21 = a21 * a10 - a11 * a20;

    // Calculate the determinant
    let det = a00 * b01 + a01 * b11 + a02 * b21;

    // singular: identity rather than leaving whatever outm held before
    if (!det) {
      return Matrix3x3.identity(outm);
    }
    det = 1.0 / det;

    out[0] = b01 * det;
    out[1] = (-a22 * a01 + a02 * a21) * det;
    out[2] = (a12 * a01 - a02 * a11) * det;
    out[3] = b11 * det;
    out[4] = (a22 * a00 - a02 * a20) * det;
    out[5] = (-a12 * a00 + a02 * a10) * det;
    out[6] = b21 * det;
    out[7] = (-a21 * a00 + a01 * a20) * det;
    out[8] = (a11 * a00 - a01 * a10) * det;
    return outm;
  }

  public static determinant(am: Matrix3x3) {
    const a = am.m;
    const a00 = a[0],
          a01 = a[1],
          a02 = a[2];
    const a10 = a[3],
          a11 = a[4],
          a12 = a[5];
    const a20 = a[6],
          a21 = a[7],
          a22 = a[8];

    return a00 * (a22 * a11 - a12 * a21) + a01 * (-a22 * a10 + a12 * a20) + a02 * (a21 * a10 - a11 * a20);
  }

  public static multiply(am: Matrix3x3, bm: Matrix3x3, outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    const out = outm.m,
          a   = am.m,
          b   = bm.m;
    const a00 = a[0],
          a01 = a[1],
          a02 = a[2];
    const a10 = a[3],
          a11 = a[4],
          a12 = a[5];
    const a20 = a[6],
          a21 = a[7],
          a22 = a[8];

    const b00 = b[0],
          b01 = b[1],
          b02 = b[2];
    const b10 = b[3],
          b11 = b[4],
          b12 = b[5];
    const b20 = b[6],
          b21 = b[7],
          b22 = b[8];

    out[0] = b00 * a00 + b01 * a10 + b02 * a20;
    out[1] = b00 * a01 + b01 * a11 + b02 * a21;
    out[2] = b00 * a02 + b01 * a12 + b02 * a22;

    out[3] = b10 * a00 + b11 * a10 + b12 * a20;
    out[4] = b10 * a01 + b11 * a11 + b12 * a21;
    out[5] = b10 * a02 + b11 * a12 + b12 * a22;

    out[6] = b20 * a00 + b21 * a10 + b22 * a20;
    out[7] = b20 * a01 + b21 * a11 + b22 * a21;
    out[8] = b20 * a02 + b21 * a12 + b22 * a22;
    return outm;
  }

  public static fromQuat(q: Quaternion, outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    const out = outm.m;
    const x = q.x,
          y = q.y,
          z = q.z,
          w = q.w;

    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;

    const xx = x * x2;
    const yx = y * x2;
    const yy = y * y2;
    const zx = z * x2;
    const zy = z * y2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;

    out[0] = 1 - yy - zz;
    out[3] = yx - wz;
    out[6] = zx + wy;

    out[1] = yx + wz;
    out[4] = 1 - xx - zz;
    out[7] = zy - wx;

    out[2] = zx - wy;
    out[5] = zy + wx;
    out[8] = 1 - xx - yy;

    return outm;
  }

  public static add(am: Matrix3x3, bm: Matrix3x3, outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    const out = outm.m,
          a   = am.m,
          b   = bm.m;
    for (let i = 0; i < 9; i++) out[i] = a[i] + b[i];
    return outm;
  }

  public static subtract(am: Matrix3x3, bm: Matrix3x3, outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    const out = outm.m,
          a   = am.m,
          b   = bm.m;
    for (let i = 0; i < 9; i++) out[i] = a[i] - b[i];
    return outm;
  }

  public static multiplyScalar(am: Matrix3x3, b: number, outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    const out = outm.m,
          a   = am.m;
    for (let i = 0; i < 9; i++) out[i] = a[i] * b;
    return outm;
  }

  //the inverse transpose of the rotation and scale of an affine matrix, to transform normals. identity when singular
  public static normalFromMat4(am: Matrix4x4, outm: Matrix3x3 | undefined = undefined) {
    outm = Matrix3x3.fromMat4(am, outm);
    return Matrix3x3.transpose(Matrix3x3.invert(outm, outm), outm);
  }

  public static fromMat4(am: Matrix4x4, outm: Matrix3x3 | undefined = undefined) {
    outm = outm || new Matrix3x3();
    const out = outm.m,
          a   = am.m;
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[4];
    out[4] = a[5];
    out[5] = a[6];
    out[6] = a[8];
    out[7] = a[9];
    out[8] = a[10];
    return outm;
  }

  public clone() {
    return Matrix3x3.clone(this);
  }
}
