import Vector3 from './Vector3';
import Quaternion from './Quaternion';
import Matrix3x3 from './Matrix3x3';
import { EPSILON } from '../utils/MathUtils';

export type Matrix4x4Type = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

// read-only inputs for the helpers below; never handed out
const AXIS_X = new Vector3(1, 0, 0);
const AXIS_Y = new Vector3(0, 1, 0);
const AXIS_Z = new Vector3(0, 0, 1);
const ORIGIN = new Vector3(0, 0, 0);
const UNIT_SCALE = new Vector3(1, 1, 1);
const lookAtX = new Vector3();
const lookAtZ = new Vector3();

export default class Matrix4x4 {
  private static tempRotation = new Matrix4x4();
  private static tempMat3 = new Matrix3x3();

  public m: Matrix4x4Type = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

  constructor(...v: number[]) {
    if (v.length === 16) {
      this.m = [...v] as Matrix4x4Type;
    }
  }

  public static clone(a: Matrix4x4) {
    return Matrix4x4.copy(a, new Matrix4x4());
  }

  // Writes into dest.m rather than replacing it, so references taken by the
  // `const out = outm.m` idiom used throughout this class stay valid.
  public static copy(src: Matrix4x4, dest: Matrix4x4) {
    const out = dest.m,
          a   = src.m;
    for (let i = 0; i < 16; i++) out[i] = a[i];
    return dest;
  }

  public static identity(outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    return outm.setValues(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    );
  }

  public static transpose(am: Matrix4x4, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m;
    // If we are transposing ourselves we can skip a few steps but have to cache some values
    if (out === a) {
      const a01 = a[1],
            a02 = a[2],
            a03 = a[3];
      const a12 = a[6],
            a13 = a[7];
      const a23 = a[11];

      out[1] = a[4];
      out[2] = a[8];
      out[3] = a[12];
      out[4] = a01;
      out[6] = a[9];
      out[7] = a[13];
      out[8] = a02;
      out[9] = a12;
      out[11] = a[14];
      out[12] = a03;
      out[13] = a13;
      out[14] = a23;
    } else {
      out[0] = a[0];
      out[1] = a[4];
      out[2] = a[8];
      out[3] = a[12];
      out[4] = a[1];
      out[5] = a[5];
      out[6] = a[9];
      out[7] = a[13];
      out[8] = a[2];
      out[9] = a[6];
      out[10] = a[10];
      out[11] = a[14];
      out[12] = a[3];
      out[13] = a[7];
      out[14] = a[11];
      out[15] = a[15];
    }

    return outm;
  }

  public static invert(am: Matrix4x4, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m;
    const a00 = a[0],
          a01 = a[1],
          a02 = a[2],
          a03 = a[3];
    const a10 = a[4],
          a11 = a[5],
          a12 = a[6],
          a13 = a[7];
    const a20 = a[8],
          a21 = a[9],
          a22 = a[10],
          a23 = a[11];
    const a30 = a[12],
          a31 = a[13],
          a32 = a[14],
          a33 = a[15];

    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;

    // Calculate the determinant
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    // singular: identity rather than leaving whatever outm held before
    if (!det) {
      return Matrix4x4.identity(outm);
    }
    det = 1.0 / det;

    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;

    return outm;
  }

  public static determinant(am: Matrix4x4) {
    const a = am.m;
    const a00 = a[0],
          a01 = a[1],
          a02 = a[2],
          a03 = a[3];
    const a10 = a[4],
          a11 = a[5],
          a12 = a[6],
          a13 = a[7];
    const a20 = a[8],
          a21 = a[9],
          a22 = a[10],
          a23 = a[11];
    const a30 = a[12],
          a31 = a[13],
          a32 = a[14],
          a33 = a[15];

    const b0 = a00 * a11 - a01 * a10;
    const b1 = a00 * a12 - a02 * a10;
    const b2 = a01 * a12 - a02 * a11;
    const b3 = a20 * a31 - a21 * a30;
    const b4 = a20 * a32 - a22 * a30;
    const b5 = a21 * a32 - a22 * a31;
    const b6 = a00 * b5 - a01 * b4 + a02 * b3;
    const b7 = a10 * b5 - a11 * b4 + a12 * b3;
    const b8 = a20 * b2 - a21 * b1 + a22 * b0;
    const b9 = a30 * b2 - a31 * b1 + a32 * b0;

    // Calculate the determinant
    return a13 * b6 - a03 * b7 + a33 * b8 - a23 * b9;
  }

  public static multiply(am: Matrix4x4, bm: Matrix4x4, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m,
          b   = bm.m;
    const a00 = a[0],
          a01 = a[1],
          a02 = a[2],
          a03 = a[3];
    const a10 = a[4],
          a11 = a[5],
          a12 = a[6],
          a13 = a[7];
    const a20 = a[8],
          a21 = a[9],
          a22 = a[10],
          a23 = a[11];
    const a30 = a[12],
          a31 = a[13],
          a32 = a[14],
          a33 = a[15];

    // Cache only the current column of the second matrix
    for (let i = 0; i < 16; i += 4) {
      const b0 = b[i],
            b1 = b[i + 1],
            b2 = b[i + 2],
            b3 = b[i + 3];
      out[i] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
      out[i + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
      out[i + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
      out[i + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    }
    return outm;
  }

  public static translate(am: Matrix4x4, v: Vector3, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m;
    const x = v.x,
          y = v.y,
          z = v.z;

    if (a !== out) {
      for (let i = 0; i < 12; i++) out[i] = a[i];
    }
    // each element only reads its own old value from the last column, so this is safe in place
    for (let i = 0; i < 4; i++) {
      out[12 + i] = a[i] * x + a[4 + i] * y + a[8 + i] * z + a[12 + i];
    }
    return outm;
  }

  public static scale(am: Matrix4x4, v: Vector3, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m;
    const x = v.x,
          y = v.y,
          z = v.z;

    for (let i = 0; i < 4; i++) {
      out[i] = a[i] * x;
      out[4 + i] = a[4 + i] * y;
      out[8 + i] = a[8 + i] * z;
      out[12 + i] = a[12 + i];
    }
    return outm;
  }

  public static rotate(am: Matrix4x4, rad: number, axis: Vector3, outm: Matrix4x4 | undefined = undefined) {
    // fromRotation gives identity for a zero axis, so the result is then am unchanged
    return Matrix4x4.multiply(am, Matrix4x4.fromRotation(rad, axis, Matrix4x4.tempRotation), outm);
  }

  public static rotateX(am: Matrix4x4, rad: number, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m;
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    const a10 = a[4];
    const a11 = a[5];
    const a12 = a[6];
    const a13 = a[7];
    const a20 = a[8];
    const a21 = a[9];
    const a22 = a[10];
    const a23 = a[11];

    if (a !== out) {
      // If the source and destination differ, copy the unchanged rows
      out[0] = a[0];
      out[1] = a[1];
      out[2] = a[2];
      out[3] = a[3];
      out[12] = a[12];
      out[13] = a[13];
      out[14] = a[14];
      out[15] = a[15];
    }

    // Perform axis-specific matrix multiplication
    out[4] = a10 * c + a20 * s;
    out[5] = a11 * c + a21 * s;
    out[6] = a12 * c + a22 * s;
    out[7] = a13 * c + a23 * s;
    out[8] = a20 * c - a10 * s;
    out[9] = a21 * c - a11 * s;
    out[10] = a22 * c - a12 * s;
    out[11] = a23 * c - a13 * s;
    return outm;
  }

  public static rotateY(am: Matrix4x4, rad: number, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m;
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    const a00 = a[0];
    const a01 = a[1];
    const a02 = a[2];
    const a03 = a[3];
    const a20 = a[8];
    const a21 = a[9];
    const a22 = a[10];
    const a23 = a[11];

    if (a !== out) {
      // If the source and destination differ, copy the unchanged rows
      out[4] = a[4];
      out[5] = a[5];
      out[6] = a[6];
      out[7] = a[7];
      out[12] = a[12];
      out[13] = a[13];
      out[14] = a[14];
      out[15] = a[15];
    }

    // Perform axis-specific matrix multiplication
    out[0] = a00 * c - a20 * s;
    out[1] = a01 * c - a21 * s;
    out[2] = a02 * c - a22 * s;
    out[3] = a03 * c - a23 * s;
    out[8] = a00 * s + a20 * c;
    out[9] = a01 * s + a21 * c;
    out[10] = a02 * s + a22 * c;
    out[11] = a03 * s + a23 * c;
    return outm;
  }

  public static rotateZ(am: Matrix4x4, rad: number, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m;
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    const a00 = a[0];
    const a01 = a[1];
    const a02 = a[2];
    const a03 = a[3];
    const a10 = a[4];
    const a11 = a[5];
    const a12 = a[6];
    const a13 = a[7];

    if (a !== out) {
      // If the source and destination differ, copy the unchanged rows
      out[8] = a[8];
      out[9] = a[9];
      out[10] = a[10];
      out[11] = a[11];
      out[12] = a[12];
      out[13] = a[13];
      out[14] = a[14];
      out[15] = a[15];
    }

    // Perform axis-specific matrix multiplication
    out[0] = a00 * c + a10 * s;
    out[1] = a01 * c + a11 * s;
    out[2] = a02 * c + a12 * s;
    out[3] = a03 * c + a13 * s;
    out[4] = a10 * c - a00 * s;
    out[5] = a11 * c - a01 * s;
    out[6] = a12 * c - a02 * s;
    out[7] = a13 * c - a03 * s;
    return outm;
  }

  public static fromTranslation(v: Vector3, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    return outm.setValues(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      v.x, v.y, v.z, 1,
    );
  }

  public static fromRotation(rad: number, axis: Vector3, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    let len = axis.length();

    if (len < EPSILON) {
      return Matrix4x4.identity(outm);
    }

    len = 1 / len;
    const x = axis.x * len,
          y = axis.y * len,
          z = axis.z * len;

    const s = Math.sin(rad);
    const c = Math.cos(rad);
    const t = 1 - c;

    return outm.setValues(
      x * x * t + c, y * x * t + z * s, z * x * t - y * s, 0,
      x * y * t - z * s, y * y * t + c, z * y * t + x * s, 0,
      x * z * t + y * s, y * z * t - x * s, z * z * t + c, 0,
      0, 0, 0, 1,
    );
  }

  public static fromXRotation(rad: number, outm: Matrix4x4 | undefined = undefined) {
    return Matrix4x4.fromRotation(rad, AXIS_X, outm);
  }

  public static fromYRotation(rad: number, outm: Matrix4x4 | undefined = undefined) {
    return Matrix4x4.fromRotation(rad, AXIS_Y, outm);
  }

  public static fromZRotation(rad: number, outm: Matrix4x4 | undefined = undefined) {
    return Matrix4x4.fromRotation(rad, AXIS_Z, outm);
  }

  public static fromRotationTranslationScale(q: Quaternion, v: Vector3, s: Vector3, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    // Quaternion math
    const x = q.x,
          y = q.y,
          z = q.z,
          w = q.w;
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;

    const xx = x * x2;
    const xy = x * y2;
    const xz = x * z2;
    const yy = y * y2;
    const yz = y * z2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    const sx = s.x;
    const sy = s.y;
    const sz = s.z;

    return outm.setValues(
      (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
      (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
      (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
      v.x, v.y, v.z, 1,
    );
  }

  public static toTranslationRotationScale(
    m: Matrix4x4,
    outQ: Quaternion | undefined = undefined,
    outV: Vector3 | undefined = undefined,
    outS: Vector3 | undefined = undefined,
  ): { rotation: Quaternion; position: Vector3; scale: Vector3 } {
    outQ = outQ || new Quaternion();
    outV = outV || new Vector3();
    outS = outS || new Vector3();

    const mat = m.m;

    // Extract translation
    outV.x = mat[12];
    outV.y = mat[13];
    outV.z = mat[14];

    // Extract scale
    const sx = Math.sqrt(mat[0] * mat[0] + mat[1] * mat[1] + mat[2] * mat[2]);
    const sy = Math.sqrt(mat[4] * mat[4] + mat[5] * mat[5] + mat[6] * mat[6]);
    const sz = Math.sqrt(mat[8] * mat[8] + mat[9] * mat[9] + mat[10] * mat[10]);

    // Handle negative scale (check determinant)
    const det = mat[0] * (mat[5] * mat[10] - mat[6] * mat[9]) -
                mat[1] * (mat[4] * mat[10] - mat[6] * mat[8]) +
                mat[2] * (mat[4] * mat[9] - mat[5] * mat[8]);

    outS.x = det < 0 ? -sx : sx;
    outS.y = sy;
    outS.z = sz;

    // Remove scale to get the rotation axes, the columns of a 3x3 rotation matrix (column-major, like Matrix4x4).
    // An axis with a scale of 0 has no direction (dividing by it gives NaN): it is left 0 and filled in below.
    const rotMat = Matrix4x4.tempMat3;
    const r = rotMat.m;
    let zeroAxes = 0,
        zeroAxis = -1;
    for (let i = 0; i < 3; i++) {
      const s = i === 0 ? outS.x : i === 1 ? sy : sz;
      if (s === 0) {
        zeroAxes++;
        zeroAxis = i;
      }
      const inv = s === 0 ? 0 : 1 / s;
      r[i * 3] = mat[i * 4] * inv;
      r[i * 3 + 1] = mat[i * 4 + 1] * inv;
      r[i * 3 + 2] = mat[i * 4 + 2] * inv;
    }

    if (zeroAxes > 1) {
      // the rotation can't be recovered
      outQ.identity();
    } else {
      if (zeroAxes === 1) {
        // one missing axis follows from the other two: x = y × z, y = z × x, z = x × y
        const i = zeroAxis * 3,
              j = ((zeroAxis + 1) % 3) * 3,
              k = ((zeroAxis + 2) % 3) * 3;
        r[i] = r[j + 1] * r[k + 2] - r[j + 2] * r[k + 1];
        r[i + 1] = r[j + 2] * r[k] - r[j] * r[k + 2];
        r[i + 2] = r[j] * r[k + 1] - r[j + 1] * r[k];
      }

      outQ.fromMat3(rotMat);
    }

    return { position: outV, rotation: outQ, scale: outS };
  }

  public static fromQuat(q: Quaternion, outm: Matrix4x4 | undefined = undefined) {
    return Matrix4x4.fromRotationTranslationScale(q, ORIGIN, UNIT_SCALE, outm);
  }

  /** Maps the view frustum to WebGPU clip space: x, y in [-1,1] and z in [0,1]. */
  public static perspective(fovy: number, aspect: number, near: number, far: number, outm: Matrix4x4 | undefined = undefined) {
    const top = near * Math.tan(fovy / 2);
    const right = top * aspect;
    return Matrix4x4.frustum(-right, right, -top, top, near, far, outm);
  }

  /** A perspective projection of the view frustum, which can be off-center, to WebGPU clip space: z in [0,1]. far can be Infinity */
  public static frustum(left: number, right: number, bottom: number, top: number, near: number, far: number, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const rl = 1 / (right - left);
    const tb = 1 / (top - bottom);
    let m22 = -1,
        m32 = -near;
    if (far != null && far !== Infinity) {
      const nf = 1 / (near - far);
      m22 = far * nf;
      m32 = far * near * nf;
    }
    return outm.setValues(
      2 * near * rl, 0, 0, 0,
      0, 2 * near * tb, 0, 0,
      (right + left) * rl, (top + bottom) * tb, m22, -1,
      0, 0, m32, 0,
    );
  }

  /** Maps the view box to WebGPU clip space: x, y in [-1,1] and z in [0,1]. */
  public static ortho(left: number, right: number, bottom: number, top: number, near: number, far: number, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const lr = 1 / (left - right);
    const bt = 1 / (bottom - top);
    const nf = 1 / (near - far);
    return outm.setValues(
      -2 * lr, 0, 0, 0,
      0, -2 * bt, 0, 0,
      0, 0, nf, 0,
      (left + right) * lr, (top + bottom) * bt, near * nf, 1,
    );
  }

  public static lookAt(eye: Vector3, center: Vector3, up: Vector3, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const eyex = eye.x,
          eyey = eye.y,
          eyez = eye.z;
    const centerx = center.x,
          centery = center.y,
          centerz = center.z;

    if (
      Math.abs(eyex - centerx) < EPSILON &&
      Math.abs(eyey - centery) < EPSILON &&
      Math.abs(eyez - centerz) < EPSILON
    ) {
      return Matrix4x4.identity(outm);
    }

    // a zero vector stays zero, rather than becoming NaN
    const invLength = (x: number, y: number, z: number) => {
      const len = Math.hypot(x, y, z);
      return len ? 1 / len : 0;
    };

    let z0 = eyex - centerx,
        z1 = eyey - centery,
        z2 = eyez - centerz;
    let len = invLength(z0, z1, z2);
    z0 *= len;
    z1 *= len;
    z2 *= len;

    //falls back to another up when up is parallel to the view direction
    const x = Vector3.lookAtRight(up, lookAtZ.setValues(z0, z1, z2), lookAtX);
    const x0 = x.x,
          x1 = x.y,
          x2 = x.z;

    let y0 = z1 * x2 - z2 * x1,
        y1 = z2 * x0 - z0 * x2,
        y2 = z0 * x1 - z1 * x0;
    len = invLength(y0, y1, y2);
    y0 *= len;
    y1 *= len;
    y2 *= len;

    return outm.setValues(
      x0, y0, z0, 0,
      x1, y1, z1, 0,
      x2, y2, z2, 0,
      -(x0 * eyex + x1 * eyey + x2 * eyez),
      -(y0 * eyex + y1 * eyey + y2 * eyez),
      -(z0 * eyex + z1 * eyey + z2 * eyez),
      1,
    );
  }

  public static add(am: Matrix4x4, bm: Matrix4x4, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m,
          b   = bm.m;
    for (let i = 0; i < 16; i++) out[i] = a[i] + b[i];
    return outm;
  }

  public static subtract(am: Matrix4x4, bm: Matrix4x4, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m,
          b   = bm.m;
    for (let i = 0; i < 16; i++) out[i] = a[i] - b[i];
    return outm;
  }

  public static multiplyScalar(am: Matrix4x4, b: number, outm: Matrix4x4 | undefined = undefined) {
    outm = outm || new Matrix4x4();
    const out = outm.m,
          a   = am.m;
    for (let i = 0; i < 16; i++) out[i] = a[i] * b;
    return outm;
  }

  public setValues(
    m00: number, m01: number, m02: number, m03: number,
    m10: number, m11: number, m12: number, m13: number,
    m20: number, m21: number, m22: number, m23: number,
    m30: number, m31: number, m32: number, m33: number,
  ) {
    const out = this.m;
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m03;
    out[4] = m10;
    out[5] = m11;
    out[6] = m12;
    out[7] = m13;
    out[8] = m20;
    out[9] = m21;
    out[10] = m22;
    out[11] = m23;
    out[12] = m30;
    out[13] = m31;
    out[14] = m32;
    out[15] = m33;
    return this;
  }

  public clone() {
    return Matrix4x4.clone(this);
  }

  public copy(a: Matrix4x4) {
    return Matrix4x4.copy(a, this);
  }

  public get inverse(): Matrix4x4 {
    return Matrix4x4.invert(this);
  }
}
