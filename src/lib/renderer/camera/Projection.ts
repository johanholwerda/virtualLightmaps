import Vector3 from '../math/Vector3';
import Matrix4x4 from '../math/Matrix4x4';
import Vector2 from '../math/Vector2';
import {haltonSequence} from '../utils/MathUtils';

export default class Projection {
  public frustumHeight: number = 1;

  protected projectionMatrix = new Matrix4x4();
  protected fov: number;
  protected _near: number;
  protected _far: number;
  protected _aspectRatio: number;
  private _frustumOffset = new Vector2();
  private _jitter = new Vector2();
  private _dirty: boolean = true;
  private _version: number = 0;
  private jitterFrame = 0;

  constructor(fov: number = 1, nearPlane: number = 0.01, farPlane: number = 10) {
    this.fov = fov;
    this.frustumHeight = Math.tan(this.fov * 0.5) * 2;
    this._near = nearPlane;
    this._far = farPlane;
    this._aspectRatio = 1;

    this.dirty = true;
  }

  public copy(projection: Projection) {
    this.FOV = projection.fov;
    this._near = projection.near;
    this._far = projection.far;
    this._aspectRatio = projection.aspectRatio;
    this.dirty = true;
  }

  public get near(): number {
    return this._near;
  }

  public set near(value: number) {
    if (value !== this._near) {
      this._near = value;
      this.dirty = true;
    }
  }

  public get far(): number {
    return this._far;
  }

  public set far(value: number) {
    if (value !== this._far) {
      this._far = value;
      this.dirty = true;
    }
  }

  public get aspectRatio(): number {
    return this._aspectRatio;
  }

  public set aspectRatio(value: number) {
    if (value !== this._aspectRatio) {
      this._aspectRatio = value;
      this.dirty = true;
    }
  }

  public get dirty(): boolean {
    return this._dirty;
  }

  public set dirty(dirty: boolean) {
    this._dirty = dirty;
    if (dirty) this._version++;
  }

  //changes with every change of the matrix. Unlike dirty, which whoever reads the matrix first clears, it tells a
  //camera whether its view projection is out of date
  public get version(): number {
    return this._version;
  }

  public get frustumCorner(): Vector3 {
    if (this.dirty) this.updateMatrix();

    const height: number = this.frustumHeight * 0.5;
    const width: number = height * this._aspectRatio;
    return new Vector3(width, height, 1);
  }

  public get matrix(): Matrix4x4 {
    this.updateMatrix();
    return this.projectionMatrix;
  }

  public get FOV(): number {
    return this.fov;
  }

  public set FOV(value: number) {
    if (value !== this.fov) {
      this.fov = value;
      this.frustumHeight = Math.tan(this.fov * 0.5) * 2;
      this.dirty = true;
    }
  }

  //can be used for AR projects where the projection matrix is provided by the AR framework.
  public setMatrix(matrix: Matrix4x4) {
    Matrix4x4.copy(matrix, this.projectionMatrix);
    this._dirty = false;
    this._version++;
  }

  protected updateMatrix() {
    if (!this.dirty) {
      return;
    }
    const top = this._near * Math.tan(this.fov / 2);
    const right = top * this._aspectRatio;
    //the offset is a fraction of the frustum size
    const offsetX = (this._frustumOffset.x + this._jitter.x) * 2 * right;
    const offsetY = (this._frustumOffset.y + this._jitter.y) * 2 * top;
    Matrix4x4.frustum(-right + offsetX, right + offsetX, -top + offsetY, top + offsetY, this._near, this._far, this.projectionMatrix);

    this.dirty = false;
  }

  public setFrustumOffsetX(value: number) {
    this._frustumOffset.x = value;
    this.dirty = true;
  }

  public setFrustumOffsetY(value: number) {
    this._frustumOffset.y = value;
    this.dirty = true;
  }

  //a sub pixel offset per frame from the halton (2, 3) sequence. x and y take the same index, so any run of consecutive
  //frames spreads its offsets evenly over the pixel. it repeats after sampleCount frames, and starts at index 1:
  //index 0 is the pixel corner in both x and y
  public jitter(resolution:Vector2, sampleCount: number = 64){
    const index = this.jitterFrame % sampleCount + 1;
    this._jitter.x = (haltonSequence(index, 2) - 0.5) / resolution.x;
    this._jitter.y = (haltonSequence(index, 3) - 0.5) / resolution.y;
    this.jitterFrame++;
    this.dirty = true;
  }

  public resetJitter(){
    this._jitter.setValues(0,0);
    this.dirty = true;
  }

  public get jitterOffset(): Vector2 {
    return this._jitter;
  }
}
