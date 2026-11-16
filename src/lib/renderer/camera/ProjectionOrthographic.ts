import Projection from './Projection';
import Vector2 from '../math/Vector2';
import Matrix4x4 from '../math/Matrix4x4';

export default class ProjectionOrthographic extends Projection {
  public bottomLeft: Vector2 = new Vector2();
  public topRight: Vector2 = new Vector2();

  constructor(
    bottomLeft: Vector2,
    topRight: Vector2,
    nearPlane: number = 0.01,
    farPlane: number = 10,
  ) {
    super(1, nearPlane, farPlane);

    this.bottomLeft.copy(bottomLeft);
    this.topRight.copy(topRight);
  }

  public setSize(width: number, height: number) {
    this.bottomLeft = new Vector2(-width, -height);
    this.topRight = new Vector2(width, height);
    this.dirty = true;
  }

  protected updateMatrix() {
    if (!this.dirty) {
      return;
    }

    const jitter = this.jitterOffset;
    const width = this.topRight.x - this.bottomLeft.x;
    const height = this.topRight.y - this.bottomLeft.y;
    const offsetX = jitter.x * width;
    const offsetY = jitter.y * height;

    Matrix4x4.ortho(
      this.bottomLeft.x + offsetX,
      this.topRight.x + offsetX,
      this.bottomLeft.y + offsetY,
      this.topRight.y + offsetY,
      this._near,
      this._far,
      this.projectionMatrix,
    );
    this.dirty = false;
  }
}
