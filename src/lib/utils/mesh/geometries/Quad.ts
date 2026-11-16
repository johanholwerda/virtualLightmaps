import Mesh from 'monks-webgpu/renderer/mesh/Mesh';
import Renderer from 'monks-webgpu/renderer/render/Renderer';

export default class Quad extends Mesh {
  constructor(renderer: Renderer) {
    super(renderer);

    const positionData: Float32Array = new Float32Array([
      -1, -1, 0,
      1, -1, 0,
      -1, 1, 0,
      1, 1, 0,
    ]);
    this.setPositions(positionData);

    const uvData: Float32Array = new Float32Array([
      0, 1,
      1, 1,
      0, 0,
      1, 0,
    ]);

    /*const uvData: Float32Array = new Float32Array([
      0, 0,
      1, 0,
      0, 1,
      1, 1,
    ]);*/
    this.setUV0(uvData);

    const normalData: Float32Array = new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]);
    this.setNormals(normalData);

    this.setIndices( [0, 1, 2, 2, 1, 3]);
  }
}
