import Mesh from '../../../renderer/mesh/Mesh';
import Renderer from '../../../renderer/render/Renderer';

export default class Torus extends Mesh {
  constructor(
    renderer: Renderer,
    segmentsX: number        = 16,
    segmentsY: number        = 32,
    radius0: number           = 1,
    radius1: number           = .5,
  ) {
    super(renderer);

    const vertexCount: number = (segmentsX + 1) * (segmentsY + 1);
    const positionData: Float32Array = new Float32Array(vertexCount * 3);
    const uvData: Float32Array = new Float32Array(vertexCount * 2);
    const normalsData: Float32Array = new Float32Array(vertexCount * 3);

    let i: number = 0;
    let iUV: number = 0;
    let iNormal: number = 0;

    for (let y: number = 0; y < segmentsY + 1; y++) {
      const phase = (y / segmentsY) * Math.PI * 2;
      const cx = Math.sin(phase);
      const cy = Math.cos(phase);

      for (let x: number = 0; x < segmentsX + 1; x++) {
        const phase2 = (x / segmentsX) * Math.PI * 2;
        const localX = Math.sin(phase2);
        const localY = Math.cos(phase2);

        const r = radius0 + localX * radius1;
        positionData[i++] = cx * r;
        positionData[i++] = localY * radius1;
        positionData[i++] = cy * r;

        uvData[iUV++] = x / segmentsX;
        uvData[iUV++] = y / segmentsY;

        normalsData[iNormal++] = cx * localX;
        normalsData[iNormal++] = localY;
        normalsData[iNormal++] = cy * localX;
      }
    }

    this.setPositions(positionData);
    this.setUV0(uvData);
    this.setNormals(normalsData);

    const indices: number[] = new Array(segmentsX * segmentsY * 6);

    const w: number = segmentsX + 1;
    let indexOffset = 0;
    for (let y: number = 0; y < segmentsY; y++) {
      for (let x: number = 0; x < segmentsX; x++) {
        const a = y * w + x;
        const b = (y + 1) * w + x;
        const c = (y + 1) * w + x + 1;
        const d = y * w + x + 1;

        indices[indexOffset++] = a;
        indices[indexOffset++] = d;
        indices[indexOffset++] = b;

        indices[indexOffset++] = b;
        indices[indexOffset++] = d;
        indices[indexOffset++] = c;
      }
    }
    this.setIndices(indices);
  }
}
