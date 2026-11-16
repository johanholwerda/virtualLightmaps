import Vector3 from "monks-webgpu/renderer/math/Vector3";
import LogGL from "monks-webgpu/renderer/core/LogGL";

//the triangles a leaf of the BVH holds at most
const LEAF_SIZE = 4;
//the bins of the SAH split per axis
const BIN_COUNT = 12;
//the entries of the traversal stack of traceMesh in rayTraceScene.wgsl, the BVH can be 1 deeper
export const MESH_STACK_SIZE = 32;
//vec4s per triangle in meshBuffer, see getMeshTriangle in rayTraceScene.wgsl
const TRIANGLE_VEC4S = 5;

/*
A triangle mesh with lightmap UVs, for the ray tracer of rayTraceScene.wgsl and the virtual lightmap.
-the positions are scaled to fit the -1 to 1 cube, like the primitives: the transform of the model gives it its size
-the charts are the UV islands: the triangles connected by shared edges, with the same UV winding. A mirrored half of a
 mesh (the same UVs, the other winding) is a chart of its own, so the UVs don't have to be unique. A chart is a surface of
 the virtual lightmap like a face of a cube, see getSurface in virtualLightmap.wgsl: its texels are never shared with
 another chart, and the coarser levels never mix 2 charts. So the charts need no padding and the lightmap no dilation:
 the texels a chart has outside its triangles are never sampled, and the resolve weights the texels by their sample count
-a chart has its own scale, from its area in m and in UV: an unwrap where the islands have a different texel density
 still gets the same texels per m everywhere. Every vertex of a triangle gets its position in the chart in local units,
 so the shader only needs the barycentrics: chart coordinate = uv * scale - the corner of the chart
meshBuffer, in vec4s: a header (node offset, triangle offset, chart offset, triangle count), the BVH nodes (2 each:
min + left child or first triangle, max + triangle count, 0 for an inner node), the triangles in BVH order (see
getMeshTriangle) and the chart sizes
 */
export default class LightmapMesh {
  public readonly triangleCount: number;
  public readonly chartCount: number;
  //half the size of the mesh in the -1 to 1 cube, the largest is 1
  public readonly halfExtents = new Vector3();
  public readonly data: ArrayBuffer;
  public bvhDepth = 0;

  private positions: Float32Array;
  private normals: Float32Array;
  private uvs: Float32Array;
  private indices: Uint32Array;
  //per triangle
  private charts: Uint32Array;
  //per triangle corner, in local units
  private chartCoords: Float32Array;
  private chartSizes: Float32Array;

  private nodeMin: number[] = [];
  private nodeMax: number[] = [];
  private nodeLeftFirst: number[] = [];
  private nodeCount: number[] = [];
  private order: Uint32Array;

  constructor(positions: Float32Array, normals: Float32Array, uvs: Float32Array, indices: Uint32Array) {
    this.positions = this.normalize(positions);
    this.normals = normals;
    this.uvs = uvs;
    this.indices = indices;
    this.triangleCount = indices.length / 3;

    this.charts = new Uint32Array(this.triangleCount);
    this.chartCount = this.createCharts();
    this.chartCoords = new Float32Array(this.triangleCount * 6);
    this.chartSizes = new Float32Array(this.chartCount * 2);
    this.layoutCharts();

    this.order = new Uint32Array(this.triangleCount).map((_, i) => i);
    this.buildBVH();
    this.data = this.pack();

    if(LogGL.ENABLED) LogGL.log('LightmapMesh: triangles', this.triangleCount, 'charts', this.chartCount, 'BVH nodes', this.nodeCount.length, 'depth', this.bvhDepth);
    if(this.bvhDepth - 1 > MESH_STACK_SIZE) LogGL.error('LightmapMesh: the BVH is deeper than the traversal stack of traceMesh');
  }

  //from the export of UnitySystem: aPos, aNormal and aUV0 of the first mesh
  public static fromJSON(json: any): LightmapMesh {
    const mesh = json.meshes[0];
    const attribute = (name: string) => {
      const a = mesh.attributes.find((a: any) => a.name == name);
      if(!a) throw new Error(`LightmapMesh: no ${name} in ${json.name}`);
      return new Float32Array(a.data);
    };
    return new LightmapMesh(attribute('aPos'), attribute('aNormal'), attribute('aUV0'), new Uint32Array(mesh.indices));
  }

  //centered and scaled to fit the -1 to 1 cube
  private normalize(positions: Float32Array): Float32Array {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for(let i = 0; i < positions.length; i++){
      min[i % 3] = Math.min(min[i % 3], positions[i]);
      max[i % 3] = Math.max(max[i % 3], positions[i]);
    }
    const half = [0, 1, 2].map(a => (max[a] - min[a]) / 2);
    const scale = 1 / Math.max(...half);
    const result = new Float32Array(positions.length);
    for(let i = 0; i < positions.length; i++){
      result[i] = (positions[i] - (min[i % 3] + max[i % 3]) / 2) * scale;
    }
    this.halfExtents.setValues(half[0] * scale, half[1] * scale, half[2] * scale);
    return result;
  }

  //twice the signed area of the triangle in UV: the sign is the winding, 0 for a triangle without UV area
  private getUVArea(t: number): number {
    const uv = this.uvs, idx = this.indices;
    const a = idx[t * 3] * 2, b = idx[t * 3 + 1] * 2, c = idx[t * 3 + 2] * 2;
    return (uv[b] - uv[a]) * (uv[c + 1] - uv[a + 1]) - (uv[c] - uv[a]) * (uv[b + 1] - uv[a + 1]);
  }

  //twice the area of the triangle in local units
  private getArea(t: number): number {
    const p = this.positions, idx = this.indices;
    const a = idx[t * 3] * 3, b = idx[t * 3 + 1] * 3, c = idx[t * 3 + 2] * 3;
    const e1x = p[b] - p[a], e1y = p[b + 1] - p[a + 1], e1z = p[b + 2] - p[a + 2];
    const e2x = p[c] - p[a], e2y = p[c + 1] - p[a + 1], e2z = p[c + 2] - p[a + 2];
    const x = e1y * e2z - e1z * e2y, y = e1z * e2x - e1x * e2z, z = e1x * e2y - e1y * e2x;
    return Math.sqrt(x * x + y * y + z * z);
  }

  //the UV islands, see the class comment. The vertices of an edge have one UV each, so triangles that share an edge are
  //next to each other in UV too. Returns the chart count
  private createCharts(): number {
    const parent = new Uint32Array(this.triangleCount).map((_, i) => i);
    const find = (i: number): number => {
      while(parent[i] != i){
        parent[i] = parent[parent[i]];
        i = parent[i];
      }
      return i;
    };
    const signs = new Int8Array(this.triangleCount);
    for(let t = 0; t < this.triangleCount; t++){
      const area = this.getUVArea(t);
      signs[t] = Math.abs(area) < 1e-12 ? 0 : Math.sign(area);
    }

    const vertexCount = this.positions.length / 3;
    const edges = new Map<number, number>();
    const idx = this.indices;
    for(let t = 0; t < this.triangleCount; t++){
      for(let e = 0; e < 3; e++){
        const a = idx[t * 3 + e], b = idx[t * 3 + (e + 1) % 3];
        const key = Math.min(a, b) * vertexCount + Math.max(a, b);
        const other = edges.get(key);
        if(other === undefined){
          edges.set(key, t);
          continue;
        }
        //a triangle without UV area goes with either winding
        if(signs[t] * signs[other] >= 0){
          const rootA = find(t), rootB = find(other);
          if(rootA != rootB) parent[rootA] = rootB;
        }
      }
    }

    const chartOfRoot = new Map<number, number>();
    for(let t = 0; t < this.triangleCount; t++){
      const root = find(t);
      let chart = chartOfRoot.get(root);
      if(chart === undefined){
        chart = chartOfRoot.size;
        chartOfRoot.set(root, chart);
      }
      this.charts[t] = chart;
    }
    if(chartOfRoot.size > 0xffff) LogGL.error('LightmapMesh: more charts than the surface id of getSurface has bits for');
    return chartOfRoot.size;
  }

  //the scale of every chart from UV to local units, its size and the position of every triangle corner in it
  private layoutCharts() {
    const n = this.chartCount;
    const area = new Float64Array(n), uvArea = new Float64Array(n);
    const uvMin = new Float64Array(n * 2).fill(Infinity), uvMax = new Float64Array(n * 2).fill(-Infinity);
    const uv = this.uvs, idx = this.indices;
    for(let t = 0; t < this.triangleCount; t++){
      const c = this.charts[t];
      area[c] += this.getArea(t);
      uvArea[c] += Math.abs(this.getUVArea(t));
      for(let k = 0; k < 3; k++){
        const v = idx[t * 3 + k] * 2;
        uvMin[c * 2] = Math.min(uvMin[c * 2], uv[v]);
        uvMin[c * 2 + 1] = Math.min(uvMin[c * 2 + 1], uv[v + 1]);
        uvMax[c * 2] = Math.max(uvMax[c * 2], uv[v]);
        uvMax[c * 2 + 1] = Math.max(uvMax[c * 2 + 1], uv[v + 1]);
      }
    }
    //a chart without UV area is a single texel
    const scales = new Float64Array(n);
    for(let c = 0; c < n; c++){
      scales[c] = uvArea[c] > 0 ? Math.sqrt(area[c] / uvArea[c]) : 0;
      this.chartSizes[c * 2] = Math.max((uvMax[c * 2] - uvMin[c * 2]) * scales[c], 1e-6);
      this.chartSizes[c * 2 + 1] = Math.max((uvMax[c * 2 + 1] - uvMin[c * 2 + 1]) * scales[c], 1e-6);
    }
    for(let t = 0; t < this.triangleCount; t++){
      const c = this.charts[t];
      for(let k = 0; k < 3; k++){
        const v = idx[t * 3 + k] * 2;
        this.chartCoords[t * 6 + k * 2] = (uv[v] - uvMin[c * 2]) * scales[c];
        this.chartCoords[t * 6 + k * 2 + 1] = (uv[v + 1] - uvMin[c * 2 + 1]) * scales[c];
      }
    }
  }

  //binned SAH, the leaves hold at most LEAF_SIZE triangles. Reorders this.order
  private buildBVH() {
    const count = this.triangleCount;
    const p = this.positions, idx = this.indices;
    const boxMin = new Float32Array(count * 3), boxMax = new Float32Array(count * 3), centroid = new Float32Array(count * 3);
    for(let t = 0; t < count; t++){
      for(let a = 0; a < 3; a++){
        const x = p[idx[t * 3] * 3 + a], y = p[idx[t * 3 + 1] * 3 + a], z = p[idx[t * 3 + 2] * 3 + a];
        boxMin[t * 3 + a] = Math.min(x, y, z);
        boxMax[t * 3 + a] = Math.max(x, y, z);
        centroid[t * 3 + a] = (boxMin[t * 3 + a] + boxMax[t * 3 + a]) / 2;
      }
    }
    const surfaceArea = (min: number[], max: number[]) => {
      const x = max[0] - min[0], y = max[1] - min[1], z = max[2] - min[2];
      return x * y + y * z + z * x;
    };
    const order = this.order;

    const addNode = () => {
      this.nodeMin.push(0, 0, 0);
      this.nodeMax.push(0, 0, 0);
      this.nodeLeftFirst.push(0);
      this.nodeCount.push(0);
      return this.nodeCount.length - 1;
    };

    const stack: [node: number, start: number, end: number, depth: number][] = [[addNode(), 0, count, 1]];
    while(stack.length){
      const [node, start, end, depth] = stack.pop()!;
      this.bvhDepth = Math.max(this.bvhDepth, depth);
      const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
      const cMin = [Infinity, Infinity, Infinity], cMax = [-Infinity, -Infinity, -Infinity];
      for(let i = start; i < end; i++){
        const t = order[i];
        for(let a = 0; a < 3; a++){
          min[a] = Math.min(min[a], boxMin[t * 3 + a]);
          max[a] = Math.max(max[a], boxMax[t * 3 + a]);
          cMin[a] = Math.min(cMin[a], centroid[t * 3 + a]);
          cMax[a] = Math.max(cMax[a], centroid[t * 3 + a]);
        }
      }
      for(let a = 0; a < 3; a++){
        this.nodeMin[node * 3 + a] = min[a];
        this.nodeMax[node * 3 + a] = max[a];
      }

      const n = end - start;
      if(n <= LEAF_SIZE){
        this.nodeLeftFirst[node] = start;
        this.nodeCount[node] = n;
        continue;
      }

      //the cheapest split over the bins of the 3 axes
      let bestAxis = -1, bestSplit = 0, bestCost = Infinity;
      for(let a = 0; a < 3; a++){
        const extent = cMax[a] - cMin[a];
        if(extent <= 0) continue;
        const binCount = new Uint32Array(BIN_COUNT);
        const binMin = Array.from({length: BIN_COUNT}, () => [Infinity, Infinity, Infinity]);
        const binMax = Array.from({length: BIN_COUNT}, () => [-Infinity, -Infinity, -Infinity]);
        for(let i = start; i < end; i++){
          const t = order[i];
          const b = Math.min(Math.floor((centroid[t * 3 + a] - cMin[a]) / extent * BIN_COUNT), BIN_COUNT - 1);
          binCount[b]++;
          for(let k = 0; k < 3; k++){
            binMin[b][k] = Math.min(binMin[b][k], boxMin[t * 3 + k]);
            binMax[b][k] = Math.max(binMax[b][k], boxMax[t * 3 + k]);
          }
        }
        //the cost of the left side of every split from the left, then add the right side from the right
        const leftCost = new Float64Array(BIN_COUNT - 1);
        let accMin = [Infinity, Infinity, Infinity], accMax = [-Infinity, -Infinity, -Infinity], accCount = 0;
        for(let s = 0; s < BIN_COUNT - 1; s++){
          accCount += binCount[s];
          accMin = accMin.map((v, k) => Math.min(v, binMin[s][k]));
          accMax = accMax.map((v, k) => Math.max(v, binMax[s][k]));
          leftCost[s] = accCount ? accCount * surfaceArea(accMin, accMax) : 0;
        }
        accMin = [Infinity, Infinity, Infinity]; accMax = [-Infinity, -Infinity, -Infinity]; accCount = 0;
        for(let s = BIN_COUNT - 1; s > 0; s--){
          accCount += binCount[s];
          accMin = accMin.map((v, k) => Math.min(v, binMin[s][k]));
          accMax = accMax.map((v, k) => Math.max(v, binMax[s][k]));
          const cost = leftCost[s - 1] + (accCount ? accCount * surfaceArea(accMin, accMax) : 0);
          if(cost < bestCost){
            bestCost = cost;
            bestAxis = a;
            bestSplit = cMin[a] + extent * s / BIN_COUNT;
          }
        }
      }

      let mid = start;
      if(bestAxis >= 0){
        for(let i = start; i < end; i++){
          if(centroid[order[i] * 3 + bestAxis] < bestSplit){
            const tmp = order[i]; order[i] = order[mid]; order[mid] = tmp;
            mid++;
          }
        }
      }
      //all centroids in one place: split in the middle of the list
      if(mid == start || mid == end) mid = start + (n >> 1);

      const left = addNode();
      addNode();
      this.nodeLeftFirst[node] = left;
      this.nodeCount[node] = 0;
      stack.push([left, start, mid, depth + 1], [left + 1, mid, end, depth + 1]);
    }
  }

  //octahedral, 2 x 16 bit snorm like pack2x16snorm. See meshDecodeNormal in rayTraceScene.wgsl
  private packNormal(x: number, y: number, z: number): number {
    const s = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
    let u = x / s, v = y / s;
    if(z < 0){
      const ou = u;
      u = (1 - Math.abs(v)) * (ou >= 0 ? 1 : -1);
      v = (1 - Math.abs(ou)) * (v >= 0 ? 1 : -1);
    }
    const snorm = (f: number) => Math.round(Math.max(-1, Math.min(1, f)) * 32767) & 0xffff;
    return (snorm(u) | (snorm(v) << 16)) >>> 0;
  }

  private pack(): ArrayBuffer {
    const nodeCount = this.nodeCount.length;
    const nodeOffset = 1;
    const triangleOffset = nodeOffset + nodeCount * 2;
    const chartOffset = triangleOffset + this.triangleCount * TRIANGLE_VEC4S;
    const vec4Count = chartOffset + this.chartCount;
    const buffer = new ArrayBuffer(vec4Count * 16);
    const f = new Float32Array(buffer);
    const u = new Uint32Array(buffer);

    u.set([nodeOffset, triangleOffset, chartOffset, this.triangleCount], 0);

    for(let i = 0; i < nodeCount; i++){
      const o = (nodeOffset + i * 2) * 4;
      f.set(this.nodeMin.slice(i * 3, i * 3 + 3), o);
      u[o + 3] = this.nodeLeftFirst[i];
      f.set(this.nodeMax.slice(i * 3, i * 3 + 3), o + 4);
      u[o + 7] = this.nodeCount[i];
    }

    const p = this.positions, nm = this.normals, idx = this.indices;
    for(let i = 0; i < this.triangleCount; i++){
      const t = this.order[i];
      const o = (triangleOffset + i * TRIANGLE_VEC4S) * 4;
      const v = [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]];
      const normal = (k: number) => this.packNormal(nm[v[k] * 3], nm[v[k] * 3 + 1], nm[v[k] * 3 + 2]);
      for(let k = 0; k < 3; k++){
        f.set(p.subarray(v[k] * 3, v[k] * 3 + 3), o + k * 4);
      }
      u[o + 3] = this.charts[t];
      u[o + 7] = normal(0);
      u[o + 11] = normal(1);
      f.set(this.chartCoords.subarray(t * 6, t * 6 + 4), o + 12);
      f.set(this.chartCoords.subarray(t * 6 + 4, t * 6 + 6), o + 16);
      u[o + 18] = normal(2);
    }

    for(let c = 0; c < this.chartCount; c++){
      f.set(this.chartSizes.subarray(c * 2, c * 2 + 2), (chartOffset + c) * 4);
    }
    return buffer;
  }
}
