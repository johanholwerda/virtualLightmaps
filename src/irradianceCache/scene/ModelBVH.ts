import Vector3 from "monks-webgpu/renderer/math/Vector3";

/*
SAH BVH over axis aligned boxes, with one box per leaf: the boxes are the primitives (see traceBVH in rayTraceScene.wgsl),
so a leaf is hit by the box test of its parent and has no node of its own.
A node holds the boxes of its 2 children, so traversal reads one node per visited node.
Node layout, 16 x 32 bits: aMin.xyz (f32), a (u32), aMax.xyz (f32), b (u32), bMin.xyz (f32), 0, bMax.xyz (f32), 0.
A child is a node index, or the box index | leafFlag, the index into the boxes the BVH was made with. So the boxes
keep their indices when the BVH is made again after one moved. See BVHNode in rayTraceScene.wgsl.
The build tries every split on the 3 axes (full sweep). The boxes are sorted once per axis, a split keeps the order of
both sides, so a node costs its box count and the build n log n. When a box moved, refit only updates the boxes above it
 */
export default class ModelBVH {
  public static readonly bytesPerNode = 64;
  public static readonly leafFlag = 0x80000000;
  private static readonly wordsPerNode = 16;
  //the indices of the boxes in the order of the leaves, left to right. Boxes that are close in the tree are close in it
  public readonly order: number[] = [];
  //the most nodes on a path from the root to a leaf
  public depth: number = 0;
  public nodeCount = 0;
  //the nodes, in the layout above
  private nodeFloats: Float32Array;
  private nodeUints: Uint32Array;
  //the parent of every node, -1 for the root, and the node that has the box as a child. For refit
  private parents: Int32Array;
  private leafParents: Int32Array;
  private boxMin: Vector3[];
  private boxMax: Vector3[];
  //the most nodes on a path from the root to a leaf that the build allows, see split
  private readonly maxDepth: number;

  //only during the build
  //min.xyz and max.xyz per box
  private boxes!: Float64Array;
  //the box indices sorted on their center, per axis. A range holds the same boxes on every axis
  private sorted!: Uint32Array[];
  private rightArea!: Float64Array;
  //1 for the boxes that go to the first child
  private isLeft!: Uint8Array;
  private partitioned!: Uint32Array;

  constructor(boxMin: Vector3[], boxMax: Vector3[], maxDepth = Infinity) {
    this.maxDepth = maxDepth;
    this.boxMin = boxMin;
    this.boxMax = boxMax;
    const count = boxMin.length;
    const data = new ArrayBuffer(Math.max(count - 1, 1) * ModelBVH.bytesPerNode);
    this.nodeFloats = new Float32Array(data);
    this.nodeUints = new Uint32Array(data);
    this.parents = new Int32Array(Math.max(count - 1, 1)).fill(-1);
    this.leafParents = new Int32Array(count);

    if (count == 1) {
      //a root with the only box as both children
      this.nodeCount = 1;
      this.nodeUints[3] = this.nodeUints[7] = ModelBVH.leafFlag;
      this.order.push(0);
      this.depth = 1;
      this.refit(0);
    } else if (count > 1) {
      this.boxes = new Float64Array(count * 6);
      const centers = new Float64Array(count * 3);
      for (let i = 0; i < count; i++) {
        const min = boxMin[i], max = boxMax[i];
        this.boxes.set([min.x, min.y, min.z, max.x, max.y, max.z], i * 6);
        centers.set([(min.x + max.x) * .5, (min.y + max.y) * .5, (min.z + max.z) * .5], i * 3);
      }
      this.sorted = [0, 1, 2].map(axis => {
        const indices = new Uint32Array(count);
        for (let i = 0; i < count; i++) indices[i] = i;
        //equal centers in the order of the boxes, so the BVH is the same on every browser
        return indices.sort((a, b) => centers[a * 3 + axis] - centers[b * 3 + axis] || a - b);
      });
      this.rightArea = new Float64Array(count);
      this.isLeft = new Uint8Array(count);
      this.partitioned = new Uint32Array(count);
      this.split(0, count, 1);
    }
  }

  public setNodeData(data: ArrayBuffer) {
    new Uint32Array(data).set(this.nodeUints.subarray(0, this.nodeCount * ModelBVH.wordsPerNode));
  }

  //after the box changed in the boxMin and boxMax of the constructor: updates the boxes of the nodes on the path from it
  //to the root. The tree stays the same, so it is less good than a new one when the box moved far
  public refit(boxIndex: number) {
    const f = this.nodeFloats, u = this.nodeUints;
    const min = this.boxMin[boxIndex], max = this.boxMax[boxIndex];
    let minX = min.x, minY = min.y, minZ = min.z;
    let maxX = max.x, maxY = max.y, maxZ = max.z;
    let child = (boxIndex | ModelBVH.leafFlag) >>> 0;
    let node = this.leafParents[boxIndex];
    while (node >= 0) {
      const o = node * ModelBVH.wordsPerNode;
      //both for the root of a single box
      for (let side = 0; side < 2; side++) {
        if (u[o + 3 + side * 4] != child) continue;
        const b = o + side * 8;
        f[b] = minX;
        f[b + 1] = minY;
        f[b + 2] = minZ;
        f[b + 4] = maxX;
        f[b + 5] = maxY;
        f[b + 6] = maxZ;
      }
      //the box of the node, for its parent
      minX = Math.min(f[o], f[o + 8]);
      minY = Math.min(f[o + 1], f[o + 9]);
      minZ = Math.min(f[o + 2], f[o + 10]);
      maxX = Math.max(f[o + 4], f[o + 12]);
      maxY = Math.max(f[o + 5], f[o + 13]);
      maxZ = Math.max(f[o + 6], f[o + 14]);
      child = node;
      node = this.parents[node];
    }
  }

  //the box around the boxes of the range, into the node at offset
  private writeBounds(start: number, count: number, indices: Uint32Array, offset: number) {
    const boxes = this.boxes;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = start; i < start + count; i++) {
      const b = indices[i] * 6;
      minX = Math.min(minX, boxes[b]);
      minY = Math.min(minY, boxes[b + 1]);
      minZ = Math.min(minZ, boxes[b + 2]);
      maxX = Math.max(maxX, boxes[b + 3]);
      maxY = Math.max(maxY, boxes[b + 4]);
      maxZ = Math.max(maxZ, boxes[b + 5]);
    }
    const f = this.nodeFloats;
    f[offset] = minX;
    f[offset + 1] = minY;
    f[offset + 2] = minZ;
    f[offset + 4] = maxX;
    f[offset + 5] = maxY;
    f[offset + 6] = maxZ;
  }

  //a child of a node: the box index | leafFlag for a single box, else the index of a new node for the range
  private getChild(start: number, count: number, depth: number, parent: number): number {
    if (count == 1) {
      const box = this.sorted[0][start];
      this.depth = Math.max(this.depth, depth);
      this.order.push(box);
      this.leafParents[box] = parent;
      return (box | ModelBVH.leafFlag) >>> 0;
    }
    const node = this.split(start, count, depth);
    this.parents[node] = parent;
    return node;
  }

  //the nodes are in depth first order, so a node is usually close to its parent in memory
  private split(start: number, count: number, depth: number): number {
    const index = this.nodeCount++;
    const boxes = this.boxes;
    const rightArea = this.rightArea;

    //the split with the least surface area weighted box count of the children
    let bestCost = Infinity;
    let bestAxis = 0;
    let bestSplit = 1;
    //the most boxes a child can hold and still have its leaves within maxDepth: a child is at depth + 1, and a subtree
    //at depth d holds at most 2^(maxDepth - d) leaves. Only limits the splits that would make the BVH too deep, the
    //rest stays SAH. Without a valid split (too many boxes for maxDepth) it is not limited
    let maxChildCount = Math.pow(2, this.maxDepth - depth - 1);
    if (2 * maxChildCount < count) maxChildCount = Infinity;
    for (let axis = 0; axis < 3; axis++) {
      const indices = this.sorted[axis];

      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let i = count - 1; i > 0; i--) {
        const b = indices[start + i] * 6;
        minX = Math.min(minX, boxes[b]);
        minY = Math.min(minY, boxes[b + 1]);
        minZ = Math.min(minZ, boxes[b + 2]);
        maxX = Math.max(maxX, boxes[b + 3]);
        maxY = Math.max(maxY, boxes[b + 4]);
        maxZ = Math.max(maxZ, boxes[b + 5]);
        const x = maxX - minX, y = maxY - minY, z = maxZ - minZ;
        rightArea[i] = x * y + y * z + z * x;
      }

      minX = minY = minZ = Infinity;
      maxX = maxY = maxZ = -Infinity;
      for (let i = 1; i < count; i++) {
        const b = indices[start + i - 1] * 6;
        minX = Math.min(minX, boxes[b]);
        minY = Math.min(minY, boxes[b + 1]);
        minZ = Math.min(minZ, boxes[b + 2]);
        maxX = Math.max(maxX, boxes[b + 3]);
        maxY = Math.max(maxY, boxes[b + 4]);
        maxZ = Math.max(maxZ, boxes[b + 5]);
        const x = maxX - minX, y = maxY - minY, z = maxZ - minZ;
        const cost = (x * y + y * z + z * x) * i + rightArea[i] * (count - i);
        if (cost < bestCost && i <= maxChildCount && count - i <= maxChildCount) {
          bestCost = cost;
          bestAxis = axis;
          bestSplit = i;
        }
      }
    }

    //the first bestSplit boxes on the best axis go to the first child. On the other axes they are moved to the front,
    //both sides stay sorted
    const best = this.sorted[bestAxis];
    const isLeft = this.isLeft;
    const end = start + count;
    for (let i = start; i < end; i++) isLeft[best[i]] = i < start + bestSplit ? 1 : 0;
    const partitioned = this.partitioned;
    for (let axis = 0; axis < 3; axis++) {
      if (axis == bestAxis) continue;
      const indices = this.sorted[axis];
      let left = start, right = start + bestSplit;
      for (let i = start; i < end; i++) {
        const box = indices[i];
        if (isLeft[box]) partitioned[left++] = box;
        else partitioned[right++] = box;
      }
      indices.set(partitioned.subarray(start, end), start);
    }

    const o = index * ModelBVH.wordsPerNode;
    this.writeBounds(start, bestSplit, best, o);
    this.writeBounds(start + bestSplit, count - bestSplit, best, o + 8);
    this.nodeUints[o + 3] = this.getChild(start, bestSplit, depth + 1, index);
    this.nodeUints[o + 7] = this.getChild(start + bestSplit, count - bestSplit, depth + 1, index);
    return index;
  }
}
