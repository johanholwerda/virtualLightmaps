import Renderer from '../render/Renderer';
import LogGL from '../core/LogGL';

export type TypedArrayData = Float32Array | Uint32Array | Uint16Array | Int16Array | Uint8Array | Int8Array;

export enum VertexInputNames {
  POSITION = 'position',
  TEXCOORD0 = 'uv',
  TEXCOORD1 = 'uv1',
  NORMAL = 'normal',
  TANGENT = 'tangent',
  COLOR = 'color',
  SKIN_INDEX = 'skinIndex',
  SKIN_WEIGHT = 'skinWeight',

  POSITION_LEGACY = 'aPos',
  TEXCOORD0_LEGACY = 'aUV0',
  TEXCOORD1_LEGACY = 'aUV1',
  NORMAL_LEGACY = 'aNormal',
  TANGENT_LEGACY = 'aTangent',
  COLOR_LEGACY = 'aColor',
  SKIN_INDEX_LEGACY = 'aSkinIndex',
  SKIN_WEIGHT_LEGACY = 'aSkinWeight',
}

export class VertexAttribute {
  public data: TypedArrayData | null;
  public name: string;
  public stride: number;
  public bytesPerElement: number;

  constructor(name: string, stride: number, data: TypedArrayData) {
    this.name = name;
    this.stride = stride;
    this.data = data;
    this.bytesPerElement = data.BYTES_PER_ELEMENT;
  }
}

export default class Mesh {
  public renderer: Renderer;
  private _vertexBuffer: GPUBuffer | null = null;
  public indexBuffer: GPUBuffer | null = null;
  //emptied with the attribute data after the upload, unless keepCPUData is set
  public indices: number[] = [];
  public vertexCount = -1;
  public indexCount = -1;
  //the size of a vertex in the vertex buffer in bytes, see getVertexLayout
  public vertexStride = 0;
  public attributes: VertexAttribute[] = [];
  public indexFormat: GPUIndexFormat = 'uint16';
  public interleaveDirty = true;
  public name = 'unNamed';
  public id: number;
  //keeps the attribute data and the indices after the upload to the gpu. needed to read them back (getData, the mesh
  //utils) or to change one attribute of a mesh with several of them later
  public keepCPUData = false;
  //incremented when an attribute is added or changes its size, so the vertex buffer layouts made for the mesh are made again
  public attributeLayoutVersion = 0;
  private vertexUsage: GPUBufferUsageFlags;
  private indexUsage: GPUBufferUsageFlags;

  public static legacyNames: Map<string, string> = new Map(
    [
      [VertexInputNames.POSITION, VertexInputNames.POSITION_LEGACY],
      [VertexInputNames.TEXCOORD0, VertexInputNames.TEXCOORD0_LEGACY],
      [VertexInputNames.TEXCOORD1, VertexInputNames.TEXCOORD1_LEGACY],
      [VertexInputNames.NORMAL, VertexInputNames.NORMAL_LEGACY],
      [VertexInputNames.TANGENT, VertexInputNames.TANGENT_LEGACY],
      [VertexInputNames.COLOR, VertexInputNames.COLOR_LEGACY],
      [VertexInputNames.SKIN_INDEX, VertexInputNames.SKIN_INDEX_LEGACY],
      [VertexInputNames.SKIN_WEIGHT, VertexInputNames.SKIN_WEIGHT_LEGACY]
    ]
  );
  public static legacyToNewNames: Map<string, string> = new Map(Array.from(Mesh.legacyNames, ([name, legacyName]) => [legacyName, name]));

  //vertexUsage and indexUsage can be passed as parameters because they could be updated by a compute shader, which needs GPUBufferUsage.STORAGE
  public constructor(renderer: Renderer, vertexUsage: GPUBufferUsageFlags = GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, indexUsage: GPUBufferUsageFlags = GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST) {
    this.id = renderer.instanceID++;
    this.renderer = renderer;
    this.vertexUsage = vertexUsage;
    this.indexUsage = indexUsage;
  }

  public setAttribute(
    name: string,
    stride: number,
    data: TypedArrayData
  ): void {
    const existing = this.getAttribute(name);
    if (existing) {
      if (existing.stride != stride || existing.bytesPerElement != data.BYTES_PER_ELEMENT) this.attributeLayoutVersion++;
      existing.data = data;
      existing.stride = stride;
      existing.bytesPerElement = data.BYTES_PER_ELEMENT;
    } else {
      this.attributes.push(new VertexAttribute(name, stride, data));
      this.attributeLayoutVersion++;
    }
    this.interleaveDirty = true;
    this.vertexCount = data.length / stride;
  }

  public getAttribute(name: string): VertexAttribute | null {
    for (const attribute of this.attributes) {
      if (attribute.name == name) {
        return attribute;
      }
    }
    return null;
  }

  public setIndices(indices: number[]): void {
    this.indices = indices;
    this.indexCount = indices.length;
    //by the indices themselves, so it doesn't matter whether the attributes are set before or after them
    let maxIndex = 0;
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] > maxIndex) maxIndex = indices[i];
    }
    const use32BitIndices = maxIndex > 65535;
    let data: Uint16Array | Uint32Array;
    if (use32BitIndices) {
      data = new Uint32Array(indices);
    } else {
      //a buffer size has to be a multiple of 4 bytes, so an odd number of 16 bit indices gets one of padding
      data = new Uint16Array(indices.length + (indices.length & 1));
      data.set(indices);
    }
    const bufferDescriptor: GPUBufferDescriptor = {
      size: data.byteLength,
      usage: this.indexUsage,
    };
    //draws earlier in this frame can still use the old buffer
    if (this.indexBuffer) this.renderer.destroyAfterSubmit(this.indexBuffer);
    this.indexBuffer = this.renderer.device.createBuffer(bufferDescriptor);
    this.renderer.writeBuffer(this.indexBuffer, data);
    this.indexFormat = use32BitIndices ? 'uint32' : 'uint16';
  }

  //where every attribute starts in a vertex and the size of a vertex, in bytes. WebGPU wants the attributes (of the
  //formats used here) and the vertex size at multiples of 4 bytes, so an attribute of 1 or 3 bytes is padded to 4
  public getVertexLayout(): { offsets: number[], arrayStride: number } {
    const offsets: number[] = [];
    let arrayStride = 0;
    for (const attribute of this.attributes) {
      offsets.push(arrayStride);
      arrayStride += Math.ceil(attribute.stride * attribute.bytesPerElement / 4) * 4;
    }
    return {offsets, arrayStride};
  }

  private interleave(): void {
    for (const attribute of this.attributes) {
      if (!attribute.data) {
        //the data of the other attributes was dropped after the first upload
        LogGL.warn('Mesh: the vertex buffer of', this.name, 'is not updated: attribute', attribute.name, 'has no data. Set keepCPUData = true before the first draw to change attributes afterwards, or set all of them again.');
        //not again for every draw: setting an attribute tries again
        this.interleaveDirty = false;
        return;
      }
    }

    const {offsets, arrayStride} = this.getVertexLayout();
    const first = this.attributes[0];
    const vertexCount = first.data!.length / first.stride;

    let data: TypedArrayData;
    if (this.attributes.length == 1 && first.stride * first.bytesPerElement == arrayStride) {
      //already laid out like the vertex buffer
      data = first.data!;
    } else {
      //byte by byte, so attributes of different types keep their own bytes (like float32 positions with unorm8x4 colors)
      const bytes = new ArrayBuffer(vertexCount * arrayStride);
      for (let j = 0; j < this.attributes.length; j++) {
        const attribute = this.attributes[j];
        const source = attribute.data!;
        //a view of the attribute's own type on the whole buffer. the offsets and the vertex size are multiples of 4,
        //so they are whole numbers of its elements
        const target = new (<new (buffer: ArrayBuffer) => TypedArrayData>source.constructor)(bytes);
        const stride = attribute.stride;
        const vertexElements = arrayStride / attribute.bytesPerElement;
        let targetIndex = offsets[j] / attribute.bytesPerElement;
        let sourceIndex = 0;
        for (let i = 0; i < vertexCount; i++) {
          for (let k = 0; k < stride; k++) {
            target[targetIndex + k] = source[sourceIndex + k];
          }
          targetIndex += vertexElements;
          sourceIndex += stride;
        }
      }
      data = new Uint8Array(bytes);
    }

    const bufferDescriptor: GPUBufferDescriptor = {
      size: data.byteLength,
      //needs only GPUBufferUsage.VERTEX, unless read in shader
      usage: this.vertexUsage,
    };
    //draws earlier in this frame can still use the old buffer
    if (this._vertexBuffer) this.renderer.destroyAfterSubmit(this._vertexBuffer);
    this._vertexBuffer = this.renderer.device.createBuffer(bufferDescriptor);
    //the view, not data.buffer: a single attribute can be a view on part of a larger buffer
    this.renderer.writeBuffer(this._vertexBuffer, data);

    this.vertexStride = arrayStride;
    this.vertexCount = vertexCount;
    this.interleaveDirty = false;

    // Free CPU memory after uploading to GPU, but keep attribute metadata for pipeline creation
    if (!this.keepCPUData) {
      for (const attribute of this.attributes) {
        attribute.data = null;
      }
      //the index buffer was uploaded in setIndices
      this.indices = [];
    }
  }

  public get vertexBuffer(): GPUBuffer {
    if (this.interleaveDirty || !this._vertexBuffer) {
      this.interleave();
    }
    return this._vertexBuffer!;
  }

  public setJSONData(json: any) {
    for (const attribute of json.attributes) {
      this.setAttribute(attribute.name, attribute.stride, new Float32Array(attribute.data));
    }

    //after the attributes: the index format depends on the vertex count
    if (json.indices && json.indices.length > 0) {
      this.setIndices(json.indices);
    }
  }

  public setPositions(data: Float32Array) {
    this.setAttribute(VertexInputNames.POSITION, 3, data);
  }

  public getPositions() {
    return this.getData(VertexInputNames.POSITION);
  }

  public setUV0(data: Float32Array) {
    this.setAttribute(VertexInputNames.TEXCOORD0, 2, data);
  }

  public getUV0() {
    return this.getData(VertexInputNames.TEXCOORD0);
  }

  public setNormals(data: Float32Array) {
    this.setAttribute(VertexInputNames.NORMAL, 3, data);
  }

  public getNormals() {
    return this.getData(VertexInputNames.NORMAL);
  }

  public getData(name: string): TypedArrayData | null {
    let va = this.getAttribute(name);

    if (!va) {
      va = this.getAttribute(Mesh.legacyNames.get(name)!);
    }

    if (va && va.data) {
      return va.data;
    } else if (va && !va.data) {
      console.warn('Mesh: getData - data was cleared after interleaving. Set keepCPUData=true before accessing vertexBuffer if you need to read back data.');
      return null;
    } else {
      LogGL.log('Mesh: getData', name, 'no data');
      return null;
    }
  }

  public destruct() {
    //not through the vertexBuffer getter, which would interleave a dirty mesh just to destroy the result.
    //after the submit: draws earlier in this frame can still use the buffers
    if (this._vertexBuffer) this.renderer.destroyAfterSubmit(this._vertexBuffer);
    this._vertexBuffer = null;
    if (this.indexBuffer) this.renderer.destroyAfterSubmit(this.indexBuffer);
    this.indexBuffer = null;
  }


}
