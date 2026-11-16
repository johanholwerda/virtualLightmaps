import Mesh, {VertexInputNames} from '../mesh/Mesh';
import VertexFragmentShader from '../material/VertexFragmentShader';
import LogGL from '../core/LogGL';
import VertexInputUtils, {VertexInput} from '../material/shaderUtils/VertexInputUtils';

type CachedLayout = {
  layout: GPUVertexBufferLayout;
  //Mesh.attributeLayoutVersion it was made for
  attributeLayoutVersion: number;
}

export default class VertexBufferLayoutManager {
  //by mesh and material, without building a key string for every draw. weak, so the meshes and materials that are
  //no longer used don't stay in memory
  private layouts: WeakMap<Mesh, WeakMap<VertexFragmentShader, CachedLayout>> = new WeakMap();

  private static readonly knownNames: Set<string> = new Set(Object.values(VertexInputNames));

  private layoutsByHash: Map<string, GPUVertexBufferLayout> = new Map();
  private vertexInputsByMaterial: WeakMap<VertexFragmentShader, VertexInput[]> = new WeakMap();

  public getVertexBufferLayout(mesh: Mesh, material: VertexFragmentShader): GPUVertexBufferLayout {
    let layoutsByMaterial = this.layouts.get(mesh);
    if (!layoutsByMaterial) {
      layoutsByMaterial = new WeakMap();
      this.layouts.set(mesh, layoutsByMaterial);
    }

    let cached = layoutsByMaterial.get(material);
    //made again when attributes were added or changed their size since
    if (!cached || cached.attributeLayoutVersion !== mesh.attributeLayoutVersion) {
      const layout = this.createVertexBufferLayout(mesh, material);
      const hash = this.getLayoutHash(layout);

      let existing = this.layoutsByHash.get(hash);
      if (!existing) {
        this.layoutsByHash.set(hash, layout);
        existing = layout;
      }
      cached = {layout: existing, attributeLayoutVersion: mesh.attributeLayoutVersion};
      layoutsByMaterial.set(material, cached);
    }
    return cached.layout;
  }

  private getLayoutHash(layout: GPUVertexBufferLayout): string {
    const attributes = Array.from(layout.attributes);
    return JSON.stringify({
      arrayStride: layout.arrayStride,
      stepMode: layout.stepMode,
      attributes: attributes.map(attr => ({
        shaderLocation: attr.shaderLocation,
        offset: attr.offset,
        format: attr.format
      }))
    });
  }


  private createVertexBufferLayout(mesh: Mesh, material: VertexFragmentShader): GPUVertexBufferLayout {
    const vertexInputs = this.getVertexInputs(material);
    const attributes: GPUVertexAttribute[] = [];
    //the layout Mesh interleaves the vertex buffer with
    const {offsets, arrayStride} = mesh.getVertexLayout();

    //the attributes an input matches by name. the fallback by location doesn't take them
    const matchedByName = new Set<number>();
    for (let j = 0; j < mesh.attributes.length; j++) {
      const name = this.getAttributeName(mesh.attributes[j].name);
      if (vertexInputs.some(vertexInput => vertexInput.name == name)) matchedByName.add(j);
    }

    for (const vertexInput of vertexInputs) {
      let index = mesh.attributes.findIndex(attribute => this.getAttributeName(attribute.name) == vertexInput.name);
      if (index < 0) {
        // Fallback: match by location index when name matching fails.
        // This supports shaders that use arbitrary parameter names (e.g. "pos" instead of "position")
        // in inline vs_main parameters. Not for the known names: a mesh without uv1 would give a uv1 input the
        // attribute at index 3, like skinIndex
        const location = vertexInput.location;
        if (location < mesh.attributes.length && !matchedByName.has(location) && !VertexBufferLayoutManager.knownNames.has(vertexInput.name)) {
          index = location;
          LogGL.log(material.name, 'VertexBufferLayoutManager', 'attribute matched by index', index, vertexInput.name, '→', mesh.attributes[index].name);
        }
      }

      if (index < 0) {
        LogGL.warn(material.name, 'VertexBufferLayoutManager', 'attribute not found: adding a dummy attribute', vertexInput, mesh.attributes);
        attributes.push({
          shaderLocation: vertexInput.location,
          offset: 0,
          format: vertexInput.format,
        });
        continue;
      }

      const attribute = mesh.attributes[index];
      if (attribute.stride * attribute.bytesPerElement != this.getFormatSize(vertexInput.format)) {
        LogGL.warn('VertexBufferLayoutManager', 'stride mismatch', vertexInput, attribute);
      }
      attributes.push({
        shaderLocation: vertexInput.location,
        offset: offsets[index],
        format: vertexInput.format,
      });
    }

    return {
      arrayStride,
      attributes: attributes,
    }
  }

  //parsed once per material, not again for every mesh it is drawn with
  private getVertexInputs(material: VertexFragmentShader): VertexInput[] {
    let vertexInputs = this.vertexInputsByMaterial.get(material);
    if (!vertexInputs) {
      vertexInputs = VertexInputUtils.getVertexInputs(material.shaderCode);
      this.vertexInputsByMaterial.set(material, vertexInputs);
    }
    return vertexInputs;
  }

  private getFormatSize(format: GPUVertexFormat): number {
    switch (format) {
      case 'float32':
      case 'uint32':
      case 'sint32':
        return 4;
      case 'float32x2':
      case 'uint32x2':
      case 'sint32x2':
        return 8;
      case 'float32x3':
      case 'uint32x3':
      case 'sint32x3':
        return 12;
      case 'float32x4':
      case 'uint32x4':
      case 'sint32x4':
        return 16;
      case 'unorm8x2':
      case 'snorm8x2':
      case 'uint8x2':
      case 'sint8x2':
        return 2;
      case 'unorm8x4':
      case 'snorm8x4':
      case 'uint8x4':
      case 'sint8x4':
        return 4;
      case 'unorm16x2':
      case 'snorm16x2':
      case 'uint16x2':
      case 'sint16x2':
      case 'float16x2':
        return 4;
      case 'unorm16x4':
      case 'snorm16x4':
      case 'uint16x4':
      case 'sint16x4':
      case 'float16x4':
        return 8;
      default:
        return 0;
    }
  }

  private getAttributeName(name: string): string {
    return Mesh.legacyToNewNames.get(name) ?? name;
  }
}
