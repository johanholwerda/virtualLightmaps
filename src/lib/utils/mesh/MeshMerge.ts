import Mesh, { VertexInputNames } from '../../renderer/mesh/Mesh';
import Vector3 from '../../renderer/math/Vector3';
import Transform from '../../renderer/scene/Transform';
import Vector4 from '../../renderer/math/Vector4';
import LogGL from '../../renderer/core/LogGL';
import Matrix3x3 from "monks-webgpu/renderer/math/Matrix3x3";

export type MeshMergeOptions = {
  lightMapOffsets: Vector4[] | null;
  addIndexAttribute: boolean;
  vertexUsage: GPUBufferUsageFlags;
  indexUsage: GPUBufferUsageFlags;
}

export const defaultMeshMergeOptions: MeshMergeOptions = {
  lightMapOffsets: null,
  addIndexAttribute: false,
  vertexUsage: typeof GPUBufferUsage !== 'undefined' ? GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST : 0,
  indexUsage: typeof GPUBufferUsage !== 'undefined' ? GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST : 0,
};

export function meshMerge(
  //can pass an array with only 1 mesh
  meshes: Mesh[],
  transforms: Transform[],
  options: Partial<MeshMergeOptions> = {}
): Mesh {
  const mergeOptions = { ...defaultMeshMergeOptions, ...options };

  if (meshes.length < 1) LogGL.error('getMergedMesh: mesh count should be more than 0');
  if (transforms.length < meshes.length) LogGL.error('getMergedMesh: transform count should be more than or equal to mesh count');
  if (meshes.length > 1 && transforms.length != meshes.length)
    LogGL.error('getMergedMesh: transform count should be equal to mesh count when mesh count is > 1');
  if (mergeOptions.lightMapOffsets && mergeOptions.lightMapOffsets.length != transforms.length)
    LogGL.error('getMergedMesh: lightMapOffsets count should be equal to transform count');

  const meshList = [...meshes];
  while (meshList.length < transforms.length) {
    meshList.push(meshList[0]);
  }

  const mergedMesh = new Mesh(meshList[0].renderer, mergeOptions.vertexUsage, mergeOptions.indexUsage);
  const mergedAttributes: Float32Array[] = [];

  mergedMesh.vertexCount = 0;
  const allAttributes: string[] = [];
  const strides: number[] = [];
  const objectCount = transforms.length;

  for (let mesh of meshList) {
    mergedMesh.vertexCount += mesh.vertexCount;
    for (let attribute of mesh.attributes) {
      if (!(allAttributes.indexOf(attribute.name) >= 0)) {
        allAttributes.push(attribute.name);
        strides.push(attribute.stride);
      }
    }
  }

  // console.log('allAttributes', allAttributes);

  const offsets: number[] = [];
  for (let i: number = 0; i < allAttributes.length; i++) {
    offsets[i] = 0;
    mergedAttributes[i] = new Float32Array(mergedMesh.vertexCount * strides[i]);
  }

  const indices = getMergedIndices(meshList, transforms);
  if (indices) mergedMesh.setIndices(indices);

  const pos = new Vector3();
  const normal = new Vector3();

  for (let i: number = 0; i < objectCount; i++) {
    const mesh = meshList[i];
    const lightMapScaleOffset: Vector4 | null = mergeOptions.lightMapOffsets ? mergeOptions.lightMapOffsets[i] : null;
    const matrix = transforms[i].worldMatrix;
    const normalMatrix = <Matrix3x3>Matrix3x3.normalFromMat4(matrix);

    const n = mesh.vertexCount;

    for (let j: number = 0; j < allAttributes.length; j++) {
      const attribute = mesh.getAttribute(allAttributes[j]);
      const mergedAttribute = mergedAttributes[j];
      let ii: number = offsets[j];
      const s = strides[j];
      if (!attribute) {
        ii += n * s;
      } else {
        const attributeData = <Float32Array>attribute.data;
        if (attribute.name == VertexInputNames.POSITION || attribute.name == VertexInputNames.POSITION_LEGACY) {
          //transform position
          let ki = 0;
          for (let k: number = 0; k < n; k++) {
            pos.x = attributeData[ki++];
            pos.y = attributeData[ki++];
            pos.z = attributeData[ki++];

            pos.transform(matrix);

            mergedAttribute[ii++] = pos.x;
            mergedAttribute[ii++] = pos.y;
            mergedAttribute[ii++] = pos.z;
          }
        } else if (attribute.name == VertexInputNames.NORMAL || attribute.name == VertexInputNames.NORMAL_LEGACY) {
          let ki = 0;

          for (let k: number = 0; k < n; k++) {
            normal.x = attributeData[ki++];
            normal.y = attributeData[ki++];
            normal.z = attributeData[ki++];

            normal.transformMat3(normalMatrix);
            normal.normalize();

            mergedAttribute[ii++] = normal.x;
            mergedAttribute[ii++] = normal.y;
            mergedAttribute[ii++] = normal.z;
          }
        } else if (attribute.name == VertexInputNames.TANGENT || attribute.name == VertexInputNames.TANGENT_LEGACY) {
          let ki = 0;

          for (let k: number = 0; k < n; k++) {
            normal.x = attributeData[ki++];
            normal.y = attributeData[ki++];
            normal.z = attributeData[ki++];
            const w = attributeData[ki++];

            normal.transformMat3(normalMatrix);
            normal.normalize();

            mergedAttribute[ii++] = normal.x;
            mergedAttribute[ii++] = normal.y;
            mergedAttribute[ii++] = normal.z;
            mergedAttribute[ii++] = w;
          }
        } else if (lightMapScaleOffset != null && (attribute.name == VertexInputNames.TEXCOORD1 || attribute.name == VertexInputNames.TEXCOORD1_LEGACY)) {
          let ki = 0;
          for (let k: number = 0; k < n; k++) {
            mergedAttribute[ii++] = attributeData[ki++] * lightMapScaleOffset.x + lightMapScaleOffset.z;
            mergedAttribute[ii++] = attributeData[ki++] * lightMapScaleOffset.y + lightMapScaleOffset.w;
          }
        } else {
          mergedAttribute.set(attributeData, ii);
          ii += n * s;
        }
      }
      offsets[j] = ii;
    }
  }

  for (let i: number = 0; i < allAttributes.length; i++) {
    mergedMesh.setAttribute(allAttributes[i], strides[i], mergedAttributes[i]);
  }
  if (mergeOptions.addIndexAttribute) {
    let ii = 0;
    const indexData = new Float32Array(mergedMesh.vertexCount);
    for (let i: number = 0; i < objectCount; i++) {
      const n = meshes[i].vertexCount;
      for (let j: number = 0; j < n; j++) {
        indexData[ii++] = i;
      }
    }
    mergedMesh.setAttribute('aIndex', 1, indexData);
  }

  return mergedMesh;
}

function getMergedIndices(meshes: Mesh[], transforms: Transform[]): number[] | null {
  if (meshes[0].indices) {
    let l = meshes.length;
    let indexCount: number = 0;
    let vertexCount: number = 0;
    for (let i: number = 0; i < l; i++) {
      if (meshes[i].indices === null) {
        throw new ReferenceError('Cannot get merged indices: expected indices to be defined on none or all meshes');
      }
      indexCount += meshes[i].indices.length;
      vertexCount += meshes[i].vertexCount;
    }
    let indices: number[] = [];
    let ii = 0;
    let indexOffset: number = 0;

    for (let i: number = 0; i < l; i++) {
      let source = meshes[i].indices;
      let il: number = source.length;

      const flipped = transforms[i].flipped;

      if (flipped >= 0) {
        for (let j: number = 0; j < il; j++) {
          indices[ii++] = source[j] + indexOffset;
        }
      } else {
        //flip winding
        for (let j: number = 0; j < il / 3; j++) {
          indices[ii++] = source[j * 3 + 0] + indexOffset;
          indices[ii++] = source[j * 3 + 2] + indexOffset;
          indices[ii++] = source[j * 3 + 1] + indexOffset;
        }
      }

      indexOffset += meshes[i].vertexCount;
    }
    return indices;
  }
  return null;
}
