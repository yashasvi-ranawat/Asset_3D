import torch
import pygltflib
import numpy as np
from typing import Union
import os


from shap_e.models.download import load_model
from shap_e.util.collections import AttrDict
from shap_e.models.nn.camera import (
    DifferentiableCameraBatch,
    DifferentiableProjectiveCamera
)
from shap_e.models.transmitter.base import Transmitter, VectorDecoder
from shap_e.rendering.torch_mesh import TorchMesh


def create_pan_cameras(size: int,
                       device: torch.device) -> DifferentiableCameraBatch:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins,
                                             axis=0)).float().to(device),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
            width=size,
            height=size,
            x_fov=0.7,
            y_fov=0.7,
        ),
    )


@torch.no_grad()
def decode_latent_mesh(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
) -> TorchMesh:
    decoded = xm.renderer.render_views(
        AttrDict(cameras=create_pan_cameras(2, latent.device)),
        params=(xm.encoder if isinstance(xm, Transmitter)
                else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    return decoded.raw_meshes[0]


def main(latents):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)

    decode_ = decode_latent_mesh(xm, latents[3]).tri_mesh()
    triangles = np.asarray(decode_.faces, dtype=np.uint32)
    points = np.asarray(decode_.verts, dtype=np.float32)
    colors = np.stack([decode_.vertex_channels[x] for x in "RGB"],
                      axis=1, dtype=np.float32)

    triangles_binary_blob = triangles.flatten().tobytes()
    points_binary_blob = points.tobytes()
    colors_binary_blob = colors.tobytes()
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=1, COLOR_0=2),
                        indices=0
                    )
                ]
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_INT,
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.FLOAT,
                count=len(colors),
                type=pygltflib.VEC3,
                max=colors.max(axis=0).tolist(),
                min=colors.min(axis=0).tolist(),
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(triangles_binary_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(triangles_binary_blob),
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=(len(triangles_binary_blob)
                            + len(points_binary_blob)),
                byteLength=len(colors_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=(len(triangles_binary_blob)
                            + len(points_binary_blob)
                            + len(colors_binary_blob))
            )
        ],
    )
    gltf.set_binary_blob(triangles_binary_blob + points_binary_blob
                         + colors_binary_blob)
    return b"".join(gltf.save_to_bytes())


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latents = torch.load("latents.pt", device)
    with open("prompt.txt", "r") as fio:
        prompt = fio.read().strip()

    glb_bin = main(latents)

    with open("test.glb", "wb") as fio:
        fio.write(glb_bin)

    # delete files
    os.remove("prompt.txt")
    os.remove("latents.pt")
