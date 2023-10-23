# Demo 3D generator

A containerised web-hostable demo of a [shap-e](https://github.com/openai/shap-e) with .glb output support for easy viewing 3D asset in AR/VR applications.

## Deployment

```sh
docker build --no-cache -t <docker tag> .
docker run --runtime nvidia --gpus all -p 8000:8000 <image tag>
```

## Client Usage

A client can to get .glb asset from a prompt from link: `<insert hostname>:8000/text/<insert prompt>`. Any glb/gltf viewer can then be used to view the asset, like https://gltf-viewer.donmccurdy.com.
