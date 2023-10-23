import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config


def main(prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    batch_size = 4
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    return latents


if __name__ == "__main__":
    with open("prompt.txt", "r") as fio:
        prompt = fio.read().strip()

    latents = main(prompt)

    torch.save(latents, "latents.pt")
