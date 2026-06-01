import argparse
import os
import sys

import pandas as pd
import torch
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.esd_checkpoint import apply_esd_checkpoint

torch.set_grad_enabled(False)


def make_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def infer_model_name(base_model: str, esd_path: str | None) -> str:
    if esd_path is not None:
        return os.path.basename(esd_path).split(".")[0]
    if "flux.2-klein" in base_model.lower() or "flux2-klein" in base_model.lower():
        return "flux2-klein"
    if "flux" in base_model.lower():
        return "flux"
    if "xl" in base_model.lower():
        return "sdxl"
    if "stable-diffusion-v1" in base_model.lower() or "compvis" in base_model.lower():
        return "sdv14"
    return "custom"


def generate_images(
    base_model,
    esd_path,
    prompts_path,
    save_path,
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    guidance_scale=7.5,
    num_inference_steps=100,
    num_samples=10,
    from_case=0,
    component=None,
    output_dir=None,
    rl_path = None
):
    """
    Generate images from a diffusers pipeline with an optional ESD checkpoint.
    """
    model_name = infer_model_name(base_model, esd_path)
    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch_dtype).to(device)

    if esd_path is not None:
        try:
            metadata, resolved_component, _ = apply_esd_checkpoint(
                pipe,
                esd_path,
                device="cpu",
                component_name=component,
            )
            if metadata.get("base_model_id") and metadata["base_model_id"] != base_model:
                print(
                    "Warning: checkpoint metadata was created for "
                    f"{metadata['base_model_id']}, but you requested {base_model}."
                )
            print(f"Loaded ESD weights into pipe.{resolved_component}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ESD checkpoint '{esd_path}' for base model '{base_model}'."
            ) from exc
    elif rl_path is not None: 
        #try rl
        procs = {}
        for name in pipe.unet.attn_processors.keys():
            cross_dim = (
                None if name.endswith("attn1.processor")
                else pipe.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden = pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden = list(reversed(pipe.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden = pipe.unet.config.block_out_channels[block_id]
            procs[name] = LoRAAttnProcessor(hidden_size=hidden, cross_attention_dim=cross_dim)
        pipe.unet.set_attn_processor(procs)
        pipe.unet.load_attn_procs(rl_path)


    df = pd.read_csv(prompts_path)
    if output_dir is not None:
        folder_path = output_dir
    else:
        folder_path = os.path.join(save_path, model_name)
    os.makedirs(folder_path, exist_ok=True)

    for _, row in df.iterrows():
        prompt = [str(row.prompt)] * num_samples
        seed = int(row.evaluation_seed)
        case_number = int(row.case_number)
        if case_number < from_case:
            continue

        images = pipe(
            prompt,
            generator=make_generator(device, seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
        for sample_idx, image in enumerate(images):
            image.save(os.path.join(folder_path, f"{case_number}_{sample_idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages",
        description="Generate images using a diffusers pipeline and optional ESD weights.",
    )
    parser.add_argument(
        "--base_model",
        help="base model to load",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--esd_path", help="path to an ESD checkpoint", type=str, default=None)
    parser.add_argument("--rl_path", help="path to an RL checkpoint", type=str, default=None)
        
    parser.add_argument(
        "--component",
        help="explicit component to update (for example: unet or transformer). Usually auto-detected.",
        type=str,
        default=None,
    )
    parser.add_argument("--prompts_path", help="path to csv file with prompts", type=str, required=True)
    parser.add_argument("--save_path", help="folder where to save images", type=str, default="esd-images/")
    parser.add_argument(
        "--output_dir",
        help="write images directly here (no model-name subfolder). Overrides save_path layout.",
        type=str,
        default=None,
    )
    parser.add_argument("--device", help="cuda device to run on", type=str, default="cuda:0")
    parser.add_argument("--guidance_scale", help="guidance scale for eval", type=float, default=7.5)
    parser.add_argument("--from_case", help="continue generating from case_number", type=int, default=0)
    parser.add_argument("--num_samples", help="number of samples per prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", help="number of inference steps", type=int, default=20)
    args = parser.parse_args()

    generate_images(
        base_model=args.base_model,
        esd_path=args.esd_path,
        rl_path=args.rl_path,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_samples=args.num_samples,
        from_case=args.from_case,
        component=args.component,
        output_dir=args.output_dir,
    )