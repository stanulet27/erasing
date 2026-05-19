"""Run the full eval pipeline (init → generate → ensemble → FID)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from eval.checkpoints import (
    ESD_CHECKPOINT,
    checkpoint_paths,
    publish_checkpoints_to_volume,
    sync_checkpoints_from_volume,
)
from eval.fid import compute_fid
from eval.layout import ERASING_ROOT, OUTPUT_MODELS, experiment_paths
from eval.results import load_results, make_results_template, save_results, update_fid_scores

def _python() -> str:
    return sys.executable


def run_esd_train(
    *,
    erase_concept: str,
    erase_from: str | None = None,
    train_method: str = "esd-u",
    iterations: int = 200,
    negative_guidance: float = 2.0,
    device: str = "cuda:0",
) -> str:
    """Train ESD with the given hyperparameters and publish weights to the checkpoints volume."""
    erase_from = erase_from or erase_concept
    cmd = [
        _python(),
        str(ERASING_ROOT / "esd_sd.py"),
        "--erase_concept",
        erase_concept,
        "--erase_from",
        erase_from,
        "--train_method",
        train_method,
        "--iterations",
        str(iterations),
        "--negative_guidance",
        str(negative_guidance),
        "--device",
        device,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ERASING_ROOT)

    esd_path = ERASING_ROOT / ESD_CHECKPOINT
    if not esd_path.is_file():
        raise FileNotFoundError(
            f"ESD training finished but checkpoint not found at {esd_path}. "
            "Check erase_concept / train_method match eval/checkpoints.py."
        )

    publish_checkpoints_to_volume()
    print(f"Published ESD checkpoint to volume: {ESD_CHECKPOINT}")
    return ESD_CHECKPOINT


def run_init(
    *,
    negative_guidance: float,
    iterations: int,
    erase_concept: str,
    prompts_path: str,
    num_images: int,
    base_model: str,
    num_inference_steps: int,
    guidance_scale: float,
) -> None:
    subprocess.run(
        [
            _python(),
            str(ERASING_ROOT / "evalscripts" / "init_eval_run.py"),
            "--negative-guidance",
            str(negative_guidance),
            "--iterations",
            str(iterations),
            "--erase-concept",
            erase_concept,
            "--prompts-path",
            prompts_path,
            "--num-images",
            str(num_images),
            "--base-model",
            base_model,
            "--num-inference-steps",
            str(num_inference_steps),
            "--guidance-scale",
            str(guidance_scale),
        ],
        check=True,
        cwd=ERASING_ROOT,
    )


def run_generate(
    *,
    negative_guidance: float,
    iterations: int,
    device: str = "cuda:0",
    only: list[str] | None = None,
) -> None:
    cmd = [
        _python(),
        str(ERASING_ROOT / "evalscripts" / "generate_eval_outputs.py"),
        "--negative-guidance",
        str(negative_guidance),
        "--iterations",
        str(iterations),
        "--device",
        device,
    ]
    if only:
        cmd.extend(["--only", *only])
    subprocess.run(cmd, check=True, cwd=ERASING_ROOT)


def run_ensemble(
    *,
    negative_guidance: float,
    iterations: int,
    only: list[str] | None = None,
) -> None:
    cmd = [
        _python(),
        str(ERASING_ROOT / "evalscripts" / "score_ensemble.py"),
        "--negative-guidance",
        str(negative_guidance),
        "--iterations",
        str(iterations),
    ]
    if only:
        cmd.extend(["--only", *only])
    subprocess.run(cmd, check=True, cwd=ERASING_ROOT)


def run_fid(
    *,
    negative_guidance: float,
    iterations: int,
    coco_dir: Path,
) -> None:
    paths = experiment_paths(negative_guidance, iterations)
    results = load_results(paths.results_path)

    scores: dict[str, float] = {}
    for model in OUTPUT_MODELS:
        folder = paths.output_dir(model)
        print(f"FID {model}: {folder} vs {coco_dir}")
        scores[model] = compute_fid(folder, coco_dir)
        print(f"  FID = {scores[model]:.4f}")

    update_fid_scores(
        results,
        sd14_base=scores["sd14_base"],
        esd=scores["esd"],
        rl=scores["rl"],
    )
    save_results(paths.results_path, results)
    deg = results["fid_vs_coco"]["degeneration_pct_vs_sd14_base"]
    print(f"Degeneration vs sd14_base: esd={deg['esd']:.2f}%  rl={deg['rl']:.2f}%")


def run_eval_pipeline(
    *,
    negative_guidance: float,
    iterations: int,
    erase_concept: str = "teddy bear",
    prompts_path: str = "eval/prompts/bears_eval_500.csv",
    num_images: int = 500,
    base_model: str = "CompVis/stable-diffusion-v1-4",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    device: str = "cuda:0",
    coco_dir: str | None = "/coco/eval",
    skip_esd_train: bool = True,
    skip_init: bool = False,
    skip_generate: bool = False,
    skip_ensemble: bool = False,
    skip_fid: bool = False,
    train_method: str = "esd-u",
    generate_only: list[str] | None = None,
) -> Path:
    if not skip_esd_train:
        run_esd_train(
            erase_concept=erase_concept,
            train_method=train_method,
            iterations=iterations,
            negative_guidance=negative_guidance,
            device=device,
        )

    sync_checkpoints_from_volume()

    esd_rel, rl_rel = checkpoint_paths()
    paths = experiment_paths(negative_guidance, iterations)

    if not skip_init:
        run_init(
            negative_guidance=negative_guidance,
            iterations=iterations,
            erase_concept=erase_concept,
            prompts_path=prompts_path,
            num_images=num_images,
            base_model=base_model,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    elif not paths.results_path.is_file():
        results = make_results_template(
            negative_guidance=negative_guidance,
            iterations=iterations,
            erase_concept=erase_concept,
            base_model=base_model,
            num_images=num_images,
            prompts_path=prompts_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            esd_checkpoint=esd_rel,
            rl_checkpoint=rl_rel,
        )
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        for model in OUTPUT_MODELS:
            paths.output_dir(model).mkdir(parents=True, exist_ok=True)
        save_results(paths.results_path, results)

    if not skip_generate:
        run_generate(
            negative_guidance=negative_guidance,
            iterations=iterations,
            device=device,
            only=generate_only,
        )

    if not skip_ensemble:
        run_ensemble(
            negative_guidance=negative_guidance,
            iterations=iterations,
            only=generate_only,
        )

    if not skip_fid:
        if not coco_dir:
            print("skip_fid: no coco_dir provided")
        else:
            coco_path = Path(coco_dir)
            if coco_path.is_dir():
                run_fid(
                    negative_guidance=negative_guidance,
                    iterations=iterations,
                    coco_dir=coco_path,
                )
            else:
                print(f"skip_fid: COCO directory not found at {coco_path}")

    print(f"Eval complete: {paths.run_dir}")
    return paths.run_dir
