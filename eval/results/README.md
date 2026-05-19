# Eval Pipeline

## One-time Setup Instructions

### Overview
| Volume | Contents |
|--------|----------|
| `esd-checkpoints` | `teddy_bear-i200-s5.safetensors` (RL weights); ESD weights written each train |
| `coco-val2017` | `/eval/train2017/` + `/eval/val2017/` — 500 matched reference JPGs |
| `eval-runs` | outputs + `results.json` per experiment (pulled locally into `erasing/eval/results/`) |


### Steps: 
1. **Download COCO eval references (500 images) & upload to modal volume**
You can see the list of 500 eval prompts `erasing/eval/prompts/bears_eval_500.csv` (250 wild bear + 250 teddy bear, seed 42). Note that eval prompts are different from `bears_combined.txt` prompts used for training (i'm pretty sure).

  ```bash
  python scripts/download_coco_eval_images.py
  modal volume put coco-val2017 ./coco_images /eval
  find coco_images -name "*.jpg" | wc -l   # expect 500
  modal volume ls coco-val2017 /eval # expect eval/train2017 and eval/val2017
  ```

  FID uses this to compare against each model’s 500 generated outputs. These are JUST the 500 images that correspond to the eval prompts in `erasing/eval/prompts/bears_eval_500.csv`. 

2. **Upload RL finetuned weights to modal volume.**
Download weights from Chris's drive. Path location in repo should be `erasing/esd-models/sd/teddy_bear-i200-s5.safetensors` 
```bash
modal volume put esd-checkpoints erasing/esd-models/sd/teddy_bear-i200-s5.safetensors /
```


## Run experiments on Modal

Train ESD for the given hyperparameters, then run the full eval pipeline:
```bash
modal run -d run_eval.py --negative-guidance {NEG_GUIDANCE} --iterations {ITERS}
```

Pull results to local
```bash
modal volume get eval-runs neg2_iter200 erasing/eval/results
```


## Output format
```text
erasing/eval/results/
  neg2_iter200/
    results.json
    outputs/{sd14_base,esd,rl}/
```