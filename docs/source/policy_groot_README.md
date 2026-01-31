# Policy Repo for GR00T-N1.6

## 1. Installation guide

Install astral [uv](https://docs.astral.sh/uv/) and start a new env

```
uv venv --python=3.10
source .venv/bin/activate
```

Install pip and wheel
```
uv pip install -U pip wheel
```

Install repo:

```bash
# Check https://pytorch.org/get-started/locally/ for your system
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Flash attention dependencies
uv pip install ninja "packaging>=24.2,<26.0"

# Install flash attention (requires CUDA)
uv pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation

# Verify installation
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')"
```

<!-- Install LeRobot with Groot Dependencies

```bash
uv pip install -e ".[groot]"
``` -->

Install Additional N1.6 Dependencies

```bash
uv pip install lerobot[transformers-dep]
# uv pip install lmdb==1.7.5
# uv pip install albumentations==1.4.18
# uv pip install "faker>=33.0.0,<35.0.0"
# uv pip install tyro
# uv pip install "matplotlib>=3.10.3,<4.0.0"
# uv pip install torchcodec==0.4.0
```

Install system ffmpeg

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg

# verify
python -c "import torch, torchcodec; print('torch', torch.__version__); print('torchcodec', torchcodec.__version__)"
```

## 2.Launch Training
```
lerobot-train   --policy.type=gr00t_n1d6   --policy.push_to_hub=false   --dataset.repo_id=izuluaga/finish_sandwich   --batch_size=2   --steps=200   --wandb.enable=false   --log_freq=2   --output_dir=./outputs/groot/  --num_workers=0
``

