This follows DPoser's installation.

Environment:

```bash
pip install -r requirements.txt
```

To train: 
```bash
python -m run.train --config configs/subvp/amass_scorefc_continuous.py --name reproduce --bodymodel-path body_models/smplx/SMPLX_NEUTRAL.npz
```

To test the pretrained result for pose completion:
```bash
python -m run.completion_cp --config configs/subvp/amass_scorefc_continuous.py --gpus 1 --hypo 10 --sample 10 --part legs --ckpt-path DPoser_fft/output/amass_amass/FFTPoser/best_model.pth --bodymodel-path body_models/smplx/SMPLX_NEUTRAL.npz
```

To test the pretrained result for pose generation:
```bash
python -m run.demo --config configs/subvp/amass_scorefc_continuous.py  --task generation --metrics --ckpt-path DPoser_fft/output/amass_amass/FFTPoser/best_mode.pth  --bodymodel-path body_models/smplx/SMPLX_NEUTRAL.npz
```
TransLinear Model has 
To check the result for using TransLinear, simply replace lib/algorithms/advanced/model.py with model_stylisation.py

To check the result for placement of FFT, replace lib/algorithms/advanced/loss.py with loss_before.py
