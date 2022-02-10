# MBRL4FIN

### setup:
download [data](https://drive.google.com/file/d/18QHvs9nWj8GsZDa6-l56zr7GIBx4G-bg/view?usp=sharing) and decompress it into "./datasets"
pip install -r requirements.txt

### run:
python main.py device="cpu"    

### Parameters configuration
We use hydra, so you can refer to the configuration in the "conf" directory.

for example:

python main.py algorithm=mbpo device=cuda:0 dynamics_model.model.propagation_method=mask dynamics_model.mask_penalty=0 overrides.save_rollout_stats=true overrides.num_steps=8000
