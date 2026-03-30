
## 📦 Installation

```bash
cd dex-mot
conda create -n dex-mot python=3.10
conda activate dex-mot

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# for qwen3.5 linear attention
pip install causal-conv1d
pip install git+https://github.com/fla-org/flash-linear-attention
```

## Data Constructing
Use these two scripts to generate standard json-style training data:
```bash
# single arm eef control, based on the first frame within an action chunk with multiprocess (very fast)
python utils/gen_json_tac_deltabase_eef_down_parallel.py

# dual arm eef control, based on the first frame within an action chunk with multiprocess (very fast)
python utils/gen_json_tac_deltabase_eef_bimanual_parallel.py
```

You can also refer to these other scripts to generate the single-arm in-lab training data from the raw format, note that these scripts are all for single arm now:
```bash
# joint control, based on the first frame within an action chunk
python utils/gen_json_tac_deltabase_joint_down.py

# eef control, based on the neighbor frame within an action chunk
python utils/gen_json_tac_deltacurr_eef_down.py

# joint control, based on the neighbor frame within an action chunk
python utils/gen_json_tac_deltacurr_joint_down.py
```

There's a good script to analyze the episode stats:
```bash 
python utils/analyze_episode.py
```

## Training
Get a quick start from the following scripts:
```bash
cd scripts
bash train_qwen3vl.sh
```

For the launching script, it now support a multi-node distributed training, remember to modify the crutial args of that:
```sh
# an example for 2-node training
MASTER_ADDR=<your-ip> # run 'ifconfig' to get the ip address of eth0
MASTER_PORT=29500
NUM_MACHINES=2
MACHINE_RANK=0 # remember to modify in different nodes, 0 for master node, 1,2... for others
```
If you simply want a single-node training:
```sh
MASTER_ADDR=<your-ip> # run 'ifconfig' to get the ip address of eth0
MASTER_PORT=29500
NUM_MACHINES=1
MACHINE_RANK=0 # remember to modify in different nodes
```
(remember: effective_batch_size = train_bsz_per_gpu × num_gpus × gradient_accumulation_steps)

The model now include 3 base models: Janus-Pro, Qwen3-VL, Qwen3.5

But only the Qwen3-VL is fully developed with reduced size of tactile expert and residual flow-matching

The other 2 VLMs only have a basic implementation

```bash
cd scripts

# For Janus-Pro
bash train_janus.sh

# For Qwen3-VL
bash train_qwen3vl.sh

# For Qwen3.5
bash train_qwen35.sh
```

## Trainingset Off-line Testing
You should pass the path of training set to the script.

```bash
cd scripts

# For Janus-Pro
bash test_janus_offline.sh

# For Qwen3-VL
bash test_qwen3vl_offline.sh

# For Qwen3.5
bash test_qwen35_offline.sh
```

## Real-world Inference
The script will create a zmq client to listen to the port to collect payload input and output predicted actions.

```bash
cd scripts

# For Janus-Pro
bash test_janus_real.sh

# For Qwen3-VL
bash test_qwen3vl_real.sh

# For Qwen3.5
bash test_qwen35_real.sh
```
