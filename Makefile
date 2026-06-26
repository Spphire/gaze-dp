WKSPACE ?= train_gaze_wam_workspace
TASK ?= gaze_wam

train_acc8_amp:
	HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 accelerate launch --config_file accelerate/8gpu-amp.yaml train.py --config-name $(WKSPACE) task=$(TASK)
