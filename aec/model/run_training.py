from dtln_aec_model import DTLN_model
import os

# use the GPU with idx 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# activate this for some reproducibility
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# path to folder containing the mixed audio training files
path_to_train_mix = "/Users/allen/Project/YaShi/aec/data/train_data/mixed"
# path to folder containing the mic audio files for training
path_to_train_mic = "/Users/allen/Project/YaShi/aec/data/train_data/mic"
# path to folder containing the lpb audio files for training
path_to_train_lpb = "/Users/allen/Project/YaShi/aec/data/train_data/lpb"
# path to folder containing the mixed audio validation data
path_to_val_mix = "/Users/allen/Project/YaShi/aec/data/train_data/mixed"
# path to folder containing the mic audio validation data
path_to_val_mic = "/Users/allen/Project/YaShi/aec/data/train_data/mic"
# path to folder containing the lpb audio files for validation
path_to_val_lpb = "/Users/allen/Project/YaShi/aec/data/train_data/lpb"

# name your training run
run_name = "DTLN_AEC_model"
# create instance of the DTLN model class
model_trainer = DTLN_model()
# build the model
model_trainer.build_dtln_aec_model()
# compile it with optimizer and cost function for training
model_trainer.compile_model()
# train the model
model_trainer.train_model(
    run_name,
    path_to_train_mix,
    path_to_train_mic,
    path_to_train_lpb,
    path_to_val_mix,
    path_to_val_mic,
    path_to_val_lpb,
)
