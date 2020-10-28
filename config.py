import torch

if torch.cuda.is_available():
  DEVICE = torch.device("cuda:2")
else:
  DEVICE = torch.device("cpu")

# Filepaths

MODELS_BASEPATH = "models/"
LOGS_BASEPATH = "logs/"
DATASETS_BASEPATH = "/home/oskar/Documents/Datasets/"

EVAL_FREQUENCY = 100

# PAMAP2

## PAMAP2 Filepaths

PAMAP2_BASEPATH = "PAMAP2_Dataset/Protocol/"
PAMAP2_FILENAMES = [f"subject10{i}.dat" for i in range(1,10)]
PAMAP2_FILEPATHS = [DATASETS_BASEPATH + PAMAP2_BASEPATH + filename for filename in PAMAP2_FILENAMES]
PAMAP2_MIN_MAX_FILEPATH = DATASETS_BASEPATH + PAMAP2_BASEPATH + "min_max.json"
PAMAP2_PREPROCESSED_FILENAME_SUFFIX = ".preprocessed.npy"

PAMAP2_TRAIN_SET = [f"{DATASETS_BASEPATH}{PAMAP2_BASEPATH}subject10{i}.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}" for i in [1, 2, 3, 4, 7, 8, 9]]
PAMAP2_VAL_SET = [f"{DATASETS_BASEPATH}{PAMAP2_BASEPATH}subject105.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}"]
PAMAP2_TEST_SET = [f"{DATASETS_BASEPATH}{PAMAP2_BASEPATH}subject106.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}"]

## PAMAP2 Preprocessing

                     # timestamp, label, heart rate
PAMAP2_COLUMN_MASK = ([False,      True,  False] +
                     # temperature, 3D Accelerometer, 3D Accelerometer,    3D Gyroscope,     3D Magnetometer,  4D orientation
                      [False,       True, True, True, False, False, False, True, True, True, True, True, True, False, False, False, False] * 3) # 3 IMUS

PAMAP2_CURRENT_FREQUENCY = 100
PAMAP2_TARGET_FREQUENCY = 30

PAMAP2_LABEL_MASK = [1,2,3,4,5,6,7,12,13,16,17,24]
PAMAP2_LABEL_REMAPPING = {24: 0, 12: 8, 13: 9, 16: 10, 17: 11}
PAMAP2_LABEL_NAMES = {0: "rope jumping",
                      1: "lying",
                      2: "sitting",
                      3: "standing",
                      4: "walking",
                      5: "running",
                      6: "cycling",
                      7: "nordic walking",
                      8: "ascending stairs",
                      9: "descending stairs",
                      10: "vacuum cleaning",
                      11: "ironing"}

## PAMAP2 

PAMAP2 = {
  "TRAIN_SET_FILEPATH": DATASETS_BASEPATH + PAMAP2_BASEPATH + "windows.train.npz",
  "VAL_SET_FILEPATH":   DATASETS_BASEPATH + PAMAP2_BASEPATH + "windows.val.npz",
  "TEST_SET_FILEPATH":  DATASETS_BASEPATH + PAMAP2_BASEPATH + "windows.test.npz",

  "TRAIN_SET": PAMAP2_TRAIN_SET,
  "VAL_SET": PAMAP2_VAL_SET,
  "TEST_SET": PAMAP2_TEST_SET,

  "NAME": "PAMAP2 - CNN-IMU-2",
  "MODEL": "CNN_IMU",
  "NUM_SENSOR_CHANNELS": 27,
  "NUM_CLASSES": 12,
  "WINDOW_SIZE": 100,
  "STEP_SIZE": 22,
  "NUM_IMUS": 3,

  "EPOCHS": 12,
  "BATCH_SIZE": 50,
  "NUM_KERNELS": 64,
  "KERNEL_LENGTH": 5,
  "POOLING_LENGTH": 2,
  "DROPOUT": 0.5,
  "LEARNING_RATE": 0.0001,
  "RMS_DECAY": 0.95,
  "NOISE": 0.01
}

# Opportunity Locomotion

## Opportunity Locomotion filepaths

OPPORTUNITY_LOCOMOTION_BASEPATH = "Opportunity/dataset/"
OPPORTUNITY_LOCOMOTION_FILENAMES = {f"S{i}-ADL{j}.dat" for i in range(1,5) for j in range(1,6)} | {f"S{i}-Drill.dat" for i in range(1,5)}
OPPORTUNITY_LOCOMOTION_FILEPATHS = {DATASETS_BASEPATH + OPPORTUNITY_LOCOMOTION_BASEPATH + filename for filename in OPPORTUNITY_LOCOMOTION_FILENAMES}
OPPORTUNITY_LOCOMOTION_MIN_MAX_FILEPATH = DATASETS_BASEPATH + OPPORTUNITY_LOCOMOTION_BASEPATH + "locomotion.min_max.json"
OPPORTUNITY_LOCOMOTION_PREPROCESSED_FILENAME_SUFFIX = ".locomotion.preprocessed.npy"

OPPORTUNITY_LOCOMOTION_VAL_SET = {f"{DATASETS_BASEPATH}{OPPORTUNITY_LOCOMOTION_BASEPATH}S{i}-ADL3.dat{OPPORTUNITY_LOCOMOTION_PREPROCESSED_FILENAME_SUFFIX}" for i in [2,3]}
OPPORTUNITY_LOCOMOTION_TEST_SET = {f"{DATASETS_BASEPATH}{OPPORTUNITY_LOCOMOTION_BASEPATH}S{i}-ADL{j}.dat{OPPORTUNITY_LOCOMOTION_PREPROCESSED_FILENAME_SUFFIX}" for i in [2,3] for j in [4,5]}
OPPORTUNITY_LOCOMOTION_TRAIN_SET = {filepath + OPPORTUNITY_LOCOMOTION_PREPROCESSED_FILENAME_SUFFIX for filepath in OPPORTUNITY_LOCOMOTION_FILEPATHS} - OPPORTUNITY_LOCOMOTION_VAL_SET - OPPORTUNITY_LOCOMOTION_TEST_SET

## Opportunity Locomotion Preprocessing
                                  #  LABEL   ACCS                BACK                 RUA                  RLA                  LUA                  LLA                  L-SHOE                 R-SHOE
OPPORTUNITY_LOCOMOTION_COLUMN_MASK = [244] + list(range(2,38)) + list(range(38,47)) + list(range(51,60)) + list(range(64,73)) + list(range(77,86)) + list(range(90,99)) + list(range(109,115)) + list(range(125,131))
OPPORTUNITY_LOCOMOTION_COLUMN_MASK = [i-1 for i in OPPORTUNITY_LOCOMOTION_COLUMN_MASK]
OPPORTUNITY_LOCOMOTION_LABEL_MASK = [0,1,2,4,5]
OPPORTUNITY_LOCOMOTION_LABEL_REMAPPING = {5: 3}
OPPORTUNITY_LOCOMOTION_LABEL_NAMES = {0: "Null", 1: "Stand", 2: "Walk", 3: "Lie", 4: "Sit"}

## Opportunity Locomotion

OPPORTUNITY_LOCOMOTION = {
  "TRAIN_SET_FILEPATH": DATASETS_BASEPATH + OPPORTUNITY_LOCOMOTION_BASEPATH + "locomotion.train.npz",
  "VAL_SET_FILEPATH":   DATASETS_BASEPATH + OPPORTUNITY_LOCOMOTION_BASEPATH + "locomotion.val.npz",
  "TEST_SET_FILEPATH":  DATASETS_BASEPATH + OPPORTUNITY_LOCOMOTION_BASEPATH + "locomotion.test.npz",

  "TRAIN_SET": OPPORTUNITY_LOCOMOTION_TRAIN_SET,
  "VAL_SET": OPPORTUNITY_LOCOMOTION_VAL_SET,
  "TEST_SET": OPPORTUNITY_LOCOMOTION_TEST_SET,

  "NAME": "OPPORTUNITY_LOCOMOTION - CNN-IMU-2",
  "MODEL": "CNN_IMU",
  "NUM_SENSOR_CHANNELS": 93,
  "NUM_CLASSES": 5,
  "WINDOW_SIZE": 24,
  "STEP_SIZE": 12,
  "NUM_IMUS": 8, # TODO Has to be more complex since IMUS are not equally sized!!!
  "BRANCHES": [],

  "EPOCHS": 12,
  "BATCH_SIZE": 100,
  "NUM_KERNELS": 64,
  "KERNEL_LENGTH": 5,
  "POOLING_LENGTH": 2,
  "DROPOUT": 0.5,
  "LEARNING_RATE": 0.0001,
  "RMS_DECAY": 0.95,
  "NOISE": 0.01
}