import torch

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
else:
  DEVICE = torch.device("cpu")

# Filepaths

MODELS_BASEPATH = "models/"
LOGS_BASEPATH = "logs/"
DATASETS_BASEPATH = "/data/ozeino/"

EVAL_FREQUENCY = 100

# PAMAP2

## PAMAP2 Filepaths

PAMAP2_BASEPATH = "PAMAP2_Dataset/Protocol/"
PAMAP2_FILENAMES = [f"subject10{i}.dat" for i in range(1,10)]
PAMAP2_FILEPATHS = [DATASETS_BASEPATH + PAMAP2_BASEPATH + filename for filename in PAMAP2_FILENAMES]
PAMAP2_MIN_MAX_FILEPATH = DATASETS_BASEPATH + PAMAP2_BASEPATH + "min_max.json"
PAMAP2_PREPROCESSED_FILENAME_SUFFIX = ".preprocessed.npy"

PAMAP2_TRAIN_SET_FILEPATH = DATASETS_BASEPATH + PAMAP2_BASEPATH + "windows.train.npz"
PAMAP2_VAL_SET_FILEPATH = DATASETS_BASEPATH + PAMAP2_BASEPATH + "windows.val.npz"
PAMAP2_TEST_SET_FILEPATH = DATASETS_BASEPATH + PAMAP2_BASEPATH + "windows.test.npz"
PAMAP2_TRAIN_SET = [f"{DATASETS_BASEPATH}{PAMAP2_BASEPATH}subject10{i}.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}" for i in [1, 2, 3, 4, 7, 8, 9]]
PAMAP2_VAL_SET = [f"{DATASETS_BASEPATH}{PAMAP2_BASEPATH}subject105.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}"]
PAMAP2_TEST_SET = [f"{DATASETS_BASEPATH}{PAMAP2_BASEPATH}subject106.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}"]

## PAMAP2 Data Preprocessing

                     # timestamp, label, heart rate
PAMAP2_COLUMN_MASK = ([True,      True,  False] +
                     # temperature, 3D Accelerometer, 3D Accelerometer,    3D Gyroscope,     3D Magnetometer,  4D orientation
                      [False,       True, True, True, False, False, False, True, True, True, True, True, True, False, False, False, False] * 3) # 3 IMUS                    

PAMAP2_LABEL_COLUMN_INDEX = 1
PAMAP2_DATA_COLUMN_INDEX = 2

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
  "TRAIN_SET_FILEPATH": PAMAP2_TRAIN_SET_FILEPATH,
  "VAL_SET_FILEPATH": PAMAP2_VAL_SET_FILEPATH,
  "TEST_SET_FILEPATH": PAMAP2_TEST_SET_FILEPATH,

  "NAME": "PAMAP2 - CNN_IMU",
  "MODEL": "CNN_IMU",
  "NUM_SENSOR_CHANNELS": 27,
  "NUM_CLASSES": 12,
  "WINDOW_SIZE": 100,
  "STEP_SIZE": 22,
  "NUM_IMUS": 3,

  "EPOCHS": 12,
  "BATCH_SIZE": 10,
  "NUM_KERNELS": 64,
  "KERNEL_LENGTH": 5,
  "POOLING_LENGTH": 2,
  "DROPOUT": 0.5,
  "LEARNING_RATE": 0.001,
  "RMS_DECAY": 0.95
}
