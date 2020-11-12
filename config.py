import torch

if torch.cuda.is_available():
  DEVICE = torch.device("cuda:2")
else:
  DEVICE = torch.device("cpu")

DETERMINISTIC = True

if DETERMINISTIC:
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
else:
  torch.backends.cudnn.benchmark = True

# Filepaths

TEST_BASEPATH = "/data/ozeino/test/"
LOGS_BASEPATH = "/data/ozeino/logs_freeze/"
MODELS_BASEPATH = "/data/ozeino/models/"
DATASETS_BASEPATH = "/data/ozeino/"

TEST_REPETITIONS = 5
EVAL_PERIOD = 50
MAX_BATCH_SIZE = 1000

# PAMAP2

## PAMAP2 Filepaths

PAMAP2_BEST_Simple_CNN = MODELS_BASEPATH + "PAMAP2-Simple_CNN-LEARNING_RATE-0.001_best_wf1_0.9026_epoch_3_iteration_349.pt"
PAMAP2_BEST_CNN_IMU = MODELS_BASEPATH + "PAMAP2-CNN_IMU-LEARNING_RATE-0.001_best_wf1_0.9154_epoch_6_iteration_49.pt"

PAMAP2_BASEPATH = DATASETS_BASEPATH + "PAMAP2/Protocol/"
PAMAP2_FILENAMES = [f"subject10{i}.dat" for i in range(1,10)]
PAMAP2_FILEPATHS = [PAMAP2_BASEPATH + filename for filename in PAMAP2_FILENAMES]
PAMAP2_MIN_MAX_FILEPATH = PAMAP2_BASEPATH + "min_max.json"
PAMAP2_PREPROCESSED_FILENAME_SUFFIX = ".preprocessed.npy"

PAMAP2_TRAIN_SET = [f"{PAMAP2_BASEPATH}subject10{i}.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}" for i in [1, 2, 3, 4, 7, 8, 9]]
PAMAP2_VAL_SET = [f"{PAMAP2_BASEPATH}subject105.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}"]
PAMAP2_TEST_SET = [f"{PAMAP2_BASEPATH}subject106.dat{PAMAP2_PREPROCESSED_FILENAME_SUFFIX}"]

## PAMAP2 Preprocessing

                     # timestamp, label, heart rate
PAMAP2_COLUMN_MASK = ([False,      True,  False] +
                     # temperature, 3D Accelerometer, 3D Accelerometer,    3D Gyroscope,     3D Magnetometer,  4D orientation
                      [False,       True, True, True, False, False, False, True, True, True, True, True, True, False, False, False, False] * 3) # 3 IMUS

# TODO maybe don't downsample? since Order_Picking has also 100Hz freq 
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
  "TRAIN_SET_FILEPATH": PAMAP2_BASEPATH + "windows.train.npz",
  "VAL_SET_FILEPATH":   PAMAP2_BASEPATH + "windows.val.npz",
  "TEST_SET_FILEPATH":  PAMAP2_BASEPATH + "windows.test.npz",

  "TRAIN_SET": PAMAP2_TRAIN_SET,
  "VAL_SET": PAMAP2_VAL_SET,
  "TEST_SET": PAMAP2_TEST_SET,

  "NAME": "PAMAP2",
  "MODEL": "CNN_IMU",
  "NUM_SENSOR_CHANNELS": 27,
  "NUM_CLASSES": 12,
  "WINDOW_SIZE": 100,
  "STEP_SIZE": 22,
  "IMUS": 3,
  "BRANCHES": ["HAND", "CHEST", "ANKLE"],

  "EPOCHS": 12,
  "BATCH_SIZE": 50,
  "NUM_KERNELS": 64,
  "KERNEL_LENGTH": 5,
  "POOLING_LENGTH": 2,
  "POOLING_STRIDE": 2,
  "DROPOUT": 0.5,
  "LEARNING_RATE": 10**-3,
  "RMS_DECAY": 0.95,
  "NOISE": 0.01,
  "FREEZE": 0
}

# Opportunity Locomotion

## Opportunity Locomotion filepaths

OPPORTUNITY_LOCOMOTION_BEST_Simple_CNN = MODELS_BASEPATH + "OPPORTUNITY_LOCOMOTION-Simple_CNN-LEARNING_RATE-0.001_best_wf1_0.8652_epoch_6_iteration_499.pt"
OPPORTUNITY_LOCOMOTION_BEST_CNN_IMU = MODELS_BASEPATH + "OPPORTUNITY_LOCOMOTION-CNN_IMU-LEARNING_RATE-0.001_best_wf1_0.8825_epoch_4_iteration_299.pt"

OPPORTUNITY_LOCOMOTION_BASEPATH = DATASETS_BASEPATH + "Opportunity_Locomotion/dataset/"
OPPORTUNITY_LOCOMOTION_FILENAMES = {f"S{i}-ADL{j}.dat" for i in range(1,5) for j in range(1,6)} | {f"S{i}-Drill.dat" for i in range(1,5)}
OPPORTUNITY_LOCOMOTION_FILEPATHS = {OPPORTUNITY_LOCOMOTION_BASEPATH + filename for filename in OPPORTUNITY_LOCOMOTION_FILENAMES}
OPPORTUNITY_LOCOMOTION_MIN_MAX_FILEPATH = OPPORTUNITY_LOCOMOTION_BASEPATH + "locomotion.min_max.json"
OPPORTUNITY_LOCOMOTION_PREPROCESSED_FILENAME_SUFFIX = ".locomotion.preprocessed.npy"

OPPORTUNITY_LOCOMOTION_VAL_SET = {f"{OPPORTUNITY_LOCOMOTION_BASEPATH}S{i}-ADL3.dat{OPPORTUNITY_LOCOMOTION_PREPROCESSED_FILENAME_SUFFIX}" for i in [2,3]}
OPPORTUNITY_LOCOMOTION_TEST_SET = {f"{OPPORTUNITY_LOCOMOTION_BASEPATH}S{i}-ADL{j}.dat{OPPORTUNITY_LOCOMOTION_PREPROCESSED_FILENAME_SUFFIX}" for i in [2,3] for j in [4,5]}
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
  "TRAIN_SET_FILEPATH": OPPORTUNITY_LOCOMOTION_BASEPATH + "locomotion.train.npz",
  "VAL_SET_FILEPATH":   OPPORTUNITY_LOCOMOTION_BASEPATH + "locomotion.val.npz",
  "TEST_SET_FILEPATH":  OPPORTUNITY_LOCOMOTION_BASEPATH + "locomotion.test.npz",

  "TRAIN_SET": OPPORTUNITY_LOCOMOTION_TRAIN_SET,
  "VAL_SET": OPPORTUNITY_LOCOMOTION_VAL_SET,
  "TEST_SET": OPPORTUNITY_LOCOMOTION_TEST_SET,

  "NAME": "OPPORTUNITY_LOCOMOTION",
  "MODEL": "CNN_IMU",
  "NUM_SENSOR_CHANNELS": 93,
  "NUM_CLASSES": 5,
  "WINDOW_SIZE": 24,
  "STEP_SIZE": 12,
  "IMUS": [36, 9, 9, 9, 9, 9, 6, 6],
  "BRANCHES": ["ACCS", "BACK", "RUA", "RLA", "LUA", "LLA", "L-SHOE", "R-SHOE"],

  "EPOCHS": 12,
  "BATCH_SIZE": 100,
  "NUM_KERNELS": 64,
  "KERNEL_LENGTH": 5,
  "POOLING_LENGTH": 2,
  "POOLING_STRIDE": 1,
  "DROPOUT": 0.5,
  "LEARNING_RATE": 10**-3,
  "RMS_DECAY": 0.95,
  "NOISE": 0.01,
  "FREEZE": 0
}

# Opportunity Gestures

## Opportunity Gestures filepaths

OPPORTUNITY_GESTURES_BEST_Simple_CNN = MODELS_BASEPATH + "OPPORTUNITY_GESTURES-Simple_CNN-LEARNING_RATE-0.001_best_wf1_0.8907_epoch_9_iteration_349.pt"
OPPORTUNITY_GESTURES_BEST_CNN_IMU = MODELS_BASEPATH + "OPPORTUNITY_GESTURES-CNN_IMU-LEARNING_RATE-0.001_best_wf1_0.8948_epoch_9_iteration_249.pt"

OPPORTUNITY_GESTURES_BASEPATH = DATASETS_BASEPATH + "Opportunity_Gestures/dataset/"
OPPORTUNITY_GESTURES_FILENAMES = {f"S{i}-ADL{j}.dat" for i in range(1,5) for j in range(1,6)} | {f"S{i}-Drill.dat" for i in range(1,5)}
OPPORTUNITY_GESTURES_FILEPATHS = {OPPORTUNITY_GESTURES_BASEPATH + filename for filename in OPPORTUNITY_GESTURES_FILENAMES}
OPPORTUNITY_GESTURES_MIN_MAX_FILEPATH = OPPORTUNITY_GESTURES_BASEPATH + "gestures.min_max.json"
OPPORTUNITY_GESTURES_PREPROCESSED_FILENAME_SUFFIX = ".gestures.preprocessed.npy"

OPPORTUNITY_GESTURES_VAL_SET = {f"{OPPORTUNITY_GESTURES_BASEPATH}S{i}-ADL3.dat{OPPORTUNITY_GESTURES_PREPROCESSED_FILENAME_SUFFIX}" for i in [2,3]}
OPPORTUNITY_GESTURES_TEST_SET = {f"{OPPORTUNITY_GESTURES_BASEPATH}S{i}-ADL{j}.dat{OPPORTUNITY_GESTURES_PREPROCESSED_FILENAME_SUFFIX}" for i in [2,3] for j in [4,5]}
OPPORTUNITY_GESTURES_TRAIN_SET = {filepath + OPPORTUNITY_GESTURES_PREPROCESSED_FILENAME_SUFFIX for filepath in OPPORTUNITY_GESTURES_FILEPATHS} - OPPORTUNITY_GESTURES_VAL_SET - OPPORTUNITY_GESTURES_TEST_SET

## Opportunity Gestures Preprocessing
                                #  LABEL   ACCS                BACK                 RUA                  RLA                  LUA                  LLA                  L-SHOE                 R-SHOE
OPPORTUNITY_GESTURES_COLUMN_MASK = [250] + list(range(2,38)) + list(range(38,47)) + list(range(51,60)) + list(range(64,73)) + list(range(77,86)) + list(range(90,99)) + list(range(109,115)) + list(range(125,131))
OPPORTUNITY_GESTURES_COLUMN_MASK = [i-1 for i in OPPORTUNITY_GESTURES_COLUMN_MASK]
OPPORTUNITY_GESTURES_LABEL_MASK = [0, 406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511, 406508, 404508, 408512, 407521, 405506]
OPPORTUNITY_GESTURES_LABEL_REMAPPING = {
  406516: 1,
  406517: 2,
  404516: 3,
  404517: 4,
  406520: 5,
  404520: 6,
  406505: 7,
  404505: 8,
  406519: 9,
  404519: 10,
  406511: 11,
  404511: 12,
  406508: 13,
  404508: 14,
  408512: 15,
  407521: 16,
  405506: 17
}
OPPORTUNITY_GESTURES_LABEL_NAMES = {
  0: "Null",
  1: "Open Door 1",
  2: "Open Door 2",
  3: "Close Door 1",
  4: "Close Door 2",
  5: "Open Fridge",
  6: "Close Fridge",
  7: "Open Dishwasher",
  8: "Close Dishwasher",
  9: "Open Drawer 1",
  10: "Close Drawer 1",
  11: "Open Drawer 2",
  12: "Close Drawer 2",
  13: "Open Drawer 3",
  14: "Close Drawer 3",
  15: "Clean Table",
  16: "Drink from Cup",
  17: "Toggle Switch"
}

## Opportunity Gestures

OPPORTUNITY_GESTURES = {
  "TRAIN_SET_FILEPATH": OPPORTUNITY_GESTURES_BASEPATH + "gestures.train.npz",
  "VAL_SET_FILEPATH":   OPPORTUNITY_GESTURES_BASEPATH + "gestures.val.npz",
  "TEST_SET_FILEPATH":  OPPORTUNITY_GESTURES_BASEPATH + "gestures.test.npz",

  "TRAIN_SET": OPPORTUNITY_GESTURES_TRAIN_SET,
  "VAL_SET": OPPORTUNITY_GESTURES_VAL_SET,
  "TEST_SET": OPPORTUNITY_GESTURES_TEST_SET,

  "NAME": "OPPORTUNITY_GESTURES",
  "MODEL": "CNN_IMU",
  "NUM_SENSOR_CHANNELS": 93,
  "NUM_CLASSES": 18,
  "WINDOW_SIZE": 24,
  "STEP_SIZE": 12,
  "IMUS": [36, 9, 9, 9, 9, 9, 6, 6],
  "BRANCHES": ["ACCS", "BACK", "RUA", "RLA", "LUA", "LLA", "L-SHOE", "R-SHOE"],

  "EPOCHS": 12,
  "BATCH_SIZE": 100,
  "NUM_KERNELS": 64,
  "KERNEL_LENGTH": 5,
  "POOLING_LENGTH": 2,
  "POOLING_STRIDE": 1,
  "DROPOUT": 0.5,
  "LEARNING_RATE": 10**-3,
  "RMS_DECAY": 0.95,
  "NOISE": 0.01,
  "FREEZE": 0
}

# ORDER_PICKING_A

## ORDER_PICKING_A Filepaths

ORDER_PICKING_A_BASEPATH = DATASETS_BASEPATH + "Order_Picking/_DO/"
ORDER_PICKING_A_FILENAMES = [f"subject0{i:02d}.dat" for i in [4,11,17]]
ORDER_PICKING_A_FILEPATHS = [ORDER_PICKING_A_BASEPATH + filename for filename in ORDER_PICKING_A_FILENAMES]
ORDER_PICKING_A_MIN_MAX_FILEPATH = ORDER_PICKING_A_BASEPATH + "min_max.json"
ORDER_PICKING_A_PREPROCESSED_FILENAME_SUFFIX = ".preprocessed.npy"

ORDER_PICKING_A_TRAIN_VAL_SETS = [([f"subject0{i:02d}.dat", f"subject0{j:02d}.dat"], f"subject0{k:02d}.dat") for i,j,k in [(4,11,17), (4,17,11), (11,17,4)]]
ORDER_PICKING_A_TRAIN_VAL_SET_FILEPATHS = [(f"{ORDER_PICKING_A_BASEPATH}subject0{i:02d}.dat_subject0{j:02d}.dat.train.npz", f"{ORDER_PICKING_A_BASEPATH}subject0{k:02d}.dat.val.npz") for i,j,k in [(4,11,17), (4,17,11), (11,17,4)]]

ORDER_PICKING_A_LABEL_MASK = [1, 2, 3, 4, 5, 7, 9, 10]
ORDER_PICKING_A_LABEL_REMAPPING = {9: 0, 10: 6}
ORDER_PICKING_A_LABEL_NAMES = {
  0: "carrying",
  1: "unknown",
  2: "sensor flip",
  3: "walking",
  4: "searching",
  5: "picking",
  6: "acknowledge",
  7: "info"
}

## ORDER_PICKING_A 

ORDER_PICKING_A = {
  "TRAIN_SET_FILEPATH": ORDER_PICKING_A_TRAIN_VAL_SET_FILEPATHS[0][0],
  "VAL_SET_FILEPATH":   ORDER_PICKING_A_TRAIN_VAL_SET_FILEPATHS[0][1],

  "NAME": "ORDER_PICKING_A",
  "MODEL": "CNN_IMU",
  "NUM_SENSOR_CHANNELS": 27,
  "NUM_CLASSES": 8,
  "WINDOW_SIZE": 100,
  "STEP_SIZE": 1,
  "IMUS": 3,
  "BRANCHES": ["L-HAND", "R-HAND", "TORSO"],

  "EPOCHS": 20,
  "BATCH_SIZE": 100,
  "NUM_KERNELS": 64,
  "KERNEL_LENGTH": 5,
  "POOLING_LENGTH": 2,
  "POOLING_STRIDE": 2,
  "DROPOUT": 0.5,
  "Simple_CNN_LEARNING_RATE": 10**-5,
  "CNN_IMU_LEARNING_RATE": 10**-4,
  "RMS_DECAY": 0.95,
  "NOISE": 0.01,
  "FREEZE": 0
}

# ORDER_PICKING_B

## ORDER_PICKING_B Filepaths

ORDER_PICKING_B_BASEPATH = DATASETS_BASEPATH + "Order_Picking/_NP/"
ORDER_PICKING_B_FILENAMES = [f"subject0{i:02d}.dat" for i in [4,14,15]]
ORDER_PICKING_B_FILEPATHS = [ORDER_PICKING_B_BASEPATH + filename for filename in ORDER_PICKING_B_FILENAMES]
ORDER_PICKING_B_MIN_MAX_FILEPATH = ORDER_PICKING_B_BASEPATH + "min_max.json"
ORDER_PICKING_B_PREPROCESSED_FILENAME_SUFFIX = ".preprocessed.npy"

ORDER_PICKING_B_TRAIN_VAL_SETS = [([f"subject0{i:02d}.dat", f"subject0{j:02d}.dat"], f"subject0{k:02d}.dat") for i,j,k in [(4,14,15), (4,15,14), (14,15,4)]]
ORDER_PICKING_B_TRAIN_VAL_SET_FILEPATHS = [(f"{ORDER_PICKING_B_BASEPATH}subject0{i:02d}.dat_subject0{j:02d}.dat.train.npz", f"{ORDER_PICKING_B_BASEPATH}subject0{k:02d}.dat.val.npz") for i,j,k in [(4,14,15), (4,15,14), (14,15,4)]]

## ORDER_PICKING_B Preprocessing

ORDER_PICKING_B_LABEL_MASK = list(range(1,8))
ORDER_PICKING_B_LABEL_REMAPPING = {7: 0}
ORDER_PICKING_B_LABEL_NAMES = {
  0: "info",
  1: "unknown",
  2: "sensor flip",
  3: "walking",
  4: "searching",
  5: "picking",
  6: "scanning"
}

## ORDER_PICKING_B 

ORDER_PICKING_B = {
  "TRAIN_SET_FILEPATH": ORDER_PICKING_B_TRAIN_VAL_SET_FILEPATHS[0][0],
  "VAL_SET_FILEPATH":   ORDER_PICKING_B_TRAIN_VAL_SET_FILEPATHS[0][1],

  "NAME": "ORDER_PICKING_B",
  "MODEL": "CNN_IMU",
  "NUM_SENSOR_CHANNELS": 27,
  "NUM_CLASSES": 7,
  "WINDOW_SIZE": 100,
  "STEP_SIZE": 1,
  "IMUS": 3,
  "BRANCHES": ["L-HAND", "R-HAND", "TORSO"],

  "EPOCHS": 20,
  "BATCH_SIZE": 100,
  "NUM_KERNELS": 64,
  "KERNEL_LENGTH": 5,
  "POOLING_LENGTH": 2,
  "POOLING_STRIDE": 2,
  "DROPOUT": 0.5,
  "Simple_CNN_LEARNING_RATE": 10**-3,
  "CNN_IMU_LEARNING_RATE": 10**-3,
  "RMS_DECAY": 0.95,
  "NOISE": 0.01,
  "FREEZE": 0
}