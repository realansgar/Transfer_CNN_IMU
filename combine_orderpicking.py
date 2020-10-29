import os
import re
import numpy as np
from config import ORDER_PICKING_A_BASEPATH, ORDER_PICKING_B_BASEPATH

idx_pattern = re.compile(r".*?(\d\d\d)\D*$") # matches three digits to determine the subject
name_pattern = re.compile(r"(.*)_\d\d\d\D*$") # matches the name of the sensor
sensor_lists = []
for folder in [ORDER_PICKING_A_BASEPATH, ORDER_PICKING_B_BASEPATH]:
  _, _, filenames = next(os.walk(folder))
  filenames = [filename for filename in filenames if filename.endswith(".csv")]
  filenames.sort() # -> Acc, Gyro, Magneto
  subject_idxs = sorted(list({idx_pattern.findall(filename)[0] for filename in filenames}))
  subjects = {subject_idx: [] for subject_idx in subject_idxs}
  for filename in filenames:
    subject_idx = idx_pattern.findall(filename)[0]
    subjects[subject_idx].append(filename)
  for subject_idx, subject in subjects.items():
    sensor_list = []
    subject_data = []
    for sensor in subject:
      sensor_list.append(name_pattern.findall(sensor)[0])
      data = np.loadtxt(folder + sensor, delimiter=",")
      if len(subject_data) == 0:
        subject_data.append(np.arange(0.01, len(data) * 0.01, 0.01).reshape((-1,1)))
        subject_data.append(data[:,[4]])
      elif len(np.argwhere((subject_data[1] != data[:,[4]]))) > 7:
        print(sensor)
        print(np.argwhere((subject_data[1] != data[:,[4]])))
        raise Exception("not matching data")
      subject_data.append(data[:,1:4])

    # some sensors recorded one value more than the others, remove the overhanging values
    min_length = min([len(x) for x in subject_data])
    subject_data = [x[:min_length] for x in subject_data]

    subject_data = np.concatenate(subject_data, axis=1)
    format_list = ["%.2f", "%d"] + ["%.10g"] * 27
    np.savetxt(f"{folder}subject{subject_idx}.dat", subject_data, fmt=format_list)
  sensor_lists.append(sensor_list)

  for x in sensor_lists:
    for y in sensor_lists:
      if x != y:
        raise "Alarm!"

  lines = ["Data columns:\n\n", "Column: 1 sec\n", "Column: 2 label\n"]
  i = 3
  for sensor in sensor_lists[0]:
    lines += [f"Column: {i} {sensor} x\n", f"Column: {i+1} {sensor} x\n", f"Column: {i+2} {sensor} x\n"]
    i += 3
  with open(folder + "column_names.txt", "w") as f:
    f.writelines(lines)
