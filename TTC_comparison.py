import csv 
import os
import matplotlib.pyplot as plt
import pandas as pd



curr_dir_path = os.path.dirname(os.path.realpath(__file__))
target_dir_path = os.path.join(curr_dir_path, 'results/')

listOfHeaders = ["Detector type", "Descriptor type", "Average TTC diff"]
listOfValues = []

lidar_TTC = []
with open(os.path.join(target_dir_path + "TTC_lidar_results.csv"), mode='r') as lidar_f:
    reader = csv.DictReader(lidar_f)
    result = {}
    for row in reader:
        for column, value in row.items():
            result.setdefault(column, []).append(value)
    #Get lidar TTC results
    lidar_TTC = result.get("TTC")



for root,dirs,files in os.walk(target_dir_path):
    for file in files:

        #Temporary list which holds values for a row
        tmp_row = []

        if(str(file) != "TTC_lidar_results.csv"):
            split_result = file.split("_")
            detector_name = split_result[0]
            descriptor_name = split_result[1]
            tmp_row.append(detector_name)
            tmp_row.append(descriptor_name)

            with open(os.path.join(target_dir_path, file), mode='r') as f:
                reader = csv.DictReader(f)
                result = {}
                for row in reader:
                    for column, value in row.items():
                        result.setdefault(column, []).append(value)

                #Get differences average between Lidar TTC and camera TTC for each combination of detector/descriptor
                camera_TTC = result.get("TTC")
                diff_TTC_accumulator = 0

                for i in range(len(camera_TTC)):
                    diff_TTC_accumulator += float(lidar_TTC[i]) - float(camera_TTC[i])
                avg_TTC_diff = round(abs(diff_TTC_accumulator / len(camera_TTC)), 3)
                tmp_row.append(avg_TTC_diff)

            listOfValues.append(tmp_row)

with open(curr_dir_path + "/assets/PF6.csv", 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(listOfHeaders)
    writer.writerows(listOfValues)


#Plot TTC graph for comparison
csvs = []
legends = []
for root,dirs,files in os.walk(target_dir_path):
    for file in files:
        df = pd.read_csv(target_dir_path + file)
        csvs.append(df)

        split_result = file.split("_")
        detector_name = split_result[0]
        descriptor_name = split_result[1]
        legends.append(detector_name + "_" + descriptor_name)


        
ax = csvs[0].plot(x="Frame", y="TTC", label=str(legends[0]))
for i in range (1,len(csvs)):
    csvs[i].plot(x="Frame", y="TTC", ax=ax, label=str(legends[i]) )

plt.show()
                
        