import csv
import os
import json
from datetime import datetime,timezone,timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np



def plot_temps(name =None, start_date = "2025-04-09T15:00:48.000Z", csv_file_path = "C:/sfg/temperaratues_chucks(in).csv",
                results_path ="C:/sfg/results", cold = True, n_test = 10):
    """Name       ------- if None, data taken for all components
       Start_date ------- actually takes 8 sencond sooner start_date to track all needed tests
       Cold       ------- if True, 1. test to plot is when chuck temp is in range [-35.2 ,  -34.8] if False then in range [19.8 , 20.2] 
       n_tests    ------- number of thermal cycles"""
    rows = []
    with open(csv_file_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # přeskočí první řádek (hlavičku)

        for r in reader:
            ts = int(r[0]) / 1000               # převod z ms na s
            dt = datetime.fromtimestamp(ts, tz = timezone.utc)  # datetime objekt s UTC zónou
            temps = [float(x) for x in r[1:]]   # teploty jako float
            rows.append([dt] + temps)


    dates = []
    temp_y = []
    temp_x = []

    for filename in os.listdir(results_path):
        path = os.path.join(results_path, filename)
        

        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        date_str = data.get("date", "")
        if not date_str:
                continue
        if name == None:
            date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            dates.append(date_obj)
            temp_y.append(data.get('properties', {}).get("DCS", {}).get("AMAC_NTCy", {}))
            temp_x.append(data.get('properties', {}).get("DCS", {}).get("AMAC_NTCx", {}))

        elif data.get("properties", "").get("det_info",{}).get("name","") == name:
            date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            dates.append(date_obj)
            temp_y.append(data.get('properties', {}).get("DCS", {}).get("AMAC_NTCy", {}))
            temp_x.append(data.get('properties', {}).get("DCS", {}).get("AMAC_NTCx", {}))
        else:
            continue
    start_date = datetime.strptime(start_date,
                        "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    print(dates)
    plot_dates = []
    plot_temps_chuck1 = []
    plot_temps_chuck2 = []
    plot_temps_chuck3 = []
    plot_temps_chuck4 = []
    plot_temps_NTCX = []
    plot_temps_NTCY = []
    counter = 0
    data_list = sorted(zip(dates, temp_x, temp_y), key=lambda x: x[0])
    dates_sorted, temp_x_sorted, temp_y_sorted = zip(*data_list)
    dates_sorted  = list(dates_sorted)
    temp_x_sorted = list(temp_x_sorted)
    temp_y_sorted = list(temp_y_sorted)
    delta = timedelta(seconds=10)




    for idx, d in enumerate(dates_sorted):
        if d <= start_date - delta:
            continue
        if counter == 10:
            break

        # najdeme odpovídající řádek v rows podle data
        matching_rows = [r for r in rows if d-delta <= r[0] <= d + delta] 
        if not matching_rows:
            continue
        row = matching_rows[0]

        if -35.2 <= row[1] <= -34.8 and cold == True:
            plot_temps_chuck1.append(row[1])
            plot_temps_chuck2.append(row[2])
            plot_temps_chuck3.append(row[3])
            plot_temps_chuck4.append(row[4])
            plot_dates.append(row[0])
            plot_temps_NTCX.append(temp_x_sorted[idx])
            plot_temps_NTCY.append(temp_y_sorted[idx])
            if cold:
                counter += 1
            cold = False

        elif 19.8 <= row[1] <= 20.2 and cold == False:
            plot_temps_chuck1.append(row[1])
            plot_temps_chuck2.append(row[2])
            plot_temps_chuck3.append(row[3])
            plot_temps_chuck4.append(row[4])
            plot_dates.append(row[0])
            plot_temps_NTCX.append(temp_x_sorted[idx])
            plot_temps_NTCY.append(temp_y_sorted[idx])
            cold = True

        elif counter >=1:
            plot_temps_chuck1.append(row[1])
            plot_temps_chuck2.append(row[2])
            plot_temps_chuck3.append(row[3])
            plot_temps_chuck4.append(row[4])
            plot_dates.append(row[0])
            plot_temps_NTCX.append(temp_x_sorted[idx])
            plot_temps_NTCY.append(temp_y_sorted[idx])

    print(plot_dates)
    colors = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(18,6))
    ax.plot(plot_dates, plot_temps_chuck1, label="chuck_1", color = colors(0), linewidth = 2)
    ax.plot(plot_dates, plot_temps_chuck2, label="chuck_2", color = colors(1), linewidth = 2)
    ax.plot(plot_dates, plot_temps_chuck3, label="chuck_3", color = colors(2), linewidth = 2)
    ax.plot(plot_dates, plot_temps_chuck4, label="chuck_4", color = colors(3), linewidth = 2)
    plot_temps_NTCX_smooth = np.convolve(plot_temps_NTCX, np.ones(3)/3, mode='same')
    ax.plot(plot_dates, plot_temps_NTCX_smooth, label="NTCX (smooth)", linewidth=1.5, linestyle='--', color = colors(4))
    plot_temps_NTCY_smooth = np.convolve(plot_temps_NTCY, np.ones(3)/3, mode='same')
    ax.plot(plot_dates, plot_temps_NTCY_smooth, label="NTCY (smooth)", linewidth=1.5, linestyle='-.', color = colors(9))
    

    # ---- KLÍČOVÁ ČÁST ----
    ax.grid(True,"both")
    ax.legend()
    ax.set_xlabel("čas")
    ax.set_ylabel("teplota [°C]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    fig.autofmt_xdate()   # hezky pootočí popisky
    plt.show()
    return rows, data_list

rows , data_list = plot_temps(name=None)



                        
               
                
        


