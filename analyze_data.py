import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from datetime import datetime, date



def histogram_with_peaks(values, output_file, threshold=1.2, mean=None, y_label="Frequency", title = None, filename = None):

    counts, bin_edges = values, np.arange(0, len(values) + 1, 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peak_indices, _ = find_peaks(counts, height=mean * threshold)
    if len(peak_indices) == 0:
        return None

    def gaussian(x, amp, mu, sigma, baseline=mean):
        return baseline + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    fitted_params = []
    for peak in peak_indices:
        left = max(peak - 5, 0)
        right = min(peak + 6, len(bin_centers))
        x_fit = bin_centers[left:right]
        y_fit = counts[left:right]
        amp_guess = counts[peak]
        mu_guess = bin_centers[peak]
        sigma_guess = (bin_edges[1] - bin_edges[0]) * 3

        try:
            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[amp_guess, mu_guess, sigma_guess])
            fitted_params.append(popt)
        except RuntimeError:
            continue

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], alpha=0.6)
    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    for i, (amp, mu, sigma) in enumerate(fitted_params):
        plt.plot(x_smooth, gaussian(x_smooth, amp, mu, sigma))
    plt.ylim(0, max(counts) * 1.1)
    plt.xlabel("Channel Number")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_file,f"{filename}_histogram"), dpi=300)
    plt.close()

    return fitted_params, bin_centers, counts, bin_edges



def detect_failed_channels(values, mean, expected_value = 600, threshold=1.2):
    failed = []
    failed_expected_value = []
    consecutive_groups = []
    counter = 0
    fails = [False,False]
    threshold = (threshold)*mean
    for i, val in enumerate(values):
        if abs(val) > threshold:
            failed.append(i+1)
            if i in failed:
                counter += 1
            else:
                counter = 1
            if counter == 8:
                consecutive_groups.append(list(range(i - 6, i + 2)))
            elif counter > 8:
                consecutive_groups[-1].append(i+1)
        else:
            counter = 0
        if abs(val) > expected_value * 1.5:
            failed_expected_value.append(i+1)

    if len(consecutive_groups) > 0:
        fails[0] = True

    if len(failed_expected_value) > 0:
        fails[1] = True 

    return failed, consecutive_groups, failed_expected_value, fails



def to_date(obj) -> date:
    """Převede string nebo datetime na čistý datetime.date"""
    if isinstance(obj, date) and not isinstance(obj, datetime):
        return obj
    if isinstance(obj, datetime):
        return obj.date()
    if isinstance(obj, str):
        return datetime.strptime(obj[:10], "%Y-%m-%d").date()
    raise ValueError(f"Neumím převést {obj} na date")

def load_json_files(folder_path, name_filter, date_start, date_end):
    # Převedeme vstupní hranice hned na začátku
    date_start = to_date(date_start)
    date_end = to_date(date_end)

    data_list = []
    filenames = []

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)

        if not os.path.isfile(path):
            continue

        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        # Zpracování jména
        name = data.get("properties", "").get("det_info",{}).get("name","")
        name = name[2:] if name.startswith("SN") else name

       


        # Zpracování datumu
        date_str = data.get("date", "")
        if not date_str:
            continue

        date_obj = to_date(date_str)
        

        # Filtr podle jména a data
        if name == name_filter and date_start <= date_obj <= date_end:
            data_list.append(data)
            filenames.append(filename)


    print(f"Loaded {len(data_list)} files for {name_filter} between {date_start} and {date_end}")
    # print([data.get("date", "") for data in data_list])
    # print(filenames)
    os.makedirs(os.path.join(folder_path, "used_files"), exist_ok=True)
    for filename in filenames:
        src = os.path.join(folder_path, filename)
        dst = os.path.join(os.path.join(folder_path, "used_files"), filename)

        if os.path.exists(src):
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
            # print(f"Copied: {filename}")
        else:
            print(f"File not found: {filename}")
    if len(data_list) == 0:
        raise ValueError(f"No data found for {name_filter} between {date_start} and {date_end}")
    return data_list



def analyze_channels(all_channels):
    n_channels = np.arange(1, len(all_channels) + 1)
    slope, intercept, *_ = linregress(n_channels, all_channels)
    y_fit = slope * n_channels + intercept
    residuals = all_channels - y_fit
    fit_err = np.sqrt(np.sum(residuals ** 2) / (len(all_channels) - 2))
    return np.mean(all_channels), fit_err, y_fit



def save_results_to_json(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)



def plot_summary(means, errors, mask, labels, y_label, title, title_short, output_file):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x_pos = np.arange(len(means))
    ax.errorbar(x_pos, means, yerr=errors, fmt='o', capsize=5, label="Mean ± fit error")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel(y_label)
    ax.set_title(f"Mean values with fit errors, {title_short}")
    ax.set_ylim(min(means) * 0.6, math.ceil(max(means) / 100) * 100)
    ax.set_xlim(-1, len(means))
    ax.grid(True)
    mask = np.array(mask)
    x_pos_w = x_pos[mask == 1]
    means_w = np.array(means)[mask == 1]
    x_pos_c = x_pos[mask == 0]
    means_c = np.array(means)[mask == 0]
    if len(x_pos_w) > 0:
        mw, bw, *_ = linregress(x_pos_w, means_w)
        ax.plot(x_pos_w, mw * x_pos_w + bw, linestyle='-', label="fit (warm)")
    if len(x_pos_c) > 0:
        mc, bc, *_ = linregress(x_pos_c, means_c)
        ax.plot(x_pos_c, mc * x_pos_c + bc, linestyle='-', label="fit (cold)")
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"{title}", fontsize=16)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

def plot_boxplot(data: list, temps: list, save_path: str, result_name: str, info_lines: tuple, yname: str, failed_indices: list):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    colors = []                       
    for t in temps[:len(data)]:  
        if t > 0:
            colors.append("#CC3B06")  
        else:
            colors.append('#4D96FF')  
    
    box = plt.boxplot(data, 
                    patch_artist=True,
                    showfliers=False,
                    widths=0.7)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(box[element], color='#2D4059', linewidth=1.5)
    
    # mark failed tests
    for i, (patch, median_line) in enumerate(zip(box['boxes'], box['medians'])):
        if (i+1) in failed_indices:
            patch.set_hatch('////')
            patch.set_edgecolor('black')
            median_line.set_linewidth(3)
    
    y_min, y_max = ax.get_ylim()
   
    text_params = {
        'fontsize': 12,
        'ha': 'left',
        'va': 'bottom',
        'bbox': dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'),
        'linespacing': 1.5
    }
    base_y = y_max + 0.1*(y_max - y_min)  
    ax.text(0.8,  
        base_y,
        info_lines[0],
        **text_params)
    ax.text(0.8,
        base_y - 0.06*(y_max - y_min),
        info_lines[1],
        **text_params)
    ax.text(0.8,
        base_y - 0.12*(y_max - y_min),
        info_lines[2],
        **text_params)
    new_ymax = base_y + 0.05*(y_max - y_min)
    ax.set_ylim(y_min, max(y_max, new_ymax))

    legend_elements = [
        Patch(facecolor='#CC7306', edgecolor='#2D4059', label='Warm Test (T > 0℃)'),
        Patch(facecolor='#4D96FF', edgecolor='#2D4059', label='Cold Test (T < 0℃)'),
        Patch(facecolor='white', edgecolor='black', hatch='////', label='Failed Test'),
        # line2D([0], [0], marker='^',color='w',markerfacecolor='#2A9D8F',markersize=15,label='Shunted Tests')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title(f"Long Term Test: {result_name}", fontsize=16, pad=1, y=1.02)
    plt.xlabel('Test Sequence', fontsize=14)
    plt.ylabel(f"{yname}", fontsize=14)
    plt.xticks(range(1, len(data)+1), [f"T{i:02d}" for i in range(1, len(data)+1)], rotation=45)
    
    for i, test_data in enumerate(data):
        median = np.median(test_data)
        plt.text(i+1, median, f'{median:.2f}', 
                horizontalalignment='center',
                fontsize=8)
    
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved：{save_path}")

def plot_boxplot_on_ax(ax, data, temps, yname, failed_indices):
    colors = ['#CC3B06' if t > 0 else '#4D96FF' for t in temps[:len(data)]]

    box = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.7
    )

    # barvy
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(box[element], color='#2D4059', linewidth=1.5)

    # mark failed
    for i, (patch, median_line) in enumerate(zip(box['boxes'], box['medians'])):
        if (i + 1) in failed_indices:
            patch.set_hatch('////')
            median_line.set_linewidth(3)

    # popisky
    ax.set_xlabel('Test Sequence')
    ax.set_ylabel(yname)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([f"T{i:02d}" for i in range(1, len(data) + 1)], rotation=45)

    y_min, y_max = ax.get_ylim()
    base_y = y_max + 0.1 * (y_max - y_min)


    new_ymax = base_y + 0.05 * (y_max - y_min)
    ax.set_ylim(y_min, max(y_max, new_ymax))

    
    legend_elements = [
        Patch(facecolor='#CC7306', edgecolor='#2D4059', label='Warm Test (T > 0℃)'),
        Patch(facecolor='#4D96FF', edgecolor='#2D4059', label='Cold Test (T < 0℃)'),
        Patch(facecolor='white', edgecolor='black', hatch='////', label='Failed Test'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    for i, test_data in enumerate(data):
        median = np.median(test_data)
        ax.text(i+1, median, f'{median:.2f}', 
                horizontalalignment='center',
                fontsize=8)


    ax.grid(True, linestyle='--', alpha=0.6, axis='y')

def plotfailed_per_test(n_failed_channels,n_tests,n_channels ,output_file,info_lines):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.bar(range(1, n_tests + 1), n_failed_channels, color='#E76F51')
    plt.xlabel('Test Sequence')
    plt.xticks(range(1, n_tests + 1), [f"T{i:02d}" for i in range(1, n_tests + 1)], rotation=45)
    plt.ylabel('Number of Failed Channels')
    plt.axhline(y=math.ceil(n_channels*0.01), color='red', linestyle='--', label = "Threshold for failed test")
    plt.title('Number of Failed Channels per Test')

    text_params = {
        'fontsize': 12,
        'ha': 'left',
        'va': 'bottom',
        'bbox': dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'),
        'linespacing': 1.5
    }
    y_min, y_max = ax.get_ylim()
    base_y = y_max + 0.1*(y_max - y_min)  
    ax.text(0.8,  
        base_y,
        info_lines[0],
        **text_params)
    ax.text(0.8,
        base_y - 0.06*(y_max - y_min),
        info_lines[1],
        **text_params)
    new_ymax = base_y + 0.05*(y_max - y_min)
    ax.set_ylim(y_min, max(y_max, new_ymax))
    
    plt.grid(
            True,              
            which='both',     
            axis='both',              
            linestyle='--',            
            linewidth=0.8,
            color='gray',
            alpha=0.6)

    
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    

    
    
def analyze_single(variables, folder_path_up, name, date_start, date_end, folder_path_down = None, per_chip=False,
                      threshold=1.2, plot_hist = False,n_chips = 6):
    if folder_path_down == None:
        folder_path_down = folder_path_up
    data_list = load_json_files(folder_path_up, name, date_start=date_start, date_end=date_end)
    sorted_list = sorted(data_list,key=lambda d: datetime.fromisoformat(d["date"].replace("Z", "+00:00")))

    results = {"component": name, "date_range": {"start": date_start, "end": date_end}}
    folder_path_up = os.path.abspath(folder_path_up)
    folder_path_down = os.path.abspath(folder_path_down)
    
    for var in variables:

        if var == "innse_away":
            y_label = "Input Noise (ENC)"
            title = f"Input Noise for {name} away"
            title_short = "Input Noise away"
        elif var == "innse_under":
            y_label = "Input Noise (ENC)"
            title = f"Input Noise for {name} under"
            title_short = "Input Noise under"
        elif var == "gain_away":
            y_label = "Gain (MV/fC)"
            title = f"Gain for {name} away"
            title_short = "Gain away"
        elif var == "gain_under":
            y_label = "Gain (MV/fC)"
            title = f"Gain for {name} under"
            title_short = "Gain under"
        elif var == "vt50_away":
            y_label = "Vt50 (mV)"
            title = f"Vt50 for {name} away"
            title_short = "Vt50 away"
        elif var == "vt50_under":
            y_label = "Vt50 (mV)"
            title = f"Vt50 for {name} under"
            title_short = "Vt50 under"
        else:
            y_label = var
            title = var
            title_short = var

        

        if not per_chip:
            means, errors, test_dates= [], [], []
            
            all_tests = [] #Zde sbíráme data pro každý test (seznam seznamu; každý vnitřní seznam je jeden test)
            all_temps = [] #Zde sbíráme teploty pro každý test
            all_failed, all_failed_groups, all_expected_value, all_fails = [], [], [], []

            for data in sorted_list:
                measurements = data.get('results', {}).get(var, {}) # Získání hodnot pro danou proměnnou a jedno měření
                if not measurements:
                    continue

                temp = data.get('properties', {}).get("DCS", {}).get("AMAC_NTCy", {}) # Získání teploty
                all_temps.append(temp) 
                file_date = data.get('date')
                test_dates.append(file_date)
                all_channels = [] # Spojení všech kanálů z 6 čipů do jednoho seznamu

                for chip in measurements:
                    all_channels.extend(chip)
                all_tests.append(all_channels)
                mean_val, fit_err, y_fit = analyze_channels(np.array(all_channels))
                means.append(mean_val)
                errors.append(fit_err)

                if plot_hist:
                    titlefiledate = file_date.replace(":","-")
                    print(titlefiledate)

                    histogram_with_peaks(all_channels,os.path.join(folder_path_down,"histograms"), threshold = threshold,
                                            mean=np.mean(all_channels), y_label=y_label, title = f"{title_short} hist, for {name} on {file_date}",filename=f"{var}_{name}_{titlefiledate[:-5]}")
                failed, failed_groups, expected_value, fails = detect_failed_channels(all_channels, mean= mean_val,threshold=1.2)
                all_failed.append(failed)
                all_failed_groups.append(failed_groups)
                all_expected_value.append(expected_value)
                all_fails.append(fails)
                # failed --> list of failed channels
                # failed_groups --> list of lists of consecutive failed channels
                # expected_value --> list of channels failed expected value
                # fails --> [bool for consecutive fails, bool for expected value fails]
            
            result_msg = []
            for msg in all_fails:
                if msg[0] and msg[1]:
                    result_msg.append("failed for 8 and more consecutive failed channels and for atleast one " \
                                        "channel noise larger that 1.5 times expected value")
                elif msg[0]:
                    result_msg.append("failed for 8 and more consecutive failed channels")
                elif msg[1]:
                    result_msg.append("failed for channel noise larger that 1.5 times expected value")
                else:
                    result_msg.append("Passed")
                

            results[var] = {
                "means": means,
                "errors": errors,
                "failed_channels": all_failed,
                "failed_groups": all_failed_groups,
                "expected_value_fails": all_expected_value,
                "result":{d: m for d, m in zip(test_dates, result_msg)}
            }

            plot_boxplot(all_tests, all_temps, os.path.join(folder_path_down,f"{var}_boxplot_{name}_{date_start}-{date_end}.png"),
                         result_name=name,
                         info_lines=(
                             f"name: {name}",
                             f"Threshold for failed channel: {threshold} x mean",
                             f"Failed channels marked with '////' hatch"
                         ),
                         yname=y_label,
                         failed_indices=failed)
            
            n_failed_channels = [len(all_failed[i]) for i in range(len(all_failed))]  # seznam počtu failnutych kanalu pro vsechny testy
            plotfailed_per_test(n_failed_channels,n_tests=len(all_failed),n_channels = len(all_tests[0]),
                                 output_file=os.path.join(folder_path_down,f"{var}_failed_channels_{name}_{date_start}-{date_end}.png"),
                                 info_lines=(
                                    f"name: {name}_{var}",
                                    f"Threshold for failed channel: {threshold} x mean"))

           
            
        else:
            # per-chip vykreslování
            all_means = []
            all_errors = []
            all_test_chips = []  # Namerene hodnoty se ukladaji tak, ze se prvni ulozi hodnoty pro jeden chip z ruznych testu
                                 # a pak se pokracuje na dalsi chip
            used_file_dates = []
            
            failed, failed_groups, expected_value, fails = [], [], [], []
            all_temps = []
            len_used_data = 0

            for i in range(n_chips):
                chip_means, chip_errors = [], []
                for data in sorted_list:
                    measurements = data.get('results', {}).get(var, {})
                    if not measurements:
                        continue
                    temp = data.get('properties', {}).get("DCS", {}).get("AMAC_NTCy", {})
                    all_temps.append(temp)
                    file_date = data.get('date')
                    if file_date not in used_file_dates:
                        used_file_dates.append(file_date)
                    len_used_data += 1
                    chip_values = measurements[i]
                   
                    all_test_chips.append(chip_values)
                    
                    mean_val, fit_err, _ = analyze_channels(np.array(chip_values))
                    chip_means.append(mean_val)
                    
                    chip_errors.append(fit_err)
                    all_means.append(mean_val)
                    all_errors.append(fit_err)
                    failed_chip, failed_groups_chip, expected_value_chip, fails_chip = detect_failed_channels(chip_values, mean= mean_val, threshold=1.2)
                    failed.append(failed_chip)
                    failed_groups.append(failed_groups_chip)
                    expected_value.append(expected_value_chip)
                    fails.append(fails_chip)

                    if plot_hist:
                        histogram_with_peaks(chip_values, os.path.join(folder_path_down,"histograms"), threshold, mean=np.mean(chip_values), y_label=y_label,
                                                title = f"{title_short} hist,for {name} on {test_dates[j]}, chip {i + 1}", filename=f"{var}_{name}_{test_dates[j][:-5].replace(':','-')}_chip_{i+1}")
            
            
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.ravel() 
            n_tests = len_used_data//6

            for i in range(n_chips):
                chip_data = all_test_chips[i*n_tests:(i+1)*n_tests]
                failed_chip = failed[i*n_tests:(i+1)*n_tests]

                plot_boxplot_on_ax(
                    axs[i],
                    data=chip_data,
                    temps=all_temps,
                    yname=y_label,
                    failed_indices = failed_chip)

            fig.suptitle(f"Long Term Test: {name}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(folder_path_down,f"{var}_boxplot_{name}_{date_start}-{date_end}_allchips.png"),dpi=300)
            plt.close(fig)

            n_failed_channels = []
            
            for i in range(n_tests):
                counter = 0
                for j in range(n_chips):
                    counter += len(failed[i + j*n_tests])
                n_failed_channels.append(counter)
            
            plotfailed_per_test(n_failed_channels,n_tests, len(all_test_chips[0])*n_chips,
                                os.path.join(folder_path_down,f"{var}_failed_channels_{name}_{date_start}-{date_end}_allchips.png"),
                                info_lines=(
                                    f"name: {name}_{var}",
                                    f"Threshold for failed channel: {threshold} x mean"))
            

            

            results[var] = {f"Chip_{i+1}": {"means": all_means[i*n_tests:(i+1)*n_tests], "errors": all_errors[i*n_tests:(i+1)*n_tests], "failed_channels": failed[i*n_tests:(i+1)*n_tests],
                        "failed_groups": failed_groups[i*n_tests:(i+1)*n_tests], "expected_value_fails": expected_value[i*n_tests:(i+1)*n_tests]}for i in range(6)}
        
            results[var]["result"] = {}
            result_msg = []
            for msg in fails:
                if msg[0] and msg[1]:
                    result_msg.append("failed for 8 and more consecutive failed channels and for atleast one " \
                                        "channel noise larger that 1.5 times expected value")
                elif msg[0]:
                    result_msg.append("failed for 8 and more consecutive failed channels")
                elif msg[1]:
                    result_msg.append("failed for channel noise larger that 1.5 times expected value")
                else:
                    result_msg.append("Passed")
            
            for i in range(n_chips):
                results[var]["result"][f"chip{i+1}"] = {d: m for d, m in zip(used_file_dates, result_msg[i*n_tests:(i+1)*n_tests])}
            

           

    output_file = os.path.join(folder_path_down, f"{name}_{date_start}-{date_end}_results.json")
    save_results_to_json(results, output_file)
    return results

def analyze_and_plot(variables, folder_path_up, component, date_start, date_end, folder_path_down = None, per_chip=False,
                      threshold=1.2, plot_hist = False, under = True, away = True, H0 = True, H1 = True):
    
    if per_chip:
        os.makedirs(os.path.join(folder_path_down, f"{component}_{date_start}-{date_end}_single"), exist_ok=True)
        os.makedirs(os.path.join(folder_path_down,f"{component}_{date_start}-{date_end}_single","histograms"), exist_ok=True)
        folder_path_down = f"{folder_path_down}/{component}_{date_start}-{date_end}_single"
    else:
        os.makedirs(os.path.join(folder_path_down, f"{component}_{date_start}-{date_end}"), exist_ok=True)
        os.makedirs(os.path.join(folder_path_down,f"{component}_{date_start}-{date_end}","histograms"), exist_ok=True)
        folder_path_down = f"{folder_path_down}/{component}_{date_start}-{date_end}"


    if H0 and H1:
        name = [f"{component}_H0", f"{component}_H1"]
    elif H0:
        name = [f"{component}_H0"]
    elif H1:
        name = [f"{component}_H1"]
    else:
        name = [f"{component}"]

    if "Input Noise" in variables:
        variables.remove("Input Noise")
        variables.append("innse")
    if "Gain" in variables:
        variables.remove("Gain")
        variables.append("gain")
    if "Vt50" in variables:
        variables.remove("Vt50")
        variables.append("vt50")
    
    if under and away:
        variables = [f"{var}_under" for var in variables] + [f"{var}_away" for var in variables]
    elif under:
        variables = [f"{var}_under" for var in variables]
    elif away:
        variables = [f"{var}_away" for var in variables]
    print(name)

    for name in name:
        analyze_single(variables, folder_path_up, name, date_start,date_end, folder_path_down, per_chip, threshold, plot_hist)



if __name__ == "__main__":
    analyze_and_plot(
        variables=["Input Noise"],
        folder_path_up=r"C:/sfg/json",
        folder_path_down = r"C:/sfg/Code/Graphs",
        component="20USEH40000148",
        date_start="2025-07-02",
        date_end="2025-07-03",
        per_chip=True,
        plot_hist = False,
        under = True,
        away = True,
        H0 = True,
        H1 = True
    )


