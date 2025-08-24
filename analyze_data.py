import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def plot_histogram_with_peaks(values, threshold=1.2, n_channels=128, mean=None, y_label="Frequency", title = None):

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
    plt.show()

    return fitted_params, bin_centers, counts, bin_edges



def detect_failed_channels(values, mean, threshold=0.2):
    failed = []
    consecutive_groups = []
    counter = 0
    threshold = (1+threshold)*mean
    for i, val in enumerate(values):
        if abs(val) < threshold:
            failed.append(i)
            if i - 1 in failed:
                counter += 1
            else:
                counter = 1
            if counter == 8:
                consecutive_groups.append(list(range(i - 7, i + 1)))
            elif counter > 8:
                consecutive_groups[-1].append(i)
        else:
            counter = 0
    return failed, consecutive_groups



def load_json_files(folder_path, name_filter, date_filter):
    files = [f for f in os.listdir(folder_path) if f.endswith('.json') and f.startswith(name_filter) and date_filter in f]
    data_list = []
    for file in files:
        path = os.path.join(folder_path, file)
        with open(path, 'r') as f:
            data = json.load(f)
        data_list.append(data)
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



def analyze_and_plot(variables, folder_path_up, name, date, folder_path_down = None, per_chip=False, threshold=1.01, plot_hist = False):
    if folder_path_down == None:
        folder_path_down = folder_path_up
    data_list = load_json_files(folder_path_up, name, date)
    results = {"component": name, "date": date}

    for var in variables:
        if var == "innse_away":
            y_label = "Input Noise (ENC)"
            title = f"Input Noise for {name} away on {date}"
            title_short = "Input Noise away"
        elif var == "gain_away":
            y_label = "Gain (MV/fC)"
            title = f"Gain for {name} away on {date}"
            title_short = "Gain away"
        elif var == "gain_under":
            y_label = "Gain (MV/fC)"
            title = f"Gain for {name} under on {date}"
            title_short = "Gain under"
        elif var == "vt50_away":
            y_label = "Vt50 (mV)"
            title = f"Vt50 for {name} away on {date}"
            title_short = "Vt50 away"
        elif var == "vt50_under":
            y_label = "Vt50 (mV)"
            title = f"Vt50 for {name} under on {date}"
            title_short = "Vt50 under"
        else:
            y_label = var
            title = var
            title_short = var

        means, errors, labels, mask = [], [], [], []

        if not per_chip:
            for data in data_list:
                measurements = data.get('results', {}).get(var, {})
                if not measurements:
                    continue
                temp = data.get('properties', {}).get("DCS", {}).get("AMAC_NTCx", {})
                mask.append(1 if temp > 0 else 0)
                file_date = data.get('date')
                labels.append(file_date)
                all_channels = []
                for chip in measurements:
                    all_channels.extend(chip)
                mean_val, fit_err, y_fit = analyze_channels(np.array(all_channels))
                means.append(mean_val)
                errors.append(fit_err)
                if plot_hist:
                    plot_histogram_with_peaks(all_channels, threshold, n_channels=np.arange(len(all_channels)),
                                              mean=np.mean(all_channels), y_label=y_label, title = f"{title_short} hist, for {name} on {file_date}")
                failed, failed_groups = detect_failed_channels(all_channels, mean= mean_val)
                results[var] = {
                    "means": means,
                    "errors": errors,
                    "failed_channels": failed,
                    "failed_groups": failed_groups
                }
            output_file = os.path.join(folder_path_down, f"{var}_mean_fit_error_{name}_{date}.png")
            plot_summary(means, errors, mask, labels, y_label, title, title_short, output_file)
        else:
            # per-chip vykreslování
            all_means = []
            all_errors = []
            labels = [data.get('date') for data in data_list]
            mask = [(1 if data.get('properties', {}).get("DCS", {}).get("AMAC_NTCx", 0) > 20 else 0) for data in data_list]
            fig1, axs1 = plt.subplots(2, 3, figsize=(15, 8))
            axs1 = axs1.ravel()
            fig2, axs2 = plt.subplots(2, 3, figsize=(18, 8))
            axs2 = axs2.ravel()
            colors = plt.cm.tab10.colors
            for i in range(6):
                chip_means, chip_errors = [], []
                for j, data in enumerate(data_list):
                    measurements = data.get('results', {}).get(var, {})
                    if not measurements:
                        continue
                    chip_values = measurements[i]
                    n_channels = list(range(1 + i * 128, i * 128 + len(chip_values) + 1))
                    mean_val, fit_err, y_fit = analyze_channels(np.array(chip_values))
                    chip_means.append(mean_val)
                    chip_errors.append(fit_err)
                    all_means.append(mean_val)
                    all_errors.append(fit_err)
                    if plot_hist:
                        plot_histogram_with_peaks(chip_values, threshold, n_channels=n_channels, mean=np.mean(chip_values), y_label=y_label,
                                                  title = f"{title_short} hist,for {name} on {labels[j]}, chip {i + 1}")
                    axs1[i].plot(n_channels, chip_values, marker='.', linestyle='', markersize=5, label=labels[j], color = colors[j % 6])
                    axs1[i].plot(n_channels, y_fit, label=f"fit {labels[j]}", color = colors[j % 6])
                axs1[i].set_title(f"Chip {i+1}")
                axs1[i].set_xlabel("Channel Number")
                axs1[i].set_ylabel(y_label)
                axs1[i].grid(True)
                axs1[i].legend()
                x_pos = np.arange(len(chip_means))
                axs2[i].errorbar(x_pos + 1, chip_means, yerr=chip_errors)
                axs2[i].set_title(f"Chip {i+1}")
                axs2[i].set_ylabel(y_label)
                axs2[i].grid(True)
            fig1.suptitle(f"{title} for all chips", fontsize=16)
            fig2.suptitle(f"Mean values with fit errors, {title} for all chips")
            fig1.tight_layout(rect=[0, 0, 1, 0.95])
            fig2.tight_layout(rect=[0, 0, 1, 0.95])
            fig1.savefig(os.path.join(folder_path_down, f"{var}_single_chips_{name}_{date}.png"))
            plt.close(fig1)
            fig2.savefig(os.path.join(folder_path_down, f"{var}_mean_fit_error_single_chips_{name}_{date}.png"))
            plt.close(fig2)
            results[var] = {f"Chip_{i+1}": {"means": all_means[i::6], "errors": all_errors[i::6]} for i in range(6)}

    output_file = os.path.join(folder_path_down, f"{name}_{date}_results.json")
    save_results_to_json(results, output_file)
    return results



if __name__ == "__main__":
    analyze_and_plot(
        variables=["innse_away"],
        folder_path_up=r"C:/sfg/json",
        folder_path_down = r"C:/sfg/Code/Graphs",
        name="SN20USEH40000148_H1",
        date="20250702",
        per_chip=True,
        plot_hist = False
    )
