import numpy as np
import pandas as pd
import ROOT
import subprocess
import glob
import os
import argparse


def sort_lists(to_sort, sort_idx):
    return [l[sort_idx] for l in to_sort]


def run_benchmark_gpu(files, n, m, w):
    n_pts = len(files)
    setup_times = np.zeros(n_pts)
    dev_st = np.zeros(n_pts)
    decomp_times = np.zeros(n_pts)
    dev_dt = np.zeros(n_pts)
    ratios = np.zeros(n_pts)
    sizes = np.zeros(n_pts)

    for i, f in enumerate(files):
        result = subprocess.run(
            [
                "../gpu_root_decomp",
                "-f",
                f,
                "-t",
                f.split(".")[-1],
                "-n",
                str(n),
                "-m",
                str(m),
                "-w",
                str(w),
            ],
            stdout=subprocess.PIPE,
        )
        output = result.stdout.decode("utf-8").split()
        setup_times[i], dev_st[i], decomp_times[i], dev_dt[i], ratios[i] = [
            float(o) for o in output[-7:-2]
        ]
        sizes[i] = int(f.split(".")[1]) * m

    # Sort the results
    print(setup_times, dev_st, decomp_times, dev_dt, ratios, sizes)
    return setup_times, dev_st, decomp_times, dev_dt, ratios, sizes


def run_benchmark_cpu(files, n, m, w):
    n_pts = len(files)
    decomp_times = np.zeros(n_pts)
    dev_dt = np.zeros(n_pts)
    ratios = np.zeros(n_pts)
    sizes = np.zeros(n_pts)

    for i, f in enumerate(files):
        result = subprocess.run(
            [
                "../cpu_root_decomp",
                "-f",
                f,
                "-s",
                f.split(".")[1],
                "-n",
                str(n),
                "-m",
                str(m),
                "-c",
                str(),
                "-w",
                str(w),
            ],
            stdout=subprocess.PIPE,
        )
        output = result.stdout.decode("utf-8").split()
        decomp_times[i] = float(output[-8])
        dev_dt[i] = float(output[-4])
        ratios[i] = float(output[-1])
        sizes[i] = int(f.split(".")[1]) * m

    # Sort the result by compression ratio
    print(decomp_times, dev_dt, sizes)
    return decomp_times, dev_dt


def plot_bar(c, gpu_results):
    setup_times, dev_st, decomp_times, dev_dt, ratios, sizes = gpu_results
    # Visualize results in bar chart
    n_pts = len(ratios)

    hs = ROOT.THStack("hs", "")
    h1 = ROOT.TH1F("h1", "Setup", 20, 0, 6)
    for i in range(n_pts):
        h1.Fill(ratios[i], setup_times[i])
        h1.SetBinError(h1.FindBin(ratios[i]), dev_st[i])
    h1.SetFillColor(ROOT.kRed)

    h2 = ROOT.TH1F("h2", "Decompression", 20, 0, 6)
    for i in range(n_pts):
        h2.Fill(ratios[i], decomp_times[i])
        h2.SetBinError(h2.FindBin(ratios[i]), dev_dt[i])
    h2.SetFillColor(ROOT.kBlue)
    c.SetRightMargin(0.32)
    hs.Add(h1)
    hs.Add(h2)
    hs.DrawClone("bar")

    h1.GetXaxis().SetTitle("Compression ratio")
    h1.GetYaxis().SetTitle("Time (ms)")
    l = ROOT.TLegend(0.687, 0.7, 0.99, 0.9)
    l.AddEntry(h1, "Setup")
    l.AddEntry(h2, "Decompression")
    l.SetTextSize(0.03)
    l.DrawClone()


def plot_line(c, cpu_results, gpu_results, title):
    cpu_decomp_times, cpu_dev_dt = cpu_results
    (
        gpu_setup_times,
        gpu_dev_st,
        gpu_decomp_times,
        gpu_dev_dt,
        ratios,
        sizes,
    ) = gpu_results

    n_pts = len(sizes)
    zeroes = np.zeros(n_pts, dtype=float)

    mg = ROOT.TMultiGraph()
    g1 = ROOT.TGraphErrors(n_pts, sizes, gpu_setup_times, zeroes, gpu_dev_st)
    g1.SetLineColor(ROOT.kRed)

    g2 = ROOT.TGraphErrors(n_pts, sizes, gpu_decomp_times, zeroes, gpu_dev_dt)
    g2.SetLineColor(ROOT.kBlue)

    g3 = ROOT.TGraphErrors(n_pts, sizes, cpu_decomp_times, zeroes, cpu_dev_dt)

    c.SetRightMargin(0.32)
    # c.SetLogx()
    mg.Add(g1)
    mg.Add(g2)
    mg.Add(g3)

    mg.SetTitle(title)
    mg.GetXaxis().SetTitle("Decompressed size (B)")
    mg.GetXaxis().SetLimits(0, np.max(sizes) + np.min(sizes))
    mg.GetYaxis().SetTitle("Time (ms)")
    mg.GetYaxis().SetLimits(0, np.max(gpu_decomp_times) + 10)
    mg.DrawClone("AL")

    l = ROOT.TLegend(0.687, 0.7, 0.99, 0.9)
    l.AddEntry(g1, "GPU - Setup")
    l.AddEntry(g2, "GPU - Decompression")
    l.AddEntry(g3, "CPU - Decompression")
    l.SetTextSize(0.03)
    l.DrawClone()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--n", help="Number of repetitions", type=int)
    # parser.add_argument("-w", "--w", help="Warmup", type=int)
    # parser.add_argument("-m", "--m", help="Multi file size", type=int)
    # args = parser.parse_args()

    # n = args.n if args.n else 5
    # m = args.m if args.m else 100
    # w = args.w if args.w else 5
    # print(f"n:{n} m:{m}")
    # os.chdir("../input")
    # files = glob.glob("uniform*.root.zstd")

    # c = ROOT.TCanvas("c1", "Decompression of ROOT compressed files")
    # gpu_results = run_benchmark_gpu(n, m, w, files)
    # cpu_results = run_benchmark_cpu(n, m, w, files)
    # plot_line(c, cpu_results, gpu_results)



    os.chdir("../input")
    c = ROOT.TCanvas("c1")
    c.Divide(3,1)

    c.cd(1)
    n = 10
    w = 10
    m = 100
    files = glob.glob("low_compression*.root.zstd")
    gpu_results = run_benchmark_gpu(files, n, m, w)
    cpu_results = run_benchmark_cpu(files, n, m, w)
    cpu_results = sort_lists(cpu_results, np.argsort(cpu_results[-1])) # Sort by size
    gpu_results = sort_lists(gpu_results, np.argsort(gpu_results[-1])) # Sort by size
    title = f"Decompression of ROOT compressed files with average ratio: {np.mean(gpu_results[-2])}"
    plot_line(c, cpu_results, gpu_results, title)

    c.cd(2)
    n = 10
    w = 10
    m = 100
    files = glob.glob("mid_compression*.root.zstd")
    gpu_results = run_benchmark_gpu(files, n, m, w)
    cpu_results = run_benchmark_cpu(files, n, m, w)
    cpu_results = sort_lists(cpu_results, np.argsort(cpu_results[-1])) # Sort by size
    gpu_results = sort_lists(gpu_results, np.argsort(gpu_results[-1])) # Sort by size
    title = f"Decompression of ROOT compressed files with average ratio: {np.mean(gpu_results[-2])}"
    c = ROOT.TCanvas("c1", title)
    plot_line(c, cpu_results, gpu_results, title)

    c.DrawClone()