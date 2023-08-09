import numpy as np
import pandas as pd
import ROOT
import subprocess
import glob
import os
import argparse

os.chdir("../input")


def fill_bar(h, x, w, d, c, s):
    h.Fill(x, w)
    h.SetBinError(h.FindBin(x), d)
    h.SetLineColor(c)
    h.SetFillColor(c)
    if s:
        h.SetFillStyle(s)

def run_packed_benchmark_gpu(files, n, m, w):
    n_pts = len(files)
    setup_times = np.zeros(n_pts)
    dev_st = np.zeros(n_pts)
    decomp_times = np.zeros(n_pts)
    dev_dt = np.zeros(n_pts)
    unpack_times = np.zeros(n_pts)
    dev_pt = np.zeros(n_pts)
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
                "-p",
            ],
            stdout=subprocess.PIPE,
        )

        output = result.stdout.decode("utf-8").split()
        (
            ratios[i],
            setup_times[i],
            dev_st[i],
            decomp_times[i],
            dev_dt[i],
            unpack_times[i],
            dev_pt[i],
        ) = [float(o) for o in output[-7:]]
        sizes[i] = int(f.split(".")[1])

    #     print(setup_times, dev_st, decomp_times, dev_dt, ratios, sizes)
    return (
        setup_times,
        dev_st,
        decomp_times,
        dev_dt,
        unpack_times,
        dev_pt,
        ratios,
        sizes,
    )


def run_packed_benchmark_cpu(files, n, m, w, c=16):
    n_pts = len(files)
    decomp_times = np.zeros(n_pts)
    dev_dt = np.zeros(n_pts)
    unpack_times = np.zeros(n_pts)
    dev_pt = np.zeros(n_pts)
    ratios = np.zeros(n_pts)
    sizes = np.zeros(n_pts)

    for i, f in enumerate(files):
        result = subprocess.run(
            [
                "../cpu_root_decomp_omp",
                "-f",
                f,
                "-s",
                f.split(".")[1],
                "-n",
                str(n),
                "-m",
                str(m),
                "-c",
                str(c),
                "-w",
                str(w),
                "-p",
            ],
            stdout=subprocess.PIPE,
        )
        output = result.stdout.decode("utf-8").split()
        ratios[i] = float(output[-21])
        decomp_times[i] = float(output[-15])
        dev_dt[i] = float(output[-11])
        unpack_times[i] = float(output[-5])
        dev_pt[i] = float(output[-1])
        sizes[i] = int(f.split(".")[1])

    #     print(decomp_times, dev_dt, sizes)
    return decomp_times, dev_dt, unpack_times, dev_pt, ratios, sizes


n = 100
w = 10
nfiles = [1, 2, 4, 8, 16, 32]
file = glob.glob("packed_floats.64000.root.zstd")
nbins = 2 * len(nfiles) + len(nfiles) + 1

title = f"Comparison of decompression + unpacking 64kB of ROOT compressed floats"
c = ROOT.TCanvas("c1", title)
hs = ROOT.THStack("hs", "")

for i, m in enumerate(nfiles):
    (
        g_setup_times,
        g_dev_st,
        g_decomp_times,
        g_dev_dt,
        g_unpack_times,
        g_dev_pt,
        g_ratios,
        g_sizes,
    ) = run_packed_benchmark_gpu(file, n, m, w)
    (
        c_decomp_times,
        c_dev_dt,
        c_unpack_times,
        c_dev_pt,
        c_ratios,
        c_sizes,
    ) = run_packed_benchmark_cpu(file, n, m, w)

    # CPU
    cpu_bin = 1 + 3*i
    h1 = ROOT.TH1F(f"c_decomp{m}", "cpu", nbins, 0, nbins)
    fill_bar(h1, cpu_bin, c_decomp_times, c_dev_dt, ROOT.kRed, None)
    hs.Add(h1)

    h2 = ROOT.TH1F(f"c_unpack{m}", "cpu", nbins, 0, nbins)
    fill_bar(h2, cpu_bin, c_unpack_times, c_dev_pt, ROOT.kBlue, None)
    hs.Add(h2)

    # GPU

    gpuFillStyle = 1001
    gpu_bin = 2 + 3*i
    h3 = ROOT.TH1F(f"g_setup{m}", "gpu", nbins, 0, nbins)
    fill_bar(h3, gpu_bin, g_setup_times, g_dev_st, ROOT.kGreen, gpuFillStyle)
    hs.Add(h3)

    h4 = ROOT.TH1F(f"g_decomp{m}", "gpu", nbins, 0, nbins)
    fill_bar(h4, gpu_bin, g_decomp_times, g_dev_dt, ROOT.kRed, gpuFillStyle)
    hs.Add(h4)

    h5 = ROOT.TH1F(f"g_unpack{m}", "gpu", nbins, 0, nbins)
    fill_bar(h5, gpu_bin, g_unpack_times, g_dev_pt, ROOT.kBlue, gpuFillStyle)
    hs.Add(h5)


c.SetRightMargin(0.33)
c.SetBottomMargin(0.2)
l = ROOT.TLegend(0.687, 0.7, 0.99, 0.9)
l.AddEntry(h5, "Unpacking")
l.AddEntry(h4, "Decompression")
l.AddEntry(h3, "Setup")
l.SetTextSize(0.03)

hs.Draw("bar")
h1.SetContour(30)

xaxis = hs.GetXaxis()
xaxis.SetTitle("Time (ms)")
xaxis.SetTitle("#splitline{Device}{Number of files}")
xaxis.SetTitleOffset(2.5)
for i in range(2, nbins, 3):
    xaxis.SetBinLabel(i, "CPU")
    xaxis.SetBinLabel(i + 1, "GPU")

hs.GetXaxis().SetTickLength(0)
hs.SetTitle(title)

ox = ROOT.TGaxis(
    0, 0, nbins, 0, 0.0, nbins, 30, "S+", 0
)
ox.SetTickSize(0)
ox.SetTickLength(0)
ox.SetLabelOffset(0.05)
ox.SetLabelFont(42)
ox.SetLabelSize(0.04)

labelbins = [3, 6, 9, 12, 15, 18]
for i in range(30):
    if i not in labelbins:
        ox.ChangeLabel(i, -1, 0.0)
for i, e in enumerate(labelbins):
    ox.ChangeLabel(e, -1, -1, -1, -1, -1, f"{nfiles[i]}")

ox.Draw()

l.Draw()
# c.Draw()
c.SaveAs("hihi.png")
