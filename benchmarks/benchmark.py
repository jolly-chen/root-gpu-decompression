import numpy as np
import ROOT
import subprocess
import glob
import os


def run_benchmark_gpu(n, files):
    n_pts = len(files)
    setup_times = np.zeros(n_pts)
    dev_st = np.zeros(n_pts)
    decomp_times = np.zeros(n_pts)
    dev_dt = np.zeros(n_pts)
    ratios = np.zeros(n_pts)

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
            ],
            stdout=subprocess.PIPE,
        )
        output = result.stdout.decode("utf-8").split()
        setup_times[i], dev_st[i], decomp_times[i], dev_dt[i], ratios[i] = [
            float(o) for o in output[-6:-1]
        ]

    # Sort the result by compression ratio
    sort_idx = np.argsort(ratios)
    setup_times = setup_times[sort_idx]
    dev_st = dev_st[sort_idx]
    decomp_times = decomp_times[sort_idx]
    dev_dt = dev_dt[sort_idx]
    ratios = ratios[sort_idx]
    print(setup_times, decomp_times, ratios)
    return setup_times, dev_st, decomp_times, dev_dt, ratios



def plot_bar(c, setup_times, dev_st, decomp_times, dev_dt, ratios):
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

def plot_line(c, setup_times, dev_st, decomp_times, dev_dt, ratios):
    n_pts = len(ratios)
    zeroes = np.zeros(n_pts, dtype=float)

    mg = ROOT.TMultiGraph()
    g1 = ROOT.TGraphErrors(n_pts, ratios, setup_times, zeroes, dev_st)
    g1.SetLineColor(ROOT.kRed)

    g2 = ROOT.TGraphErrors(n_pts, ratios, decomp_times, zeroes, dev_dt)
    g2.SetLineColor(ROOT.kBlue)
    c.SetRightMargin(0.32)
    mg.Add(g1)
    mg.Add(g2)

    mg.GetXaxis().SetTitle("Compression ratio")
    mg.GetXaxis().SetLimits(0, 6)
    mg.GetYaxis().SetTitle("Time (ms)")
    mg.GetYaxis().SetLimits(0, np.max(decomp_times) + 10)
    mg.DrawClone("AL")

    l = ROOT.TLegend(0.687, 0.7, 0.99, 0.9)
    l.AddEntry(g1, "Setup")
    l.AddEntry(g2, "Decompression")
    l.SetTextSize(0.03)
    l.DrawClone()



if __name__ == "__main__":
    n = 5
    os.chdir("../input")
    files = glob.glob("*.root.zstd")

    c = ROOT.TCanvas("c1", "Decompression of ROOT compressed files")
    results = run_benchmark_gpu(n, files)
    plot_line(c, *results)
