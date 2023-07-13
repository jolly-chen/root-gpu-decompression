import ROOT
import sys
import numpy as np

if __name__ == "__main__":
    file = sys.argv[1]
    with open(file, "r") as f:
        data = np.fromfile(f, dtype=np.uint8)

    c = ROOT.TCanvas("c1", "Data distribution")
    h = ROOT.TH1I("h1", "h1", 256, 0, 255)
    for b in data:
        h.Fill(b)
    h.Draw()
