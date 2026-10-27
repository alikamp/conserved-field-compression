#!/usr/bin/env python3

import subprocess
import re
import csv

path = "/mnt/c/Inputs/SDRbench/minSSIMinputs/"

files = [
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-RELHUM_1_26_1800_3600.f32",
    "sdrbench-ISABEL-100x500x500-TCf48.bin.f32",
    "sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-velocity_x.f32",
]

eb_values = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
mode_values = [0]

csv_file = "SLEEKresults.csv"

with open(csv_file, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)

    writer.writerow([
        "input",
        "error_bound",
        "mode",
        "compression_ratio",
        "preprocessing_GBps",
        "compression_GBps",
        "encoding_GBps",
        "decompression_GBps",
        "postprocessing_GBps",
        "decoding_GBps",
        "abs_eb",
        "used_eb",
        "eb_violations",
        "mass_i",
        "energy_i"
    ])

    for m in mode_values:
        for infile in files:
            for e in eb_values:

                cmd = ["./cSLEEKcompr", path+infile, "t.t", str(e), str(m)]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )

                output = result.stdout

                ratio = re.search(r'([0-9.]+)x', output)
                encoding = re.search(
                    r'encoding throughput:\s+([0-9.]+)\s+Gbytes/s',
                    output
                )


                cmd = ["./cSLEEKdecom", "t.t", "tt.tt", str(e), str(m), path+infile]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )

                output = result.stdout

                decoding = re.search(
                    r'decoding throughput:\s+([0-9.]+)\s+Gbytes/s',
                    output
                )


                cmd = ["./check_conservation", path+infile, "tt.tt", str(e)]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )

                output = result.stdout

                abseb = re.search(
                    r'resulting ABS EB:\s+([0-9.]+)\s+',
                    output
                )
                usedeb = re.search(
                    r'utilized  ABS EB:\s+([0-9.]+)\s+',
                    output
                )
                violations = re.search(
                    r'num_eb_violations:\s+([0-9]+)\s+',
                    output
                )
                mass = re.search(
                    r'\s+mass:\s+([0-9.e+-]+)\s+',
                    output
                )
                energy = re.search(
                    r'\s+energy:\s+([0-9.e+-]+)\s+',
                    output
                )


                if all([ratio, encoding, decoding, abseb, usedeb, violations, mass, energy]):
                    writer.writerow([
                        infile,
                        e,
                        m,
                        float(ratio.group(1)),
                        float(0.0),
                        float(encoding.group(1)),
                        float(0.0),
                        float(decoding.group(1)),
                        float(0.0),
                        float(0.0),
                        float(abseb.group(1)),
                        float(abseb.group(1)),
                        int(violations.group(1)),
                        float(mass.group(1)),
                        float(energy.group(1))
                    ])

                    print(
                        f"{infile} eb={e} mode={m} "
                        f"ratio={ratio.group(1)} "
                        f"enc={encoding.group(1)} "
                        f"dec={decoding.group(1)} "
                        f"abs={abseb.group(1)} "
                        f"used={abseb.group(1)} "
                        f"vio={violations.group(1)} "
                        f"mass={mass.group(1)} "
                        f"en={energy.group(1)} "
                    )
                else:
                    print(
                        f"WARNING: Could not parse output "
                        f"when processing {infile} for eb={e} mode={m}"
                    )

    cmd = ["rm", "-f", "t.t", "tt.tt"]
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True
    )

print(f"Results written to {csv_file}")
