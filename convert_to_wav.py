#!/usr/bin/env python3

import os
import subprocess

def main():
    with open("manifest.txt", "r") as f:
        manifest = f.readlines()

    for f in manifest:
        f = f.replace("\n", "")
        outfile = os.path.splitext(f)[0]
        outfile += ".wav"

        # Convert from original file type to WAV@44.1kHz
        res = subprocess.check_output(["ffmpeg", "-i", f, "-ar", "44100", outfile])
        for line in res:
            print(res)

if __name__ == "__main__":
    main()
