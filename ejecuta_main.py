import subprocess
import sys
import os

veces = 27

for i in range(veces):
    print(f"\n----- Ejecución {i + 1} de {veces} -----")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run([sys.executable, "main.py"], env=env)
    print(f"main.py terminó la ejecución {i + 1}.\n")
