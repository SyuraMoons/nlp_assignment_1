"""Run the three per-site scrapers in parallel (one process per site). Each site
already rate-limits itself against its own host, so running them concurrently roughly
triples throughput without hitting any single site harder.

Run with: python -m scrapers.run_all
"""
import subprocess
import sys

SITE_MODULES = ["scrapers.kontan", "scrapers.bisnis", "scrapers.cnbc_indonesia"]


def main():
    procs = [
        subprocess.Popen([sys.executable, "-m", module])
        for module in SITE_MODULES
    ]
    exit_codes = [p.wait() for p in procs]

    for module, code in zip(SITE_MODULES, exit_codes):
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"{module}: {status}")

    if any(exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
