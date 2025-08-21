# DoH Website Traffic Capture Script

This folder contains the script `doh_captures.py`, which automates the process of visiting a list of websites using Chrome with DNS-over-HTTPS (DoH) enabled, capturing network traffic and SSL key logs for each visit. The script is designed for large-scale data collection for website fingerprinting research.

## Script Overview

- Visits each URL multiple times using Chrome in headless mode with DoH enabled.
- Captures network traffic using `tcpdump` and saves SSL key logs for each visit.
- Merges SSL key logs per URL and archives all results into a zip file.
- Handles process cleanup and error management automatically.

## Main Features

- **DoH Enabled:** Chrome is configured to use Google's DoH server for all DNS queries.
- **Automated Visits:** Each URL is visited a specified number of times (`NUM_VISITS`).
- **Traffic Capture:** Network traffic is captured in `.pcap` files using `tcpdump`.
- **SSL Key Logging:** SSL keys are logged for each visit to enable encrypted traffic analysis.
- **Error Handling:** Skips domains after repeated failures and cleans up temporary files and processes.
- **Archiving:** All results are archived into a single zip file for easy transfer and analysis.

## Common Parameters

- `DOH_SERVER`: DoH server URL (default: Google's DoH)
- `URL_FILE`: Path to the file containing the list of URLs to visit
- `OUTPUT_DIR`: Directory to store all capture results
- `NUM_VISITS`: Number of visits per URL

## Outputs

- Per-URL directories containing `.pcap` files and SSL key logs
- Merged SSL key logs per URL
- A zip archive of all results

## Usage

1. Prepare a text file (`urls.txt`) with one URL per line.
2. Run the script:
   ```bash
   python doh_captures.py
   ```
3. Results will be saved in the specified output directory and archived as a zip file.

## Notes

- Do not run the script as root or with sudo.
- The script is designed for Linux environments; some features may require adaptation for other OSes.
- For more details, refer to the code comments in `doh_captures.py` or contact the project maintainer.
