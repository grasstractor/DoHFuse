# DNS-over-HTTP/3 (DoH3) Packet Analyzer

## Overview
This tool scans a root folder of per-site subdirectories containing `.pcap` captures, decodes QUIC/HTTP/3 payloads with a TLS key log, extracts DNS-over-HTTP/3 (`application/dns-message`) traffic, and produces:
1. A human-readable analysis text file per input `.pcap`
2. A per-site CSV summary
3. A cross-site master CSV summary

## Prerequisites
- **Go** (1.20+ recommended)
- **Wireshark/tshark** installed; confirm `tshark` path
- A **merged SSL key log** file (QUIC/TLS secrets) that matches the captures
- Go dependency:
  ```bash
  go get github.com/miekg/dns
  ```

## Directory Layout
```
captures-onlydoh3/           # MAIN_DIR; contains one subfolder per site
  ├── siteA/
  │   ├── foo1.pcap
  │   └── foo2.pcap
  ├── siteB/
  │   └── bar1.pcap
  └── ...
captures-onlydoh3/merged_ssl_url.log   # SSLKEYLOG
```

## Configuration
Edit constants at the top of the Go file if needed:
- `MAIN_DIR` — root containing site subfolders with `.pcap` files
- `SSLKEYLOG` — path to the merged SSL key log file
- `OUTPUT_DIR` — destination folder for reports/CSVs
- `MASTER_CSV` — name of the cross-site summary CSV
- `LOG_FILE` — log file name placed under `OUTPUT_DIR`
- `TSHARK_PATH` — absolute path to `tshark` (e.g., Windows path)
- `MAX_WORKERS` — number of sites to process concurrently

Adjust the `DNSServers` map to include other resolvers if needed.

## Build
```bash
go build -o doh3_analyzer pcap_doh3_packetinfo_en.go
```

## Run
```bash
./doh3_analyzer
```

## Outputs
All outputs are written under `OUTPUT_DIR`:
- **Per-site folder**:
  - Per-`pcap` text reports: `<pcap_basename>_analysis.txt`
  - Per-site CSV: `<site>_dns_summary.csv`
- **Master CSV**: `captures-onlydoh3_summary.csv`
- **Log file**: `analysis_captures-onlydoh3.log`

## Notes
- `tshark` path on Windows must be absolute, e.g. `F:\\Wireshark\\tshark.exe`
- The SSL key log must match the sessions in `.pcap` files
- Only frames with `Content-Type: application/dns-message` are counted

# PCAP to CSV Converter (QUIC/DoH3 Focus)

## Overview
This tool parses `.pcap` captures, extracts **UDP/443 (QUIC) packets involving target DNS resolvers**, and saves them into structured CSV files.

Each CSV contains per-packet fields:
- `RelativeTime(s)` — seconds since first packet
- `IntervalTime(s)` — time since previous packet
- `SrcIP`, `SrcPort`
- `DstIP`, `DstPort`
- `Length(Bytes)`
- `Direction` (1 = client→target, 0 = target→client)

## Configuration

Constants can be changed in the code:

- PCAP_DIR — root containing .pcap files

- OUTPUT_DIR — where CSVs are saved

- MAX_WORKERS — number of goroutines

- TARGET_ADDRESSES — IPs of DNS resolvers (Google DNS default)

## Outputs

- CSVs under OUTPUT_DIR (same relative folder layout as PCAPs)

- Each row corresponds to one UDP/443 packet involving target DNS resolvers