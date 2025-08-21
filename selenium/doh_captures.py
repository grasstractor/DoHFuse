import os
import sys
import time
import shutil
import zipfile
import tempfile
import subprocess
import psutil
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests


# ========== Do not run as root ==========
# if os.geteuid() == 0:
#     print("[❌] Please do not run this program as root or with sudo!")
#     sys.exit(1)

# ========== Parameter Configuration ==========
DOH_SERVER = "https://dns.google/dns-query"
URL_FILE = "urls.txt"
OUTPUT_DIR = os.path.abspath("./captures-test-doh-5")
SSLKEYLOG_MERGED = os.path.join(OUTPUT_DIR, "sslkeys_merged.log")
ARCHIVE_PATH = os.path.join(OUTPUT_DIR, "capture_results.zip")
NUM_VISITS = 100  # Number of visits per URL

# ========== Read URL list ==========
def load_urls(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# ========== Build Chrome launch options ==========
def build_chrome_options(profile_path, sslkeylog_file):
    options = Options()
    options.add_argument(f"--user-data-dir={profile_path}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-client-side-phishing-detection")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-default-apps")
    options.add_argument("--no-first-run")
    options.add_argument("--headless=new")

    # Force enable DoH
    options.add_experimental_option("localState", {
        "dns_over_https.mode": "secure",
        "dns_over_https.templates": "https://dns.google/dns-query"
    })

    options.environment = {"SSLKEYLOGFILE": sslkeylog_file}
    return options

# ========== Merge multiple sslkeys files ==========
def merge_ssl_keys(log_files, merged_path):
    with open(merged_path, "w") as outfile:
        for file in log_files:
            if os.path.exists(file) and os.path.getsize(file) > 0:
                with open(file, "r") as infile:
                    outfile.write(infile.read())
                print(f"    📝 Merged: {os.path.basename(file)} -> {os.path.basename(merged_path)}")
            elif os.path.exists(file):
                print(f"    ⚠️ Skipped empty file: {os.path.basename(file)}")

# ========== Kill all Chrome processes ==========
def kill_chrome_processes():
    """Ensure all Chrome processes are terminated"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Match Chrome related processes
            if 'chrome' in proc.info['name'].lower() or 'chromedriver' in proc.info['name'].lower():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                print(f"    ⚠️ Force killed leftover process: {proc.info['name']} (PID: {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

# ========== Main Program ==========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    urls = load_urls(URL_FILE)

    print(f"[+] Loaded {len(urls)} URLs")
    print(f"[+] Each URL will be visited {NUM_VISITS} times")

    # Store all file lists
    all_pcap_files = []
    all_ssl_files = []
    per_url_merged_ssl_files = []  # Store merged SSL files for each URL
    
    total_runs = len(urls) * NUM_VISITS
    current_run = 0


    for url_idx, url in enumerate(urls):
        hostname = url.replace("https://", "").replace("http://", "").split("/")[0]
        url_dir = os.path.join(OUTPUT_DIR, f"url_{url_idx+1}_{hostname}")
        os.makedirs(url_dir, exist_ok=True)

        url_ssl_logs = []
        consecutive_errors = 0  # New: consecutive error counter
        driver_binary = ChromeDriverManager().install()
        for visit_idx in range(1, NUM_VISITS + 1):
            current_run += 1

            # New: skip domain after 5 consecutive errors
            if consecutive_errors >= 5:
                print(f"⚠️ Domain {hostname} encountered 5 consecutive errors, skipping remaining visits")
                break

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pcap_path = os.path.join(url_dir, f"capture_{visit_idx}_{ts}_{hostname}.pcap")
            sslkeylog_file = os.path.join(url_dir, f"sslkeys_{visit_idx}.log")

            # ↓ No longer pre-adding to list, only add after success
            # all_pcap_files.append(pcap_path)
            # all_ssl_files.append(sslkeylog_file)
            # url_ssl_logs.append(sslkeylog_file)

            profile_path = tempfile.mkdtemp(prefix=f"chrome_profile_{url_idx+1}_{visit_idx}_")
            print(f"[{current_run}/{total_runs}] Visiting {url_idx+1}-{visit_idx}: {url}")
            print(f"    🗂️ Temp directory: {profile_path}")

            tcpdump_proc = None
            driver = None

            try:
                DNS_HOST_FILTER = [
                    "host 8.8.8.8", "host 8.8.4.4",
                    "host 2001:4860:4860::8888", "host 2001:4860:4860::8844"
                ]
                bpf_filter = ["udp", "and", "port", "443", "and", "("] + \
                            [" or ".join(DNS_HOST_FILTER)] + [")"]

                tcpdump_cmd = [
                    "tcpdump",
                    "-i", "any",
                    "-s", "0",
                    "-B", "4096",
                    "-n",
                    "-w", pcap_path,
                ] + bpf_filter

                tcpdump_proc = subprocess.Popen(tcpdump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


                os.environ["SSLKEYLOGFILE"] = sslkeylog_file

                options = build_chrome_options(profile_path, sslkeylog_file)
                service = Service(driver_binary)
                driver = webdriver.Chrome(service=service, options=options)

                start_time = time.time()
                driver.get("https://"+url)
                time.sleep(3)

                status = "success"
                try:
                    if "404" in driver.title or "error" in driver.title:
                        status = "possible failure"
                        consecutive_errors += 1  # New: increment error counter
                    else:
                        consecutive_errors = 0  # New: reset error counter
                except:
                    status = "page exception"
                    consecutive_errors += 1  # New: increment error counter

                driver.quit()
                elapsed = time.time() - start_time

                print(f"    ⏱️ Time elapsed: {elapsed:.2f}s, Status: {status}")
                print(f"    📦 Capture saved: {os.path.basename(pcap_path)}")
                print(f"    🔑 SSL key saved: {os.path.basename(sslkeylog_file)}")

            except Exception as e:
                print(f"[!] Error: {str(e)[:100]}")
                status = "error"
                consecutive_errors += 1  # New: increment error counter

            finally:
                try:
                    if driver:
                        driver.quit()
                except Exception as e:
                    print(f"    ⚠️ Error closing browser: {str(e)[:50]}")

                if tcpdump_proc:
                    tcpdump_proc.terminate()
                    tcpdump_proc.wait()
                    print("    ✅ tcpdump terminated")

                kill_chrome_processes()

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        shutil.rmtree(profile_path)
                        print(f"    ✅ Temp directory deleted (attempt {attempt+1}/{max_retries})")
                        break
                    except Exception as e:
                        print(f"    ⚠️ Failed to delete temp directory (attempt {attempt+1}/{max_retries}): {str(e)[:80]}")
                        time.sleep(1)
                        kill_chrome_processes()
                else:
                    print(f"    ❌ Unable to delete temp directory: {profile_path}")

                print("-" * 60)

            # Only keep files and paths if successful, otherwise delete files
            if status == "success":
                all_pcap_files.append(pcap_path)
                all_ssl_files.append(sslkeylog_file)
                url_ssl_logs.append(sslkeylog_file)
            else:
                # Delete invalid files
                for path in (pcap_path, sslkeylog_file):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                            print(f"    🗑️ Deleted invalid file: {os.path.basename(path)}")
                    except Exception as e:
                        print(f"    ⚠️ Failed to delete file: {path}, Reason: {str(e)}")


        url_merged_ssl = os.path.join(url_dir, f"sslkeys_url_{url_idx+1}.log")
        print(f"[🔧] Merging all SSL key logs for {hostname} -> {os.path.basename(url_merged_ssl)}")
        merge_ssl_keys(url_ssl_logs, url_merged_ssl)
        per_url_merged_ssl_files.append(url_merged_ssl)
        all_ssl_files.append(url_merged_ssl)

    # Archive all results
    print("[📦] Archiving all results...")
    with zipfile.ZipFile(ARCHIVE_PATH, "w") as archive:
    # Add all pcap files
        for file in all_pcap_files:
            if os.path.exists(file):
                rel_path = os.path.relpath(file, OUTPUT_DIR)
                archive.write(file, arcname=rel_path)
        
    # Add all SSL files
        for file in all_ssl_files:
            if os.path.exists(file) and os.path.getsize(file) > 0:
                rel_path = os.path.relpath(file, OUTPUT_DIR)
                archive.write(file, arcname=rel_path)

    print(f"[✅] All files have been archived as: {ARCHIVE_PATH}")
    print(f"[+] Total visits completed: {total_runs}")
    print(f"[+] File structure:")
    print(f"    - Each URL has a separate directory: url_1_example.com/")
    print(f"    - Each URL directory contains: {NUM_VISITS} pcap files + {NUM_VISITS} individual SSL logs + 1 merged SSL log")
    print(f"    - Root directory contains: merged SSL key logs for all URLs (sslkeys_merged.log)")

if __name__ == "__main__":
    main()