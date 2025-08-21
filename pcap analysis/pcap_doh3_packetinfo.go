package main

import (
	"bufio"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/miekg/dns"
)

// ================= Configuration =================
//
// MAIN_DIR:    Root folder that contains per-site subfolders; each subfolder holds .pcap files.
// SSLKEYLOG:   Path to the merged SSL key log file for QUIC/TLS decryption.
// OUTPUT_DIR:  Where all analysis artifacts are written.
// MASTER_CSV:  Path (inside OUTPUT_DIR) for the cross-site master summary CSV.
// LOG_FILE:    Relative path (inside OUTPUT_DIR) for the analysis log file.
// TSHARK_PATH: Absolute path to tshark executable.
// MAX_WORKERS: Max number of sites processed concurrently.
//
// =================================================
const (
	MAIN_DIR    = "captures-onlydoh3"
	SSLKEYLOG   = "captures-onlydoh3/merged_ssl_url.log"
	OUTPUT_DIR  = "captures-onlydoh3_analysis"
	MASTER_CSV  = "captures-onlydoh3_summary.csv"
	LOG_FILE    = "analysis_captures-onlydoh3.log"
	TSHARK_PATH = "F:\\Wireshark\\tshark.exe"
	MAX_WORKERS = 10
)

// DNS resolvers we consider as targets (used to classify direction)
var DNSServers = map[string]bool{
	"8.8.8.8":              true,
	"8.8.4.4":              true,
	"2001:4860:4860::8844": true,
	"2001:4860:4860::8888": true,
}

// PacketInfo is retained for potential future extension (not persisted currently)
type PacketInfo struct {
	Number      string
	Time        string
	Length      int
	SrcIP       string
	DstIP       string
	StreamIDs   []string
	ContentType string
	Direction   string
	DNSData     string
	DNSMsg      *dns.Msg
}

type FileStats struct {
	FilePath    string
	FileName    string
	SiteName    string
	C2SCount    int
	S2CCount    int
	C2SLengths  []int
	S2CLengths  []int
	TotalCount  int
	TargetCount int
}

type SiteStats struct {
	SiteName    string
	FileStats   []FileStats
	C2SCount    int
	S2CCount    int
	C2SMinLen   int
	C2SMaxLen   int
	S2CMinLen   int
	S2CMaxLen   int
	TotalCount  int
	TargetCount int
}

var (
	logMutex sync.Mutex
	logFile  *os.File
)

// initLog sets log output to OUTPUT_DIR/LOG_FILE
func initLog() error {
	var err error
	logFile, err = os.OpenFile(filepath.Join(OUTPUT_DIR, LOG_FILE), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	log.SetOutput(logFile)
	return nil
}

// logMessage prints to stdout and writes to the log file with a timestamp
func logMessage(message string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	logLine := fmt.Sprintf("[%s] %s", timestamp, message)
	fmt.Println(logLine)

	logMutex.Lock()
	defer logMutex.Unlock()
	logFile.WriteString(logLine + "\n")
}

// runTshark runs tshark to decode QUIC/HTTP3 with the provided keylog file and returns JSON-decoded packets
func runTshark(pcapPath string) ([]map[string]interface{}, error) {
	cmd := exec.Command(
		TSHARK_PATH,
		"-r", pcapPath,
		"-Y", "quic",
		"-o", fmt.Sprintf("tls.keylog_file:%s", SSLKEYLOG),
		"-T", "json",
		"-e", "frame.number",
		"-e", "frame.time",
		"-e", "frame.len",
		"-e", "ip.len",
		"-e", "ipv6.plen",
		"-e", "ip.src",
		"-e", "ipv6.src",
		"-e", "ip.dst",
		"-e", "ipv6.dst",
		"-e", "quic.stream.stream_id",
		"-e", "http3.headers.content_type",
		"-e", "http3.data",
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("tshark execution failed: %v, output: %s", err, string(output))
	}

	var packets []map[string]interface{}
	if err := json.Unmarshal(output, &packets); err != nil {
		return nil, fmt.Errorf("JSON unmarshal failed: %v", err)
	}

	return packets, nil
}

// parseDNSData parses hex-encoded DNS message (from HTTP/3 data)
func parseDNSData(hexData string) (*dns.Msg, error) {
	hexStr := strings.ReplaceAll(hexData, ":", "")
	data, err := hex.DecodeString(hexStr)
	if err != nil {
		return nil, err
	}

	msg := &dns.Msg{}
	if err := msg.Unpack(data); err != nil {
		return nil, err
	}
	return msg, nil
}

// classifyDirection returns "c2s", "s2c", or "unknown" based on whether src/dst matches known DNS resolvers
func classifyDirection(srcIP, dstIP string) string {
	if DNSServers[dstIP] {
		return "c2s"
	} else if DNSServers[srcIP] {
		return "s2c"
	}
	return "unknown"
}

// analyzePcapFile processes a single .pcap for a given site, writes a per-file text report, and aggregates statistics
func analyzePcapFile(pcapPath, siteName string) (FileStats, error) {
	stats := FileStats{
		FilePath: pcapPath,
		FileName: filepath.Base(pcapPath),
		SiteName: siteName,
	}

	packets, err := runTshark(pcapPath)
	if err != nil {
		return stats, err
	}

	siteOutputDir := filepath.Join(OUTPUT_DIR, siteName)
	if err := os.MkdirAll(siteOutputDir, os.ModePerm); err != nil {
		return stats, fmt.Errorf("failed to create output directory: %v", err)
	}

	outputFileName := strings.TrimSuffix(stats.FileName, ".pcap") + "_analysis.txt"
	outputPath := filepath.Join(siteOutputDir, outputFileName)
	outputFile, err := os.Create(outputPath)
	if err != nil {
		return stats, fmt.Errorf("failed to create output file: %v", err)
	}
	defer outputFile.Close()

	writer := bufio.NewWriter(outputFile)
	defer writer.Flush()

	header := fmt.Sprintf(
		"DNS-over-HTTP/3 Analysis Report\nGenerated at: %s\nSite: %s\nFile: %s\nTarget DNS servers: %v\n%s\n\n",
		time.Now().Format("2006-01-02 15:04:05"),
		siteName,
		pcapPath,
		DNSServers,
		strings.Repeat("=", 80),
	)
	writer.WriteString(header)

	for _, pkt := range packets {
		fields, ok := pkt["_source"].(map[string]interface{})["layers"].(map[string]interface{})
		if !ok {
			continue
		}

		length := 0
		if frameLen, ok := fields["frame.len"].([]interface{}); ok && len(frameLen) > 0 {
			if lenStr, ok := frameLen[0].(string); ok {
				if lenVal, err := strconv.Atoi(lenStr); err == nil {
					length = lenVal
				}
			}
		}

		srcIP := ""
		if ipSrc, ok := fields["ip.src"].([]interface{}); ok && len(ipSrc) > 0 {
			srcIP = ipSrc[0].(string)
		} else if ipv6Src, ok := fields["ipv6.src"].([]interface{}); ok && len(ipv6Src) > 0 {
			srcIP = ipv6Src[0].(string)
		}

		dstIP := ""
		if ipDst, ok := fields["ip.dst"].([]interface{}); ok && len(ipDst) > 0 {
			dstIP = ipDst[0].(string)
		} else if ipv6Dst, ok := fields["ipv6.dst"].([]interface{}); ok && len(ipv6Dst) > 0 {
			dstIP = ipv6Dst[0].(string)
		}

		// Only consider packets that involve our target DNS servers
		if !DNSServers[srcIP] && !DNSServers[dstIP] {
			continue
		}

		stats.TargetCount++

		contentType := ""
		if http3Content, ok := fields["http3.headers.content_type"].([]interface{}); ok && len(http3Content) > 0 {
			contentType = http3Content[0].(string)
		}

		// Filter to DoH3 application payloads only
		if contentType != "application/dns-message" {
			continue
		}

		streamIDs := []string{}
		if quicStreams, ok := fields["quic.stream.stream_id"].([]interface{}); ok {
			for _, stream := range quicStreams {
				streamIDs = append(streamIDs, stream.(string))
			}
		}

		dnsData := ""
		if http3Data, ok := fields["http3.data"].([]interface{}); ok && len(http3Data) > 0 {
			dnsData = http3Data[0].(string)
		}

		var dnsMsg *dns.Msg
		if dnsData != "" {
			if msg, err := parseDNSData(dnsData); err == nil {
				dnsMsg = msg
			}
		}

		direction := classifyDirection(srcIP, dstIP)
		stats.TotalCount++

		switch direction {
		case "c2s":
			stats.C2SCount++
			stats.C2SLengths = append(stats.C2SLengths, length)
		case "s2c":
			stats.S2CCount++
			stats.S2CLengths = append(stats.S2CLengths, length)
		}

		line := fmt.Sprintf("Frame #%s | Time: %s | Length: %d bytes | Direction: %s\n",
			fields["frame.number"].([]interface{})[0],
			fields["frame.time"].([]interface{})[0],
			length,
			direction)
		line += fmt.Sprintf("Src IP: %s -> Dst IP: %s\n", srcIP, dstIP)
		line += fmt.Sprintf("QUIC Stream IDs: %s\n", strings.Join(streamIDs, ", "))
		line += fmt.Sprintf("Content-Type: %s\n", contentType)

		if dnsMsg != nil {
			line += "\nDNS Message (parsed):\n"
			line += fmt.Sprintf("  ID: %d\n", dnsMsg.Id)
			line += fmt.Sprintf("  Opcode: %s\n", dns.OpcodeToString[dnsMsg.Opcode])
			line += fmt.Sprintf("  Rcode: %s\n", dns.RcodeToString[dnsMsg.Rcode])

			if len(dnsMsg.Question) > 0 {
				line += "\nQuestions:\n"
				for _, q := range dnsMsg.Question {
					line += fmt.Sprintf("  %s (Type: %s, Class: %s)\n",
						q.Name,
						dns.TypeToString[q.Qtype],
						dns.ClassToString[q.Qclass])
				}
			}

			if len(dnsMsg.Answer) > 0 {
				line += "\nAnswers:\n"
				for _, rr := range dnsMsg.Answer {
					line += fmt.Sprintf("  %s\n", rr.String())
				}
			}
		} else if dnsData != "" {
			line += "\nDNS Raw Hex: " + dnsData + "\n"
		}

		line += fmt.Sprintf("%s\n\n", strings.Repeat("-", 80))
		writer.WriteString(line)
	}

	// If nothing was captured for this file, remove the empty analysis file
	if stats.C2SCount == 0 && stats.S2CCount == 0 {
		outputFile.Close()
		os.Remove(outputPath)
		return stats, nil
	}

	// Per-file summary footer
	summary := fmt.Sprintf("\n%s\nFile Summary:\n", strings.Repeat("=", 80))
	summary += fmt.Sprintf("Total frames (all): %d\n", stats.TotalCount)
	summary += fmt.Sprintf("Frames involving target DNS servers: %d\n", stats.TargetCount)
	summary += fmt.Sprintf("DoH3 frames (application/dns-message): %d\n", stats.C2SCount+stats.S2CCount)

	if stats.C2SCount > 0 {
		c2sMin, c2sMax := stats.C2SLengths[0], stats.C2SLengths[0]
		for _, l := range stats.C2SLengths {
			if l < c2sMin {
				c2sMin = l
			}
			if l > c2sMax {
				c2sMax = l
			}
		}
		summary += fmt.Sprintf("Client → Server (C2S): %d frames, length range: %d–%d bytes\n",
			stats.C2SCount, c2sMin, c2sMax)
	}

	if stats.S2CCount > 0 {
		s2cMin, s2cMax := stats.S2CLengths[0], stats.S2CLengths[0]
		for _, l := range stats.S2CLengths {
			if l < s2cMin {
				s2cMin = l
			}
			if l > s2cMax {
				s2cMax = l
			}
		}
		summary += fmt.Sprintf("Server → Client (S2C): %d frames, length range: %d–%d bytes\n",
			stats.S2CCount, s2cMin, s2cMax)
	}

	writer.WriteString(summary)

	return stats, nil
}

// processSite walks a site folder, runs analyzePcapFile on each .pcap, and aggregates site-level stats
func processSite(sitePath, siteName string, wg *sync.WaitGroup, results chan<- SiteStats) {
	defer wg.Done()

	siteStats := SiteStats{
		SiteName: siteName,
	}

	logMessage(fmt.Sprintf("Start processing site: %s (%s)", siteName, sitePath))

	files, err := os.ReadDir(sitePath)
	if err != nil {
		logMessage(fmt.Sprintf("Failed to read directory: %s, error: %v", sitePath, err))
		return
	}

	for _, file := range files {
		if file.IsDir() || filepath.Ext(file.Name()) != ".pcap" {
			continue
		}

		pcapPath := filepath.Join(sitePath, file.Name())
		logMessage(fmt.Sprintf("Analyzing file: %s", pcapPath))

		start := time.Now()
		fileStats, err := analyzePcapFile(pcapPath, siteName)
		elapsed := time.Since(start)

		if err != nil {
			logMessage(fmt.Sprintf("Analysis failed: %s, error: %v", pcapPath, err))
			continue
		}

		if fileStats.C2SCount > 0 || fileStats.S2CCount > 0 {
			siteStats.FileStats = append(siteStats.FileStats, fileStats)
			siteStats.C2SCount += fileStats.C2SCount
			siteStats.S2CCount += fileStats.S2CCount
			siteStats.TotalCount += fileStats.TotalCount
			siteStats.TargetCount += fileStats.TargetCount

			for _, length := range fileStats.C2SLengths {
				if siteStats.C2SMinLen == 0 || length < siteStats.C2SMinLen {
					siteStats.C2SMinLen = length
				}
				if length > siteStats.C2SMaxLen {
					siteStats.C2SMaxLen = length
				}
			}

			for _, length := range fileStats.S2CLengths {
				if siteStats.S2CMinLen == 0 || length < siteStats.S2CMinLen {
					siteStats.S2CMinLen = length
				}
				if length > siteStats.S2CMaxLen {
					siteStats.S2CMaxLen = length
				}
			}
		}

		logMessage(fmt.Sprintf("Completed: %s, elapsed: %v, C2S frames: %d, S2C frames: %d",
			pcapPath, elapsed, fileStats.C2SCount, fileStats.S2CCount))
	}

	if len(siteStats.FileStats) > 0 {
		results <- siteStats
	}

	logMessage(fmt.Sprintf("Finished site: %s, total C2S: %d, total S2C: %d",
		siteName, siteStats.C2SCount, siteStats.S2CCount))
}

// createSiteCSV writes per-site CSV summary with English headers
func createSiteCSV(siteStats SiteStats) string {
	siteOutputDir := filepath.Join(OUTPUT_DIR, siteStats.SiteName)
	if err := os.MkdirAll(siteOutputDir, os.ModePerm); err != nil {
		logMessage(fmt.Sprintf("Failed to create directory: %s, error: %v", siteOutputDir, err))
		return ""
	}

	csvPath := filepath.Join(siteOutputDir, siteStats.SiteName+"_dns_summary.csv")
	file, err := os.Create(csvPath)
	if err != nil {
		logMessage(fmt.Sprintf("Failed to create CSV: %s, error: %v", csvPath, err))
		return ""
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	headers := []string{
		"File Name", "Total Frames", "Target Frames",
		"C2S Frames", "C2S Min Length", "C2S Max Length",
		"S2C Frames", "S2C Min Length", "S2C Max Length",
	}
	writer.Write(headers)

	for _, fileStat := range siteStats.FileStats {
		c2sMin, c2sMax := 0, 0
		if len(fileStat.C2SLengths) > 0 {
			c2sMin, c2sMax = fileStat.C2SLengths[0], fileStat.C2SLengths[0]
			for _, l := range fileStat.C2SLengths {
				if l < c2sMin {
					c2sMin = l
				}
				if l > c2sMax {
					c2sMax = l
				}
			}
		}

		s2cMin, s2cMax := 0, 0
		if len(fileStat.S2CLengths) > 0 {
			s2cMin, s2cMax = fileStat.S2CLengths[0], fileStat.S2CLengths[0]
			for _, l := range fileStat.S2CLengths {
				if l < s2cMin {
					s2cMin = l
				}
				if l > s2cMax {
					s2cMax = l
				}
			}
		}

		record := []string{
			fileStat.FileName,
			strconv.Itoa(fileStat.TotalCount),
			strconv.Itoa(fileStat.TargetCount),
			strconv.Itoa(fileStat.C2SCount),
			strconv.Itoa(c2sMin),
			strconv.Itoa(c2sMax),
			strconv.Itoa(fileStat.S2CCount),
			strconv.Itoa(s2cMin),
			strconv.Itoa(s2cMax),
		}

		writer.Write(record)
	}

	if siteStats.C2SCount > 0 || siteStats.S2CCount > 0 {
		writer.Write([]string{
			"TOTAL",
			strconv.Itoa(siteStats.TotalCount),
			strconv.Itoa(siteStats.TargetCount),
			strconv.Itoa(siteStats.C2SCount),
			strconv.Itoa(siteStats.C2SMinLen),
			strconv.Itoa(siteStats.C2SMaxLen),
			strconv.Itoa(siteStats.S2CCount),
			strconv.Itoa(siteStats.S2CMinLen),
			strconv.Itoa(siteStats.S2CMaxLen),
		})
	}

	logMessage(fmt.Sprintf("Site CSV created: %s", csvPath))
	return csvPath
}

// createMasterCSV writes the cross-site summary CSV with English headers
func createMasterCSV(sites []SiteStats) string {
	csvPath := filepath.Join(OUTPUT_DIR, MASTER_CSV)
	file, err := os.Create(csvPath)
	if err != nil {
		logMessage(fmt.Sprintf("Failed to create master CSV: %s, error: %v", csvPath, err))
		return ""
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	headers := []string{
		"Site", "File Count", "Total Frames", "Target Frames",
		"Total C2S Frames", "C2S Min Length", "C2S Max Length",
		"Total S2C Frames", "S2C Min Length", "S2C Max Length",
	}
	writer.Write(headers)

	for _, site := range sites {
		record := []string{
			site.SiteName,
			strconv.Itoa(len(site.FileStats)),
			strconv.Itoa(site.TotalCount),
			strconv.Itoa(site.TargetCount),
			strconv.Itoa(site.C2SCount),
			strconv.Itoa(site.C2SMinLen),
			strconv.Itoa(site.C2SMaxLen),
			strconv.Itoa(site.S2CCount),
			strconv.Itoa(site.S2CMinLen),
			strconv.Itoa(site.S2CMaxLen),
		}
		writer.Write(record)
	}

	logMessage(fmt.Sprintf("Master CSV created: %s", csvPath))
	return csvPath
}

func main() {
	startTime := time.Now()

	if err := os.MkdirAll(OUTPUT_DIR, os.ModePerm); err != nil {
		log.Fatalf("Failed to create output dir: %v", err)
	}

	if err := initLog(); err != nil {
		log.Fatalf("Failed to initialize logger: %v", err)
	}
	defer logFile.Close()

	logMessage(fmt.Sprintf("Scanning root directory: %s", MAIN_DIR))
	logMessage(fmt.Sprintf("DNS resolvers: %v", DNSServers))

	siteDirs, err := os.ReadDir(MAIN_DIR)
	if err != nil {
		logMessage(fmt.Sprintf("Failed to read root directory: %v", err))
		return
	}

	var wg sync.WaitGroup
	results := make(chan SiteStats, len(siteDirs))
	workerSem := make(chan struct{}, MAX_WORKERS)

	for _, siteDir := range siteDirs {
		if !siteDir.IsDir() {
			continue
		}

		siteName := siteDir.Name()
		sitePath := filepath.Join(MAIN_DIR, siteName)

		wg.Add(1)
		workerSem <- struct{}{}

		go func(path, name string) {
			defer func() {
				<-workerSem
			}()
			processSite(path, name, &wg, results)
		}(sitePath, siteName)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var siteStatsList []SiteStats
	for stats := range results {
		createSiteCSV(stats)
		siteStatsList = append(siteStatsList, stats)
	}

	if len(siteStatsList) > 0 {
		createMasterCSV(siteStatsList)
	}

	elapsed := time.Since(startTime)
	logMessage(fmt.Sprintf("Analysis completed. Total elapsed: %v", elapsed))
	logMessage(fmt.Sprintf("Output directory: %s", OUTPUT_DIR))
}
