package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gopacket/gopacket"
	"github.com/gopacket/gopacket/layers"
	"github.com/gopacket/gopacket/pcap"
)

// ================= Configuration =================
//
// PCAP_DIR:    Root folder containing .pcap files (site subfolders allowed)
// OUTPUT_DIR:  Directory where CSV files will be written
// MAX_WORKERS: Max number of concurrent goroutines (suggest ~ CPU cores * 2)
//
// =================================================
const (
	PCAP_DIR    = "captures-onlydoh3"
	OUTPUT_DIR  = "captures_csv_onlydoh3"
	MAX_WORKERS = 12
)

// Target DNS resolver IPs
var TARGET_ADDRESSES = map[string]bool{
	"8.8.8.8":              true,
	"8.8.4.4":              true,
	"2001:4860:4860::8844": true,
	"2001:4860:4860::8888": true,
}

// PacketInfo represents extracted information for one packet
type PacketInfo struct {
	RelativeTime string
	IntervalTime string
	SrcIP        string
	SrcPort      string
	DstIP        string
	DstPort      string
	Length       int
	Direction    int // 1: to target, 0: from target
}

// processPcapFile parses one .pcap and writes a CSV with extracted packet info
func processPcapFile(pcapPath, csvPath string) (int, error) {
	// Ensure output directory exists
	if err := os.MkdirAll(filepath.Dir(csvPath), 0755); err != nil {
		return 0, fmt.Errorf("failed to create directory: %w", err)
	}

	// Open CSV file for writing
	csvFile, err := os.Create(csvPath)
	if err != nil {
		return 0, fmt.Errorf("failed to create CSV file: %w", err)
	}
	defer csvFile.Close()

	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()

	// Write CSV header
	if err := csvWriter.Write([]string{
		"RelativeTime(s)", "IntervalTime(s)", "SrcIP", "SrcPort",
		"DstIP", "DstPort", "Length(Bytes)", "Direction",
	}); err != nil {
		return 0, fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Open pcap file
	handle, err := pcap.OpenOffline(pcapPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open PCAP file: %w", err)
	}
	defer handle.Close()

	packetCount := 0
	var firstPacketTime time.Time
	var prevPacketTime time.Time
	packetSource := gopacket.NewPacketSource(handle, handle.LinkType())

	// Process packets
	for packet := range packetSource.Packets() {
		networkLayer := packet.NetworkLayer()
		if networkLayer == nil {
			continue
		}

		transportLayer := packet.TransportLayer()
		if transportLayer == nil || transportLayer.LayerType() != layers.LayerTypeUDP {
			continue
		}

		udp, _ := transportLayer.(*layers.UDP)
		srcPort := strconv.Itoa(int(udp.SrcPort))
		dstPort := strconv.Itoa(int(udp.DstPort))

		// Only keep QUIC packets (UDP port 443)
		if srcPort != "443" && dstPort != "443" {
			continue
		}

		srcIP := networkLayer.NetworkFlow().Src().String()
		dstIP := networkLayer.NetworkFlow().Dst().String()

		// Keep packets only if src or dst is in TARGET_ADDRESSES
		_, srcIsTarget := TARGET_ADDRESSES[srcIP]
		_, dstIsTarget := TARGET_ADDRESSES[dstIP]
		if !srcIsTarget && !dstIsTarget {
			continue
		}

		timestamp := packet.Metadata().Timestamp

		// Initialize reference timestamps
		if firstPacketTime.IsZero() {
			firstPacketTime = timestamp
			prevPacketTime = timestamp
		}

		relativeTime := timestamp.Sub(firstPacketTime).Seconds()
		intervalTime := timestamp.Sub(prevPacketTime).Seconds()
		prevPacketTime = timestamp

		// Determine direction (1: client->target, 0: target->client)
		direction := 0
		if dstIsTarget {
			direction = 1
		}

		info := PacketInfo{
			RelativeTime: fmt.Sprintf("%.6f", relativeTime),
			IntervalTime: fmt.Sprintf("%.6f", intervalTime),
			SrcIP:        srcIP,
			SrcPort:      srcPort,
			DstIP:        dstIP,
			DstPort:      dstPort,
			Length:       packet.Metadata().Length,
			Direction:    direction,
		}

		row := []string{
			info.RelativeTime,
			info.IntervalTime,
			info.SrcIP,
			info.SrcPort,
			info.DstIP,
			info.DstPort,
			strconv.Itoa(info.Length),
			strconv.Itoa(info.Direction),
		}

		if err := csvWriter.Write(row); err != nil {
			return packetCount, fmt.Errorf("failed to write CSV row: %w", err)
		}

		packetCount++
	}

	return packetCount, nil
}

// findPcapFiles recursively finds all .pcap files
func findPcapFiles(rootDir string) ([]string, error) {
	var pcapFiles []string

	err := filepath.Walk(rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(strings.ToLower(path), ".pcap") {
			pcapFiles = append(pcapFiles, path)
		}
		return nil
	})

	return pcapFiles, err
}

// getRelativePath returns relative path of file against base dir
func getRelativePath(fullPath, baseDir string) (string, error) {
	relPath, err := filepath.Rel(baseDir, fullPath)
	if err != nil {
		return "", err
	}
	return relPath, nil
}

func main() {
	startTime := time.Now()
	fmt.Println("PCAP to CSV Tool - Go Implementation")
	fmt.Printf("Start time: %s\n", startTime.Format("2006-01-02 15:04:05"))
	fmt.Printf("Input dir: %s\nOutput dir: %s\n", PCAP_DIR, OUTPUT_DIR)
	fmt.Printf("Target addresses: %v\n", TARGET_ADDRESSES)
	fmt.Printf("Max workers: %d\n", MAX_WORKERS)
	fmt.Println("=" + strings.Repeat("-", 60))

	// Find all pcap files
	pcapFiles, err := findPcapFiles(PCAP_DIR)
	if err != nil {
		log.Fatalf("Error finding PCAP files: %v", err)
	}

	if len(pcapFiles) == 0 {
		fmt.Printf("No PCAP files found under %s\n", PCAP_DIR)
		return
	}

	fmt.Printf("Found %d PCAP files\n", len(pcapFiles))

	// Worker pool
	jobs := make(chan string, len(pcapFiles))
	results := make(chan struct {
		path        string
		csvPath     string
		packetCount int
		err         error
	}, len(pcapFiles))

	var wg sync.WaitGroup

	for i := 0; i < MAX_WORKERS; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for pcapPath := range jobs {
				relPath, err := getRelativePath(pcapPath, PCAP_DIR)
				if err != nil {
					results <- struct {
						path        string
						csvPath     string
						packetCount int
						err         error
					}{pcapPath, "", 0, err}
					continue
				}

				csvRelPath := strings.TrimSuffix(relPath, filepath.Ext(relPath)) + ".csv"
				csvPath := filepath.Join(OUTPUT_DIR, csvRelPath)

				packetCount, err := processPcapFile(pcapPath, csvPath)
				results <- struct {
					path        string
					csvPath     string
					packetCount int
					err         error
				}{pcapPath, csvPath, packetCount, err}
			}
		}(i)
	}

	// Submit jobs
	for _, file := range pcapFiles {
		jobs <- file
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	processedFiles := 0
	skippedFiles := 0
	totalPackets := 0
	successFiles := 0
	errorFiles := 0

	fmt.Println("\nProcessing files:")
	fmt.Println(strings.Repeat("-", 100))

	for result := range results {
		if result.err != nil {
			fmt.Printf("[!] Failed: %-60s -> Error: %v\n", filepath.Base(result.path), result.err)
			errorFiles++
		} else if result.packetCount == 0 {
			fmt.Printf("[ ] Skipped: %-60s (0 QUIC packets)\n", filepath.Base(result.path))
			skippedFiles++
		} else {
			fmt.Printf("[✓] Success: %-60s -> %-60s (%d packets)\n",
				filepath.Base(result.path), filepath.Base(result.csvPath), result.packetCount)
			successFiles++
			totalPackets += result.packetCount
		}
		processedFiles++
		progress := float64(processedFiles) / float64(len(pcapFiles)) * 100
		fmt.Printf("Progress: %d/%d (%.1f%%) | Success: %d | Skipped: %d | Errors: %d | Packets: %d\n",
			processedFiles, len(pcapFiles), progress, successFiles, skippedFiles, errorFiles, totalPackets)
	}

	duration := time.Since(startTime)
	hours := int(duration.Hours())
	minutes := int(duration.Minutes()) % 60
	seconds := int(duration.Seconds()) % 60

	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("Processing complete!")
	fmt.Printf("Elapsed: %02d:%02d:%02d\n", hours, minutes, seconds)
	fmt.Printf("Total files: %d\n", len(pcapFiles))
	fmt.Printf("Processed successfully: %d\n", successFiles)
	fmt.Printf("Skipped files: %d (no QUIC packets)\n", skippedFiles)
	fmt.Printf("Failed files: %d\n", errorFiles)
	fmt.Printf("Total packets processed: %d\n", totalPackets)
	fmt.Printf("CSV output directory: %s\n", OUTPUT_DIR)
	fmt.Println(strings.Repeat("=", 100))
}
