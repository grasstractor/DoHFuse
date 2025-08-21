# Feature Extraction Scripts

This folder contains Python scripts for extracting features from network traffic data for website fingerprinting research. Each script focuses on a different type of feature or extraction method. Below is a brief description of each file:

## File Overview

- **extract_intervaltimeandstat.py**
  - Extracts packet interval time features and statistical features from raw network traffic data.
  - Outputs processed features suitable for machine learning models.

- **extract_odoh3.py**
  - The feature extraction is based on the methodology described in the paper "Privacy Analysis of Oblivious DNS over HTTPS: a Website Fingerprinting Study".
  - Focuses on protocol-specific characteristics and relevant statistics.

- **extract_packet_length.py**
  - Extracts packet length features from network traffic data.
  - Provides signed packet length sequences and related statistics.

## Usage

Each script is designed to be run independently. Please refer to the comments and argument parsers within each script for specific usage instructions, required input formats, and configurable parameters.

## Notes

- The extracted features are intended for use in downstream machine learning models for website fingerprinting.
- For model training and evaluation, refer to the corresponding folders and scripts in the main project directory.

---
For further details, see the code comments in each script or contact the project maintainer.
