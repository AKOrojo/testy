import os
import csv
import json
import ollama
import subprocess
import sys
import logging
import re

# --- Configuration ---

# Directory where your single-flow PCAP files are stored.
# The script will create this directory if it doesn't exist.
PCAP_DIRECTORY = "/home/abanisenioluwa_oroj1/PycharmProjects/ByteFlow/preprocessing/flow_split/data/flows"

# The name of the CSV file where results will be saved.
OUTPUT_CSV_FILE = "pcap_analysis_results.csv"

# List of Ollama models to query for analysis.
# Make sure you have pulled these models with `ollama pull <model_name>`
MODELS_TO_QUERY = [
    "gemma3:27b",
    "qwen3:32b",
    "llama3.3:70b",
]

# The prompt template that will be sent to the models.
PROMPT_TEMPLATE = """
### ROLE
You are an expert network analyst. Your task is to generate a detailed, structured analysis of a benign network communication flow. This data will be used to train a new AI model, so the output must be highly detailed, accurate, and easy to parse.

### TASK
Analyze the provided network flow data from tshark json. First, provide a step-by-step reasoning process. Then, produce a structured report in Markdown format. Finally, write a concise summary paragraph that synthesizes all the information.

---
### 1. REASONING (Chain of Thought)
Before generating the report, detail your thought process here. Explicitly mention the key packets or data points from the JSON that justify your conclusions. For example:
- "I see a packet with `tcp.flags.syn == 1`... confirming a successful connection."
- "I observe an `http.request.method == GET` in packet #4. The `http.host` field shows the client is contacting 'www.example.com'."
- "In packet #6, I see the `dns.qry.name` is 'api.google.com', indicating a DNS query."
- "In the response packet, I can see the `http.content_type` is 'application/json', and I can summarize the structure of the JSON payload."

---
### 2. STRUCTURED REPORT
Provide the analysis in the following Markdown format. Fill in every field based on the data.

#### Flow Identification
- **Protocol:** (e.g., TCP, UDP)
- **Source:** (e.g., 192.168.1.10:54321)
- **Destination:** (e.g., 93.184.216.34:80)

#### Timeline & Volume
- **Start Timestamp:** (Extract from the first packet)
- **End Timestamp:** (Extract from the last packet)
- **Duration (seconds):**
- **Total Packets:**
- **Bytes (Client to Server):**
- **Bytes (Server to Client):**

#### Key Events
- **Connection Setup:** (e.g., "Successful 3-way handshake (SYN, SYN-ACK, ACK) observed.")
- **Application Data:** (e.g., "Client sent an HTTP/1.1 GET request for '/index.html'.")
- **Server Response:** (e.g., "Server responded with an HTTP/1.1 200 OK.")
- **Session Teardown:** (e.g., "Connection closed via a 4-way handshake (FIN/ACK, FIN/ACK).")

#### Payload Summary
- **Content Type:** (e.g., HTTP, DNS, TLS Handshake, Unknown)
- **Unencrypted Data Summary:** (Summarize any clear-text data. For example: "The client performed a DNS A record lookup for 'example.com'. The server responded with the IP address 93.184.216.34." or "The client requested the webpage '/news/latest' from the host 'global-news.com'. The server returned a 2,480-byte HTML document.")

---
### 3. FINAL SUMMARY
Synthesize all the information from the structured report, including the payload summary, into a single, coherent paragraph. This paragraph(s) should be a self-contained summary of the entire flow.

---
{flow_json}
"""

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_json_from_pcap(pcap_path):
    """
    Converts an entire PCAP file to a JSON string using tshark.

    Args:
        pcap_path (str): The path to the PCAP file.

    Returns:
        str: A JSON string representing all packets in the file, or None on error.
    """
    logging.info(f"Converting {os.path.basename(pcap_path)} to JSON...")
    tshark_cmd = ['tshark', '-r', pcap_path, '-T', 'json']
    try:
        process = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        return process.stdout
    except FileNotFoundError:
        logging.error(
            "tshark command not found. Please ensure Wireshark/tshark is installed and in your system's PATH.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"tshark returned an error for {pcap_path}: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {pcap_path}: {e}")
        return None


def analyze_flow_with_model(flow_json_str, model_name):
    """
    Sends flow data to an Ollama model, gets the analysis, and filters out the reasoning.

    Args:
        flow_json_str (str): JSON string representing the flow.
        model_name (str): Name of the Ollama model.

    Returns:
        str: The filtered analysis text from the model, or an error message.
    """
    full_prompt = PROMPT_TEMPLATE.format(flow_json=flow_json_str)
    try:
        logging.info(f"Querying model: {model_name}...")
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': full_prompt}]
        )

        analysis_text = response['message']['content'].strip()

        # **MODIFICATION START**
        # 1. Filter out any <think>...</think> blocks from the response.
        # The re.DOTALL flag allows '.' to match newline characters.
        analysis_text = re.sub(r'<think>.*?</think>', '', analysis_text, flags=re.DOTALL).strip()

        # 2. Find the start of the main report to remove the preceding "REASONING" section.
        # We look for "STRUCTURED REPORT" or similar headers as the starting point.
        report_keywords = [
            "### 2. STRUCTURED REPORT",
            "### STRUCTURED REPORT",
            "### **Analysis of the Provided DNS Traffic**"
        ]

        report_start_index = -1
        for keyword in report_keywords:
            found_index = analysis_text.find(keyword)
            if found_index != -1:
                report_start_index = found_index
                break

        # If a keyword is found, slice the text to keep everything from that point onward.
        if report_start_index != -1:
            analysis_text = analysis_text[report_start_index:]
        # **MODIFICATION END**

        return analysis_text.strip()

    except Exception as e:
        logging.error(f"Error querying model {model_name}: {e}")
        return f"Error: Could not get a response from model {model_name}."


def main():
    """
    Main function to orchestrate the PCAP analysis process.
    """
    if not os.path.isdir(PCAP_DIRECTORY):
        logging.info(f"Directory '{PCAP_DIRECTORY}' not found. Creating it.")
        os.makedirs(PCAP_DIRECTORY)
        logging.info(f"Please place your single-flow .pcap files in the '{PCAP_DIRECTORY}' and run again.")
        sys.exit(0)

    pcap_files = [f for f in os.listdir(PCAP_DIRECTORY) if f.endswith((".pcap", ".pcapng"))]

    if not pcap_files:
        logging.warning(f"No .pcap or .pcapng files found in '{PCAP_DIRECTORY}'. Exiting.")
        sys.exit(0)

    logging.info(f"Found {len(pcap_files)} PCAP flow files to analyze.")

    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['flow_file', 'model', 'analysis_text']
        csv_writer.writerow(header)

        for pcap_filename in pcap_files:
            pcap_path = os.path.join(PCAP_DIRECTORY, pcap_filename)
            flow_json_str = get_json_from_pcap(pcap_path)

            if not flow_json_str:
                logging.warning(f"Skipping {pcap_filename} due to a processing error.")
                continue

            for model_name in MODELS_TO_QUERY:
                analysis = analyze_flow_with_model(flow_json_str, model_name)
                row = [pcap_filename, model_name, analysis]
                csv_writer.writerow(row)
                csvfile.flush()

    logging.info(f"Analysis complete. Results have been saved to '{OUTPUT_CSV_FILE}'.")


if __name__ == "__main__":
    main()
