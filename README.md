🚀 Mininet Multipath Traffic Visualizer

A complete setup to simulate a multipath software-defined network (SDN) using Mininet + POX, generate traffic between hosts, and visualize live link load in a D3.js dashboard.

This project includes:

✓ Multipath Mininet Topology
✓ POX Controller with Random Multipath Forwarding
✓ Link Statistics Collector (linkstats.py)
✓ Stats Proxy Server (proxy.py)
✓ Traffic Generator with Auto-Discovery (traffic_controller.py)
✓ Live D3.js Visualization (visualizer.html)

📦 1. Requirements

Install:

sudo apt update
sudo apt install mininet python3 python3-pip


Install POX (inside your home directory):

git clone https://github.com/noxrepo/pox.git

📁 2. Project Structure

Your cloned repo should look like:

/your-repo
│
├── multipath.py              # Mininet topology
├── traffic_controller.py     # Auto host discovery + flow generator
├── proxy.py                  # Exposes POX stats on port 8001
├── topology.json             # Used by D3 visualizer
├── visualizer.html           # Browser UI
│
└── pox/                      # POX controller folder
    └── ext/
        ├── linkstats.py      # Collects port stats into pox_stats.json
        └── random_multipath.py  # Random multipath routing module


Make sure linkstats.py and random_multipath.py are placed inside:

~/pox/pox/ext/

🛰 3. Step 1: Start POX Controller

Open Terminal #1:

cd ~/pox
./pox.py openflow.discovery ext.linkstats ext.random_multipath

🖧 4. Step 2: Start Mininet Topology

Open Terminal #2 inside your cloned repo:

sudo mn --custom multipath.py --topo multipath --controller=remote,ip=127.0.0.1

🔌 5. Step 3: Start Traffic Generator (Auto Discovery)

Open Terminal #3 in your repo:

sudo python3 traffic_controller.py


This script:

Auto-discovers all Mininet host PIDs

Starts iperf TCP + UDP servers on each host

Sends flows you defined inside FLOWS = [ … ]

Repeats them forever if enabled

Example output:

[DISCOVERED HOSTS]: {'h1': {...}, 'h2': {...}}
[TCP] h1 → h2: bw=10M, t=15s

🌐 6. Step 4: Start Stats Proxy Server

Open Terminal #4:

python3 proxy.py


The proxy exposes POX link statistics as:

http://localhost:8001/stats


Used by the frontend (browser).

📊 7. Step 5: Open the Visualizer

Open:

visualizer.html


You will see:

Nodes = hosts + switches

Links colored by current traffic load

Line thickness increases with load

Colors update every 1 second

Traffic is visible only after traffic_controller.py generates flows.

🧠 8. How It Works
POX Modules:

linkstats.py
Polls all OpenFlow switches every 1 second
→ stores stats in pox_stats.json

random_multipath.py
Installs a random path:
s1→s4 OR s2→s4 OR s3→s4
for every new TCP/UDP/ICMP flow

Traffic Generator:

Discovers host namespaces

Starts iperf servers

Sends controlled flows between hosts

Produces sustained traffic for visualization or ML training

Visualizer:

Reads topology.json

Fetches stats from proxy

Draws colored, animated links:

Color scale:

Blue → Idle

Orange → Light

Red → Heavy