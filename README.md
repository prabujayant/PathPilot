# 🚀 Mininet Multipath Traffic Visualizer

A complete setup to simulate a **multipath SDN (Software-Defined
Network)** using **Mininet + POX**, generate traffic across multiple
host pairs, and visualize **live link utilization** using a **D3.js
interactive dashboard**.

## 📦 1. Requirements

Install core dependencies:

``` bash
sudo apt update
sudo apt install mininet python3 python3-pip
```

Install POX in your home directory:

``` bash
git clone https://github.com/noxrepo/pox.git
```

## 📁 2. Project Structure

    /your-repo
    │
    ├── multipath.py
    ├── traffic_controller.py
    ├── proxy.py
    ├── topology.json
    ├── visualizer.html
    │
    └── pox/
        └── ext/
            ├── linkstats.py
            └── random_multipath.py

Ensure the POX modules are placed in:

    ~/pox/pox/ext/

## 🛰 3. Step 1: Start the POX Controller

``` bash
cd ~/pox
./pox.py openflow.discovery ext.linkstats ext.random_multipath
```

## 🖧 4. Step 2: Start Mininet Topology

``` bash
sudo mn --custom multipath.py --topo multipath --controller=remote,ip=127.0.0.1
```

## 🔌 5. Step 3: Start Traffic Generator

``` bash
sudo python3 traffic_controller.py
```

## 🌐 6. Step 4: Start the Stats Proxy

``` bash
python3 proxy.py
```

Stats available at:

    http://localhost:8001/stats

## 📊 7. Step 5: Open the Visualizer

Open:

    visualizer.html

## 🧠 8. How It Works

### POX Modules

-   `linkstats.py` -- Polls switches and stores stats.
-   `random_multipath.py` -- Installs random multipath routes.

### Traffic Generator

-   Auto-discovers hosts\
-   Starts iperf servers\
-   Generates continuous flows

### Visualizer

-   Reads topology.json\
-   Fetches live stats\
-   Updates link color & thickness
