import subprocess
import time
import re

# ------------------- USER CONFIG -------------------
# Define flows YOU want (not random)
# proto: "tcp", "udp", "ping"
FLOWS = [
    {
        "src": "h1",
        "dst": "h2",
        "proto": "UDP",
        "bandwidth": "10M",   # only for tcp/udp
        "interval": 1,        # for ping only
        "duration": 15
    }
]

REPEAT_FOREVER = True
SLEEP_BETWEEN_FLOWS = 1   # seconds
# ----------------------------------------------------


def discover_hosts():
    """Automatically discover Mininet hosts and their PIDs+IPs."""
    print("[INFO] Discovering Mininet hosts...")

    out = subprocess.check_output(
        "ps aux | grep 'mininet:h' | grep -v grep",
        shell=True
    ).decode()

    hosts = {}

    for line in out.splitlines():
        parts = line.split()
        pid = parts[1]

        match = re.search(r"mininet:(h\d+)", line)
        if match:
            host = match.group(1)
            ip = f"10.0.0.{host[1:]}"
            hosts[host] = {"pid": pid, "ip": ip}

    print("[DISCOVERED HOSTS]:", hosts)
    return hosts


def run_in_ns(pid, cmd):
    """Execute a command inside Mininet namespace using mnexec."""
    full_cmd = f"sudo mnexec -a {pid} {cmd}"
    print("[EXEC]", full_cmd)
    return subprocess.Popen(full_cmd, shell=True)


def start_servers_all(hosts):
    """Start iperf TCP/UDP servers on all hosts."""
    for h, info in hosts.items():
        pid = info["pid"]

        print(f"[INFO] Resetting any old iperf servers on {h}...")

        subprocess.call(f"sudo mnexec -a {pid} pkill iperf", shell=True)

        # Start new servers
        subprocess.Popen(f"sudo mnexec -a {pid} iperf -s &", shell=True)
        subprocess.Popen(f"sudo mnexec -a {pid} iperf -s -u &", shell=True)

        print(f"[OK] Servers ready on {h}")


def start_flow(flow, hosts):
    src = flow["src"]
    dst = flow["dst"]
    proto = flow["proto"].lower()

    src_pid = hosts[src]["pid"]
    dst_ip = hosts[dst]["ip"]

    duration = flow["duration"]

    if proto == "udp":
        bw = flow.get("bandwidth", "10M")
        cmd = f"iperf -c {dst_ip} -u -b {bw} -t {duration} -i 1 &"
        print(f"[UDP] {src} → {dst}: bw={bw}, t={duration}s")
        run_in_ns(src_pid, cmd)

    elif proto == "tcp":
        bw = flow.get("bandwidth", "10M")
        cmd = f"iperf -c {dst_ip} -b {bw} -t {duration} -i 1 &"
        print(f"[TCP] {src} → {dst}: bw={bw}, t={duration}s")
        run_in_ns(src_pid, cmd)

    elif proto == "ping":
        interval = flow.get("interval", 0.1)
        cmd = f"ping {dst_ip} -i {interval} -w {duration} &"
        print(f"[PING] {src} → {dst}: interval={interval}, t={duration}s")
        run_in_ns(src_pid, cmd)

    else:
        print(f"[ERROR] Unknown protocol: {proto}")


def main():
    print("\n==============================================")
    print("       AUTO DISCOVERY TRAFFIC GENERATOR       ")
    print("==============================================\n")

    hosts = discover_hosts()
    start_servers_all(hosts)

    while True:
        for flow in FLOWS:
            start_flow(flow, hosts)
            time.sleep(SLEEP_BETWEEN_FLOWS)

        if not REPEAT_FOREVER:
            break

        print("\n[INFO] Repeating flows again...\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped traffic generator.")
