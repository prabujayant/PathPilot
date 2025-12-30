# features.py
import numpy as np
from topology import LINKS

def extract_features(stats_json):
    """
    Output shape: [num_links * 4]
    Features per link:
    - tx_bps
    - rx_bps
    - tx_dropped
    - queue_len
    """

    features = []

    for (src, dst) in LINKS:
        # Map logical link to switch + port
        # This assumes your topology mapping
        dpid = src.replace("s", "")
        port = "1"  # adapt if needed

        port_stats = stats_json.get(dpid, {}).get(port, {})

        features.extend([
            port_stats.get("tx_bps", 0),
            port_stats.get("rx_bps", 0),
            port_stats.get("tx_dropped", 0),
            port_stats.get("queue_len", 0),
        ])

    return np.array(features, dtype=np.float32)
