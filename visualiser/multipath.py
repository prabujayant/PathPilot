from mininet.topo import Topo

class MultiPathTopo(Topo):
    def build(self):

        # Hosts
        h1 = self.addHost('h1')   # Source
        h2 = self.addHost('h2')   # Destination

        # Switches (3 parallel paths)
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        s4 = self.addSwitch('s4')

        # Path 1: h1 -> s1 -> s4 -> h2
        self.addLink(h1, s1)
        self.addLink(s1, s4)

        # Path 2: h1 -> s2 -> s4 -> h2
        self.addLink(h1, s2)
        self.addLink(s2, s4)

        # Path 3: h1 -> s3 -> s4 -> h2
        self.addLink(h1, s3)
        self.addLink(s3, s4)

        # Final hop to destination
        self.addLink(s4, h2)

# Required dictionary for Mininet to load the custom topo
topos = { 'multipath': MultiPathTopo }
