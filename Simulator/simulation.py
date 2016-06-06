import ReCAM


def simulation():
    tmp=ReCAM.Simulator.ReCAM(1000);
    print "size in bytes = ", tmp.sizeInBytes
    print "bits per row = ", tmp.bitsPerRow

simulation()