import numpy as np
import struct
import os
import math

class PyFina(np.ndarray):

    def __new__(cls, id, dir, start, step, npts):
        """
        decoding the .meta file

        id (4 bytes, Unsigned integer)
        npoints (4 bytes, Unsigned integer, Legacy : use instead filesize//4 )
        interval (4 bytes, Unsigned integer)
        start_time (4 bytes, Unsigned integer)

        """
        with open("{}/{}.meta".format(dir,id),"rb") as f:
            f.seek(8,0)
            hexa = f.read(8)
            aa= bytearray(hexa)
            if len(aa)==8:
                decoded=struct.unpack('<2I', aa)
            else:
                print("corrupted meta - aborting")
                return
        meta = {
                 "interval":decoded[0],
                 "start_time":decoded[1],
                 "npoints":os.path.getsize("{}/{}.dat".format(dir,id))//4
               }
        """
        decoding and sampling the .dat file
        values are 32 bit floats, stored on 4 bytes
        to estimate value(time), position in the dat file is calculated as follow :
        pos = (time - meta["start_time"]) // meta["interval"]
        Nota : no NAN value - if a NAN is detected, the algorithm will fetch the first non NAN value in the future
        """
        verbose = False
        obj = np.zeros(npts).view(cls)

        end = start + (npts-1) * step
        time = start
        i = 0
        with open("{}/{}.dat".format(dir,id), "rb") as ts:
            while time < end:
                time = start + step * i
                pos = (time - meta["start_time"]) // meta["interval"]
                if pos >=0 and pos < meta["npoints"]:
                    #print("trying to find point {} going to index {}".format(i,pos))
                    ts.seek(pos*4, 0)
                    hexa = ts.read(4)
                    aa= bytearray(hexa)
                    if len(aa)==4:
                      value=struct.unpack('<f', aa)[0]
                      if not math.isnan(value):
                          obj[i] = value
                      else:
                          if verbose:
                              print("NAN at pos {} uts {}".format(pos, meta["start_time"]+pos*meta["interval"]))
                          j=1
                          while True:
                              #print(j)
                              ramble=(pos+j)*4
                              ts.seek(ramble, 0)
                              hexa = ts.read(4)
                              aa= bytearray(hexa)
                              value=struct.unpack('<f', aa)[0]
                              if math.isnan(value):
                                  j+=1
                              else:
                                  break
                          obj[i] = value
                    else:
                      print("unpacking problem {} len is {} position is {}".format(i,len(aa),position))
                i += 1
        """
        storing the "signature" of the "sampled" feed
        """
        obj.start = start
        obj.step = step

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.start = getattr(obj, 'start', None)
        self.step = getattr(obj, 'step', None)

    def timescale(self):
        """
        return the time scale of the feed as a numpy array
        """
        return np.arange(0,self.step*self.shape[0],self.step)
