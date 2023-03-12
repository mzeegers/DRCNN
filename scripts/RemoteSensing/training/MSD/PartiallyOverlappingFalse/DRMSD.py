import numpy as np
import msdnet
#import pyqtgraph as pq
import time

from numba import cuda, float32, int32

class DataReductionMSDNet(msdnet.network.MSDNet):
    def __init__(self, reductionlist, d, dil, nin, nout, dosegmentation=True, gpu = True):
        self.gpu = gpu
        if(isinstance(reductionlist, int)):
            reductionlist = [reductionlist]
        super().__init__(d,dil,reductionlist[-1],nout, gpu)
        self.redw = []
        self.redw.append(np.zeros((nin,reductionlist[0]),dtype=np.float32))
        for i in range(1,len(reductionlist)):
            self.redw.append(np.zeros((reductionlist[i-1],reductionlist[i]),dtype=np.float32))
        self.redb = []
        for red in reductionlist:
            self.redb.append(np.zeros(red,dtype=np.float32))
        
        self.redwg = [np.zeros_like(redw) for redw in self.redw]
        self.redbg = [np.zeros_like(redb) for redb in self.redb]
        self.redlist = reductionlist
        # self.tempim = pq.image()
        self.dosegmentation = dosegmentation
    
    def forward(self, im, Timer = False):
        
        start = time.monotonic()

        #TODO: convert below to GPU code -> DONE
        self.redim = []
        self.redim.append(im.copy())
        for red in self.redlist:
            self.redim.append(np.zeros((red,*im.shape[1:]),dtype=np.float32))

        for i in range(1,len(self.redim)):

            if(self.gpu == False):
                ###Apply linear combination operations
                for j in range(self.redim[i].shape[0]):
                    self.redim[i][j].fill(self.redb[i-1][j])
                    for k in range(self.redim[i-1].shape[0]):
                        msdnet.operations.combine(self.redim[i-1][k], self.redim[i][j], self.redw[i-1][k][j])
                        # self.redim[i][j] += self.redim[i-1][k]*self.redw[i-1][k][j]
                msdnet.operations.leakyrelu(self.redim[i],0.01)

            else:
                
                for j in range(self.redim[i].shape[0]):
                    self.redim[i][j].fill(self.redb[i-1][j])
                A = self.redim[i]
                B = self.redim[i-1]
                
                #Combine (=weighted average) all images
                bpg2d, tpb2d = get2dgridsize((A.shape[-2],A.shape[-1]))
                Bc = cuda.to_device(B)
                Ac = cuda.to_device(A)
                wg = cuda.to_device(self.redw[i-1])
                comb_all_all_cuda[bpg2d, tpb2d](Bc, Ac, wg)

                #Apply leaky relu
                bpg, tpb = get1dgridsize(A.size)
                leaky_relu2d_cuda[bpg, tpb](Ac.ravel(), 0.01)
                self.redim[i] = Ac.copy_to_host()
        
        '''
        ##Alternative implementation
        if(self.gpu == False):
            for i in range(1,len(self.redim)):
                ###Apply linear combination operations
                for j in range(self.redim[i].shape[0]):
                    self.redim[i][j].fill(self.redb[i-1][j])
                    for k in range(self.redim[i-1].shape[0]):
                        msdnet.operations.combine(self.redim[i-1][k], self.redim[i][j], self.redw[i-1][k][j])
                        # self.redim[i][j] += self.redim[i-1][k]*self.redw[i-1][k][j]
                msdnet.operations.leakyrelu(self.redim[i],0.01)

        else:

            #Put everything on GPU
            Ac = [cuda.to_device(self.redim[0])]
            wg = []
            for i in range(1,len(self.redim)):
                for j in range(self.redim[i].shape[0]):
                    self.redim[i][j].fill(self.redb[i-1][j])
                Ac.append(cuda.to_device(self.redim[i]))
                wg.append(cuda.to_device(self.redw[i-1]))

            #Perform computation
            for i in range(1,len(self.redim)):
                
                #Combine (=weighted average) all images
                bpg2d, tpb2d = get2dgridsize((self.redim[i].shape[-2],self.redim[i].shape[-1]))
                comb_all_all_cuda[bpg2d, tpb2d](Ac[i-1], Ac[i], wg[i-1])

                #Apply leaky relu
                bpg, tpb = get1dgridsize(self.redim[i].size)
                leaky_relu2d_cuda[bpg, tpb](Ac[i].ravel(), 0.01)
                self.redim[i] = Ac[i].copy_to_host()'''
        #until here
        
        if(Timer == True):
            start = time.monotonic()
        out = super().forward(self.redim[-1])
        
        if(Timer == True):
            diff = time.monotonic() - start

        if self.dosegmentation:
            self.out.softmax()
            out = self.out.copy()
        # self.tempim.setImage(np.vstack([im.transpose() for im in self.redim[-1]]))
        # print(np.vstack([im for im in self.redim[-1]]).min(),np.vstack([im for im in self.redim[-1]]).max())
        # pq.QtGui.QApplication.processEvents()
        # for redim in self.redim:
        #     pq.image(redim)
        # print(self.redim[-1][0].min(),self.redim[-1][0].max())
        # print(self.redim[-1][1].min(),self.redim[-1][1].max())
        # print(self.redim[-1][2].min(),self.redim[-1][2].max())
        # print(self.redim[-1][3].min(),self.redim[-1][3].max())
        print("Step FORWARD time" , time.monotonic()-start)
        if(Timer == True):
            return out, diff        
        return out
    
    def backward(self, im):
        start = time.process_time()
        super().backward(im)
        #TODO: convert below to GPU code
        do = self.deltaout.copy()
        delta = self.delta.copy()
        self.reddel = []
        for red in self.redlist:
            self.reddel.append(np.zeros((red,*im.shape[1:]),dtype=np.float32))

        for i in range(self.redlist[-1]):   #TODO: Deze functies (en in het bijzonder conv2d) moeten geupdate worden
            
            if(self.gpu == False):
                for j in range(do.shape[0]):
                    msdnet.operations.combine(do[j], self.reddel[-1][i], self.w[j][i])
            else:

                bpg2d, tpb2d = get2dgridsize((self.reddel[-1][i].shape[-2], self.reddel[-1][i].shape[-1]))
                A1c = cuda.to_device(self.reddel[-1][i])
                B1c = cuda.to_device(do)
                W1c = cuda.to_device(self.w[:,i].copy())
                
                comb_all_cuda[bpg2d, tpb2d](B1c, A1c, W1c)
                self.reddel[-1][i] = A1c.copy_to_host()


            fb = np.zeros((self.d,*self.fshape),dtype=np.float32)
            for j in range(self.d):
                fb[j] = self.f[j][i][self.revf]
            for j in range(self.d):
                msdnet.operations.conv2d(delta[j], self.reddel[-1][i], fb[j], self.dl[j])
            # print(self.reddel[-1][i].min(),self.reddel[-1][i].max())
            # input()


            if(self.gpu == False):
                msdnet.operations.leakyrelu2(self.redim[-1][i], self.reddel[-1][i], 0.01)
            else:       
                bpg, tpb = get1dgridsize(self.reddel[-1][i].size)
                Ac = cuda.to_device(self.reddel[-1][i])
                Bc = cuda.to_device(self.redim[-1][i])
                leaky_relu2_cuda[bpg, tpb](Bc.ravel(), Ac.ravel(), 0.01)           
                self.reddel[-1][i] = Ac.copy_to_host()
        print("Step BACKWARD time" , time.process_time()-start)
        

        if(False):
            for i in reversed(range(len(self.reddel)-1)):
                # print(i, self.redlist[i], self.redlist[i+1], self.reddel[i].shape, self.redw[i+1].shape, self.redim[i+2].shape)
                
                Ac = cuda.to_device(self.reddel[i])
                Bc = cuda.to_device(self.reddel[i+1])
                Wc = cuda.to_device(self.redw[i+1])

                bpg2d, tpb2d = get2dgridsize((self.reddel[i].shape[-2], self.reddel[i].shape[-1]))
                comb_all_all_cuda[bpg2d, tpb2d](Bc, Ac, Wc)

                    
                bpg, tpb = get1dgridsize(self.reddel[i].size)
                Cc = cuda.to_device(self.redim[i+1])
                leaky_relu2_cuda[bpg, tpb](Cc.ravel(), Ac.ravel(), 0.01)
                self.reddel[i] = Ac.copy_to_host() 

        else:
            for i in reversed(range(len(self.reddel)-1)):
                for j in range(self.redlist[i]):
                    for k in range(self.redlist[i+1]):
                        msdnet.operations.combine(self.reddel[i+1][k], self.reddel[i][j], self.redw[i+1][j][k])
                    # self.reddel[i][j] += self.redw[i+1][j][k] * self.redim[i+2][k]
                    msdnet.operations.leakyrelu2(self.redim[i+1][j], self.reddel[i][j], 0.01)
        # import pyqtgraph as pq
        # pq.image(self.reddel[-1])
        # # pq.image(self.delta)
        # input()
        print("Step GRADIENT time" , time.process_time()-start)

    def initialize(self):
        super().initialize()
        for redw in self.redw:
            redw[:] = np.sqrt(2/(redw.shape[0]+redw.shape[1]))*np.random.normal(size=redw.shape)
        for redb in self.redb:
            redb[:] = 0


    def gradient_zero(self):
        super().gradient_zero()
        for redwg in self.redwg:
            redwg[:]=0
        for redbg in self.redbg:
            redbg[:]=0
    
    def gradient(self):
        start = time.process_time()
        super().gradient()
        if(self.gpu == False):
        #TODO: convert below to GPU code
            for i in range(len(self.redlist)):
                for j in range(self.redbg[i].shape[0]):
                    self.redbg[i][j] += msdnet.operations.sum(self.reddel[i][j])
            for i in range(len(self.redlist)):
                for j in range(self.redwg[i].shape[0]):
                    for k in range(self.redwg[i].shape[1]):
                        self.redwg[i][j][k] += msdnet.operations.multsum(self.redim[i][j], self.reddel[i][k])
                        # self.redwg[i][j][k] += (self.redim[i][j]*self.reddel[i][k]).sum()
                        # print(self.redim[i][j].min(),self.redim[i][j].max(),self.reddel[i][k].min(),self.reddel[i][k].max())
            # for redwg in self.redwg:
            #     print(redwg.min(),redwg.max())
            # print(self.wg)
        else:

            for i in range(len(self.redlist)): #TODO: Find a way to do this without looping
                
                tmp = cuda.device_array(24*self.reddel[i].shape[0])
                Rdi = cuda.to_device(self.reddel[i])
                fastsumall[24,1024](Rdi,tmp)
                self.redbg[i] += tmp.copy_to_host().reshape((self.reddel[i].shape[0],24)).sum(1)

                tmp = cuda.device_array(24*self.reddel[i].shape[0]*self.redim[i].shape[0])
                Rdi = cuda.to_device(self.reddel[i])
                Redmimi = cuda.to_device(self.redim[i])
                fastmult[24,1024](Redmimi,Rdi,tmp)
                self.redwg[i] += tmp.copy_to_host().reshape((self.redim[i].shape[0],self.reddel[i].shape[0],24)).sum(2)
    
    def getgradients(self):
        curu = super().getgradients()
        redwgu = np.hstack([redwg.ravel() for redwg in self.redwg])
        redbgu = np.hstack([redbg.ravel() for redbg in self.redbg])
        return np.hstack([curu,redwgu.ravel(),redbgu.ravel()]).ravel()
    
    def updategradients_internal(self, u):
        # u[self.w.size:super().getgradients().size-2]=0
        super().updategradients_internal(u)
        idx = super().getgradients().size
        for q,redw in enumerate(self.redw):
            redwr = redw.ravel()
            for i in range(redwr.shape[0]):
                # if q == len(self.redw)-1:
                redwr[i] += u[idx]
                idx+=1
        for q,redb in enumerate(self.redb):
            redbr = redb.ravel()
            for i in range(redbr.shape[0]):
                # if q == len(self.redb)-1:
                redbr[i] += u[idx]
                idx+=1
            # print(redb)


    
    def to_dict(self):
        dct = super().to_dict()
        dct['redlist'] = self.redlist
        dct['nin'] = self.nin
        dctw = {}
        for i in range(len(self.redw)):
            dctw['{:05d}'.format(i)] = self.redw[i].copy()
        dct['redw'] = dctw
        dctb = {}
        for i in range(len(self.redb)):
            dctb['{:05d}'.format(i)] = self.redb[i].copy()
        dct['redb'] = dctb
        return dct
    
    def load_dict(self, dct):
        super().load_dict(dct)
        dctw = dct['redw']
        for i in range(len(self.redw)):
            self.redw[i] = dctw['{:05d}'.format(i)]
        dctb = dct['redb']
        for i in range(len(self.redb)):
            self.redb[i] = dctb['{:05d}'.format(i)]
    
    @classmethod
    def from_dict(cls, dct, gpu = True):
        n = cls(dct['redlist'], dct['d'], None, dct['nin'], dct['nout'], gpu = gpu)
        n.load_dict(dct)
        return n
    
    @classmethod
    def from_file(cls, fn, gpu = True):
        dct = msdnet.store.get_dict(fn, 'network')
        return cls.from_dict(dct, gpu = gpu)
    
    def normalizeoutput(self, datapoints):
        if self.dosegmentation:
            pass
        else:
            super().normalizeoutput(datapoints)

def get1dgridsize(sz, tpb = 1024):
    """Return CUDA grid size for 1d arrays.
    
    :param sz: input array size
    :param tpb: (optional) threads per block
    """
    return (sz + (tpb - 1)) // tpb, tpb

def get2dgridsize(sz, tpb = (8, 8)):
    """Return CUDA grid size for 2d arrays.
    
    :param sz: input array size
    :param tpb: (optional) threads per block
    """
    bpg0 = (sz[0] + (tpb[0] - 1)) // tpb[0]
    bpg1 = (sz[1] + (tpb[1] - 1)) // tpb[1]
    return (bpg0, bpg1), tpb

def leaky_relu(self, i, factor, bpg2d, tpb2d):
    leaky_relu2d_cuda[bpg2d, tpb2d](self.arr, i, factor)

##Additional Numba functions
@cuda.jit(fastmath=True)
def leaky_relu2d_cuda(out,factor):
    i = cuda.grid(1)
    if i<out.size:
        if out[i] < 0:
            out[i] *= factor


@cuda.jit(fastmath=True)
def leaky_relu2_cuda(inp, out, factor):
    i = cuda.grid(1)
    if i < inp.size:
        if inp[i] <= 0:
            out[i] *= factor

@cuda.jit(fastmath=True)
def comb_all_all_cuda(inp, out, w):
    i, j = cuda.grid(2)
    if i<out.shape[1] and j<out.shape[2]:
        for l in range(out.shape[0]):
            tmp = float32(0)
            for k in range(inp.shape[0]):
                tmp += w[k, l] * inp[k, i, j]
            out[l, i, j] += tmp

@cuda.jit(fastmath=True)
def comb_all_cuda(inp, out, w):
    i, j = cuda.grid(2)
    if i<out.shape[0] and j<out.shape[1]:
        tmp = float32(0)
        for k in range(w.shape[0]):
            tmp += w[k] * inp[k,i, j]
        out[i, j] += tmp

@cuda.jit(fastmath=True)
def comb_cuda(inp, out, w):
    i = cuda.grid(1)
    if i<out.size:
        out[i] += w*inp[i]

def fastmult_impl(a, b, out):
    tx = int32(cuda.threadIdx.x)
    gtx = tx + cuda.blockIdx.x * 1024
    gsize = 1024 * cuda.gridDim.x
    sz2 = a[0].size
    nc = a[0].shape[1]
    fshared = cuda.shared.array(shape=1024, dtype=float32)
    fidx = 0
    for ai in range(a.shape[0]):
        for bi in range(b.shape[0]):
            sumv = float32(0)
            for i in range(gtx,sz2,gsize):
                sumv += a[ai,i//nc,i%nc]*b[bi,i//nc,i%nc]
            fshared[tx] = sumv
            cuda.syncthreads()
            sz = int32(512)
            while sz>0:
                if tx<sz:
                    fshared[tx] += fshared[tx+sz]
                cuda.syncthreads()
                sz//=2
            if tx==0:
                out[cuda.blockIdx.x + fidx] = fshared[0]
            fidx += cuda.gridDim.x

def fastsumall_impl(a, out):
    tx = int32(cuda.threadIdx.x)
    gtx = tx + cuda.blockIdx.x * 1024
    gsize = 1024 * cuda.gridDim.x
    sz2 = a[0].size
    nc = a[0].shape[1]
    fshared = cuda.shared.array(shape=1024, dtype=float32)
    fidx = 0
    for ai in range(a.shape[0]):
        sumv = float32(0)
        for i in range(gtx,sz2,gsize):
            sumv += a[ai,i//nc,i%nc]
        fshared[tx] = sumv
        cuda.syncthreads()
        sz = int32(512)
        while sz>0:
            if tx<sz:
                fshared[tx] += fshared[tx+sz]
            cuda.syncthreads()
            sz//=2
        if tx==0:
            out[cuda.blockIdx.x + fidx] = fshared[0]
        fidx += cuda.gridDim.x

maxregisters = 64
fastsumall = cuda.jit(fastsumall_impl, fastmath=True, max_registers=maxregisters)
fastmult = cuda.jit(fastmult_impl, fastmath=True, max_registers=maxregisters)
while maxregisters>16:
    tmp = cuda.to_device(np.zeros((1,1,1),dtype=np.float32))
    out = cuda.to_device(np.zeros(1024,dtype=np.float32))
    try:
        fastsumall[24,1024](tmp,out)
        fastmult[24,1024](tmp,tmp,out)
    except cuda.cudadrv.driver.CudaAPIError:
        maxregisters -= 16
        fastsumall = cuda.jit(fastsumall_impl, fastmath=True, max_registers=maxregisters)
        fastmult = cuda.jit(fastmult_impl, fastmath=True, max_registers=maxregisters)
        print('Lowering maximum number of CUDA registers to ', maxregisters)
        continue
    break
