from __future__ import print_function

import os
import sys
import glob
import numpy as np
import numpy.ma as ma  # masked arrays
#import pylibrary.tifffile as tifffile  # for tiff file read
import tifffile
import matplotlib.pyplot as mpl

import pylibrary.PlotHelpers as PH
import pylibrary.fileselector as FS
import ephysanalysis as ep
import pyqtgraph as pg
from pyqtgraph import metaarray
from pyqtgraph import configfile
import montage.imreg
import mahotas as MH

configfilename = 'montage.cfg'

class Montager():
    def __init__(self, celldir='', verbose=False):
        self.verbose = verbose
        self.cmap = 'gist_gray_r'
        if celldir is not '':
            if os.path.isfile(configfilename):
                try:
                    self.configuration = configfile.readConfigFile(configfilename)
                except:
                    self.init_cfg()
            else: # build base configuration file
                self.init_cfg()
            print (celldir)
            print(self.configuration)
            if celldir == '':  # no cell dir, us gui
                celldir = self.configuration['Recents']
                fileselect = FS.FileSelector(dialogtype='dir', startingdir=celldir)
                celldir = fileselect.fileName
                if celldir is None:
                    exit(0)
            self.configuration['Recents'] = celldir
            configfile.writeConfigFile(self.configuration, configfilename)
            self.videos = glob.glob(celldir + '/video*.ma')
            self.images = glob.glob(celldir + '/image*.tif')
            if len(self.videos) == 0 and len(self.images) == 0:
                exit(0)
            self.celldir = celldir.replace('_', '\_')
        else:
            self.videos = None # glob.glob(celldir + '/video*.ma')
            self.images = None # glob.glob(celldir + '/image*.tif')
            self.celldir = ''
            
    def get_images(self, celldir):
        self.celldir = celldir
        self.images = glob.glob(celldir + '/image*.tif')

    def get_image(self, imagename):
        self.celldir = ''
        self.images = [imagename]


    def set_image(self, celldir, imagename):
        self.celldir = celldir
        self.images = glob.glob(os.path.join(celldir, imagename))
    
    def init_cfg(self):
        self.configuration = {'Recents': '', }
        configfile.writeConfigFile(self.configuration, configfilename)
        
    def list_images(self):
        print(self.videos)
        print(self.images)
        print('videos:')
        for vf in self.videos:
            vp, v = os.path.split(vf)
            print('   %s' % v)
        print('images')
        for im in self.images:
            ip, i = os.path.split(im)
            print('   %s' % i)
        print()

    def process_videos(self, show=True, window='pyqtgraph'):
        self.loadvideos()
        self.flatten_all_videos()
        self.merge_flattened_videos()

        if window == 'mpl':
            f, ax = mpl.subplots(1,1)
            f.suptitle(self.celldir)
            #ax = ax.ravel()
            self.show_merged_MPL(self.merged_image, ax, show=show)
        else:
            self.show_merged_pyqtgraph(self.merged_image, show=show)
        
    def loadvideos(self):
        """
        loads all videos in the dir into a list
        and in the same order, grabs the metadata
        """
        self.videodata = []
        self.videometadata = []
        print('Loading video files:')
        for j, im in enumerate(sorted(self.videos)):
            print ('   %s' % im)
            imdat = metaarray.MetaArray(file=im, readAllData=True)
            info = imdat[0].infoCopy()
            self.videodata.append(imdat.asarray())
            imd, imf = os.path.split(im)
            index = self.getIndex(currdir=imd)
            try:
                if imf in self._index.keys():
                    self.videometadata.append(self._index[imf])
            except:
                raise Exception('No matching .index entry for video file %s' % imf)

    def process_images(self, show=True):
        self.load_images()
        self.show_images(show=show)
    
    def load_images(self):
        self.imagedata = []
        self.imagemetadata = []
        print('Loading image files: ')
        for j, im in enumerate(sorted(self.images)):
            print ('   %s' % im)
            with tifffile.TiffFile(im) as tif:
                images = tif.asarray()
            self.imagedata.append(images)
            imd, imf = os.path.split(im)
            index = self.getIndex(currdir=imd)
            try:
                if imf in self._index.keys():
                    self.imagemetadata.append(self._index[imf])
                    
            except:
                raise Exception('No matching .index entry for image file %s' % imf)

    def show_images(self, show=True):
        r, c = PH.getLayoutDimensions(len(self.images), pref='height')
        f, ax = mpl.subplots(r, c)
        f.suptitle(self.celldir, fontsize=9)
        if isinstance(ax, list) or isinstance(ax, np.ndarray):
            ax = ax.ravel()
        else:
            ax = [ax]
        PH.noaxes(ax)
        for i, img in enumerate(self.imagedata):
#            print (self.imagemetadata[i])
            fna, fne = os.path.split(self.images[i])
            imfig = ax[i].imshow(self.gamma_correction(img, 2.2))
            PH.noaxes(ax[i])
            ax[i].set_title(fne.replace('_', '\_'), fontsize=8)
            imfig.set_cmap(self.cmap)
        if show:
            mpl.show()        
    
    def generate_colormap(self, mplname):
        from matplotlib import cm

        # Get the colormap
        colormap = cm.get_cmap(mplname) 
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        self.pgMergedImage.setLookupTable(lut)

    def flatten_all_videos(self):
        self.allflat = []
        for i, vdat in enumerate(self.videodata):
            print('   Flattening video %03d' % i)
            self.allflat.append(self.flatten_video(vdat))

    def flatten_video(self, video):
        """
        Merge a video stack to a single image using max projection and filter
        """
        maxv = 0.
        stack,h,w = video.shape
        focus = np.array([MH.sobel(t, just_filter=True) for t in video])
        best = np.argmax(focus, 0)
        video = video.reshape((stack,-1)) # image is now (stack, nr_pixels)
        video = video.transpose() # image is now (nr_pixels, stack)
        video = video[np.arange(len(video)), best.ravel()] # Select the right pixel at each location
        video = video.reshape((h,w)) # reshape to get final result
        return video

    def show_merged_MPL(self, video, ax, show=True):
        imfig = ax.imshow(video)
        PH.noaxes(ax)
        imfig.set_cmap('viridis')
        if show:
            mpl.show()
    
    def show_merged_pyqtgraph(self, video, show=True):
        self.pgMergedImage = pg.image(video)  # 

    def show_unmerged(self):
        r, c = PH.getLayoutDimensions(len(self.videos), pref='height')
        f, ax = mpl.subplots(r, c)
        ax = ax.ravel()
        PH.noaxes(ax)
        
    def rebin(self, a, newshape ):
        '''
        Rebin an array to a new shape.
        '''
        assert len(a.shape) == len(newshape)
        if a.shape == newshape:
            return(a)  # skip..
        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        rb = a[tuple(indices)]
        nbins = np.prod(newshape)/np.prod(a.shape)
        return rb/nbins
        
    def merge_flattened_videos(self):
        minx = 100000.  # start with really big values
        maxx = -100000.
        miny = 100000.
        maxy = -100000.
        print ('Combining videos')

        exposures = []
        scalex = []  # for each image keep track of the scale
        scaley = []
        pos = []
        scale = []
        region = []
        binning = []
        vscalex = 1.
        vscaley = 1.
        yflip = 1.
        # find extent for all images
        for i, v in enumerate(self.videometadata):
            vt = v['transform']
            vr = v['region']
            vb = v['binning']
            exposures.append(np.array(v['exposure']))
            extentx = [vt['pos'][0] + float(vr[0])*vt['scale'][0],
                       vt['pos'][0] + float(vr[2])*vt['scale'][0]
                      ]
            extenty = [
                      vt['pos'][1] + float(vr[1])*vt['scale'][1],
                      vt['pos'][1] + float(vr[3])*vt['scale'][1]
                    ]
            if self.verbose:
                print('extents: ', extentx, extenty)
                print('region: ', vr)
                print('scale: ', vt['scale'])
                print('binning: ', vb)
            minx = np.min((minx, extentx[0], extentx[1]))
            maxx = np.max((maxx, extentx[0], extentx[1]))

            miny = np.min((miny, extenty[0], extenty[1]))
            maxy = np.max((maxy, extenty[0], extenty[1]))

            vscalex = np.min((np.fabs(vt['scale'][0]), vscalex))
            vscaley = np.min((np.fabs(vt['scale'][1]), vscaley))
            if vt['scale'][1] < 0.:
                yflip = -1
        grid_x = int((maxx-minx)/vscalex) # add 1 for boundary or int flooring
        grid_y = int((maxy-miny)/vscaley)
        if self.verbose:
            print('minx: %15g  maxx: %15g miny: %g  maxy: %g' % (minx*1000, maxx*1000, miny, maxy))
            print('spanx: %f  spany: %f' % (1000*(maxx-minx), 1000.*(maxy-miny)))
            print('scalex, scaley: ', vscalex, vscaley*yflip)
            print('grid_x, y: ', grid_x, grid_y)
        self.merged_image = np.zeros((grid_x, grid_y))  # create space to hold image
        self.n_images = np.zeros((grid_x, grid_y))
        for i, vdata in enumerate(self.allflat):  # place the images onto the big map
            vdata = vdata/exposures[i]
            v = self.videometadata[i]
            vt = v['transform']
            vp = vt['pos']
            vr = v['region']
            vb = v['binning']
            
            ix = int((vp[0]-minx)/vscalex)
            iy = int((vp[1]-miny)/vscaley)
            if self.verbose:
                print('vb: ', vb)
                print('vr: ', vr)
                print('Mapping Vvideo image i: %d' % i, vdata.shape)
                print('to new shape: ', vdata.shape[0]*vb[0], vdata.shape[1]*vb[1])
            vdata = self.rebin(vdata, (vdata.shape[0]*vb[0], vdata.shape[1]*vb[1]))
            # check bounds for mapping
            ixw = ix+vdata.shape[0]
            iyw = iy+vdata.shape[1]


            mis = self.merged_image.shape
            delx = 0
            dely = 0
            if ixw > mis[0]:
                delx = ixw - mis[0]
            if iyw > mis[1]:
                dely = iyw - mis[1]
            if delx > 0 or dely > 0:
                print('    **** Padding *****')
                self.merged_image = np.pad(self.merged_image,
                    pad_width=((0, delx),(0, dely+1)),
                    mode='constant', constant_values=0.)
                self.n_images = np.pad(self.n_images,
                    pad_width=((0, delx),(0, dely+1)),
                    mode='constant', constant_values=0.)
            
            yseq = range(iyw, iy, -1)
            print('max yseq: ', max(yseq))
            # find if region already has an image here, and do a registration using the area that overlaps
            # print('i: ', i)
            # refimg = self.n_images[ix:ixw, yseq]  # get the ref image (and flip)
            # if np.sum(refimg) > 100.: # require 100 points overlap
            #     # print(np.sum(refimg))# make intersecting mask
            #    #  print(refimg.shape, len(range(ix,ixw))*len(yseq), np.prod(refimg.shape))
            #    #  maskb = np.ma.masked_where(refimg > 0., refimg)  # check
            #    #  maskn = np.ma.masked_where(np.ma.getmask(maskb), vdata)
            #    #  maskn = np.where(refimg > 0, vdata)
            #    #  print('maskb shape: ', np.array(maskb).shape, np.ma.count(maskb))
            #    #  print('maskn shape: ', maskn.shape, np.ma.count(maskn))
            #     refimg[refimg>0] = 1
            #     di = np.argwhere(refimg*vdata > 0)
            #     print ('di shape: ', di.shape)
            #     print('di: ', di)
            # #    vdp = vdata*refimg
            #  #   vdp[vdp == 0.0] = np.nan
            # #    refimg[refimg == 0.0] = np.nan
            #     fi = 0
            #     li = refimg.shape[0]
            #     for i in range(refimg.shape[0]):
            #         v = np.argwhere(refimg[i,:] > 0)
            #         print (len(v))
            #         if len(v) > 0:
            #             fi = min(fi, v[0])
            #             li = max(li, v[-1])
            #     fj = 0
            #     lj = refimg.shape[1]
            #     for j in range(refimg.shape[1]):
            #         v = np.argwhere(refimg[:,j] > 0)
            #         if len(v) > 0:
            #             fj = min(fj, v[0])
            #             lj = max(lj, v[-1])
            #
            #     print('fi li fj lj', fi, li, fj, lj)
            #     # mxl = np.argwhere(vdp > 0.)[0]
            #     # vdp = vdp[mxl]
            #     # print(mxl)
            #     # print( vdp.shape, vdata.shape)
            #     # exit(1)
            #     off = imreg.translation(vdata[fi:li, fj:lj], refimg[fi:li, fj:lj])
            #     print ('Registration Offsets: ', off)
            #     offt = np.array(off).T
            #
            #     #exit(1)
            
            self.n_images[ix:ixw, yseq] += 1.0
            
            self.merged_image[ix:ixw, yseq] += vdata
        self.n_images[self.n_images == 0.] = np.nan  # trick for avoiding divide by 0
        self.merged_image = self.merged_image/self.n_images
        # minintensity = np.nanmin(np.nanmin(self.merged_image[self.merged_image > 0]))
        # self.merged_image -= minintensity
        #self.merged_image = np.nan_to_num(self.merged_image-minintensity)
        self.merged_image = self.gamma_correction(self.merged_image, 2.2)
        #self.merged_image = self.threshold_image(self.merged_image, 2000.)
       # self.merged_image = self.rolling_ball(self.merged_image)
    
    def gamma_correction(self, image, gamma=2.2, imagescale=2^16):
        if gamma == 0.0:
            return image
        imagescale= float(imagescale)
        corrected = (np.power((image/imagescale), (1. / gamma)))*imagescale
        return corrected
    
    def threshold_image(self, image, thr):
        image[image<thr] = 0.
        image[image>=thr] = 1.
        return image

    def rolling_ball(self, image, imgdepth=255):
        import scipy.ndimage as scim
        from skimage.morphology import ball
        # Read image
        # Create 3D ball structure
        s = ball(50)
        # Take only the upper half of the ball
        h = int((s.shape[1] + 1) / 2)
        # Flat the 3D ball to a weighted 2D disc
        s = s[:h, :, :].sum(axis=0)
        # Rescale weights into 0-255
        s = (imgdepth * (s - s.min())) / (s.max()- s.min())
        # Use im-opening(im,ball) (i.e. white tophat transform) (see original publication)
        im_corr = scim.white_tophat(image, structure=s)
        return im_corr

    def getIndex(self, currdir=''):
        self._readIndex(currdir=currdir)
        if self._index is not None:
            return self._index['.']
        else:
            return None

    def _readIndex(self, currdir=''):
        self._index = None
        indexFile = os.path.join(currdir, '.index')
#        print self.protocol, currdir, indexFile
        if not os.path.isfile(indexFile):
           # print("Directory '%s' is not managed!" % (self.dataname))
            return self._index
        self._index = configfile.readConfigFile(indexFile)
        return self._index
        

if __name__ == '__main__':
#    sel = FS.FileSelector(dialogtype='dir')
#    if sel.fileName is not None:
    M = Montager(verbose=False)
    M.list_images()
    # M.process_videos(window='pyqtgraph', show=True)
    # if sys.flags.interactive == 0:
    #     pg.QtGui.QApplication.exec_()
    M.process_images(show=True)
     