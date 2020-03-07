from __future__ import print_function

import os
import sys
from pathlib import Path
import numpy as np
import numpy.ma as ma  # masked arrays
import scipy
import matplotlib.pyplot as mpl
from pyqtgraph import configfile

import pyqtgraph as pg

import pylibrary.tools.tifffile as tiffile
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.fileselector as FS
import pylibrary.plotting.pyqtgraph_plothelpers as PGH
from ephys.ephysanalysis import metaarray
import imreg_dft as imreg
# import montage.imreg
# import mahotas as MH
from shapely.geometry import Polygon, MultiPolygon
from descartes.patch import PolygonPatch
from shapely.ops import unary_union

configfilename = 'montage.cfg'

class Montager():
    def __init__(self, celldir=None, verbose=False, use_config=True):
        self.configurationFiles = None
        self.verbose = verbose
        self.cmap = 'gist_gray_r'
        self.videos = []
        self.images = []
        self.celldir = celldir
        self.use_config = use_config
    
    def run(self):
        celldir = self.celldir

        # celldir = Path('/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het')
        if celldir is None:
            if self.use_config and Path(configfilename).is_file():
                self.configurationFiles = configfile.readConfigFile(configfilename)
                celldir = Path(self.configurationFiles['Recents'])
            elif not Path(configfilename).is_file():
                self.init_cfg()
                if self.configurationFiles is not None:
                    celldir = self.configurationFiles['Recents']
                else:
                    celldir = '.'
            else:
                exit() # print('Celldir: ', celldir)
            if celldir is None or len(str(celldir)) == 0:
                celldir = '/'
            fileselect = FS.FileSelector(dialogtype='dir', startingdir=celldir)
            if fileselect is None:
                print('No directory selected, exiting')
                exit(0)
            celldir = Path(fileselect.fileName)
            if celldir is not None and len(str(celldir)) > 0 and str(celldir) != '.':
                print('Saving celldir to config: ', celldir)
                self.configurationFiles['Recents'] = str(celldir)
                configfile.writeConfigFile(self.configurationFiles, configfilename)
        else:
            celldir = Path(celldir)
        print('celldir: ', celldir)
        self.videos = list(celldir.glob('video*.ma'))
        self.images = list(celldir.glob('image*.tif'))
        # if len(self.videos) == 0 and len(self.images) == 0:
        #     exit(0)
        self.celldir = celldir
        self.celldirname = str(celldir).replace('_', '\_')
        # else:
        #     self.videos = None # glob.glob(celldir + '/video*.ma')
        #     self.images = None # glob.glob(celldir + '/image*.tif')
        #     self.celldir = Path(celldir)
        self.list_images_and_videos()

    def setup(self, mosaics):
        for m in mosaics:
            if m['name'].startswith('video_'):
                self.videos.append(Path(self.celldir.parent, m['name']))
        self.celldirname = str(self.celldir).replace('_', '\_')
        

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
        print('initializing configuration file')
        self.configuration = {'Recents': '', }
        configfile.writeConfigFile(self.configuration, configfilename)

    def list_images_and_videos(self):
        print('videos: ', self.videos)
        print('images: ', self.images)
        print('videos:')
        for vf in self.videos:
            vp, v = os.path.split(vf)
            print('   %s' % v)
        print('images')
        for im in self.images:
            ip, i = os.path.split(im)
            print('   %s' % i)
        print()

    def process_videos(self, show=True, window='pyqtgraph', gamma=1.0, merge_gamma=1.0, sigma=2.0, 
        register=True, mosaic_data=None):
        self.loadvideos()
        self.flatten_all_videos(gamma=gamma, sigma=sigma)
        
        self.merge_flattened_videos(imagedata=self.allflat, metadata=self.videometadata, gamma=merge_gamma, 
            register=register, mosaic_data=mosaic_data)
        if show:
            if window == 'mpl':
                f, ax = mpl.subplots(1,1)
                f.suptitle(self.celldir)
                #ax = ax.ravel()
                self.show_merged_MPL(self.merged_image, ax, show=show)
            elif window == 'pyqtgraph':
                self.show_merged_pyqtgraph(self.merged_image, show=show)
            else:
                pass

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

    def process_images(self, window='pyqtgraph', show=True, gamma=2.2):
        self.load_images()
        self.show_images(window=window, show=show, gamma=gamma)

    def process_images_and_videos(self, show=True, window='pyqtgraph', gamma=1.0, merge_gamma=1.0, sigma=2.0, 
            register=True, mosaic_data=None):
        """
        I don't recommend this, although it works - makes a very large image
        """
        self.loadvideos()
        self.flatten_all_videos(gamma=gamma, sigma=sigma)
        self.load_images()
        for i, d in enumerate(self.imagedata):
            self.allflat.append(self.imagedata[i])
            self.videometadata.append(self.imagemetadata[i])
        self.merge_flattened_videos(imagedata=self.allflat, metadata=self.videometadata, gamma=merge_gamma, 
            register=register, mosaic_data=mosaic_data)

        if show:
            if window == 'mpl':
                f, ax = mpl.subplots(1,1)
                f.suptitle(self.celldir)
                #ax = ax.ravel()
                self.show_merged_MPL(self.merged_image, ax, show=show)
            else:
                self.show_merged_pyqtgraph(self.merged_image, show=show)
        

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

    def show_images(self, window='pyqtgraph', show=True, gamma=2.2):
        rows, cols = PH.getLayoutDimensions(len(list(self.images)), pref='height')
        if window == 'mpl':
            f, ax = mpl.subplots(rows, cols)
            f.suptitle(self.celldir, fontsize=9)
            if isinstance(ax, list) or isinstance(ax, np.ndarray):
                ax = ax.ravel()
            else:
                ax = [ax]
            PH.noaxes(ax)
            for i, img in enumerate(self.imagedata):
    #            print (self.imagemetadata[i])
                fna, fne = os.path.split(self.images[i])
                print('gamma: ', gamma)
                imfig = ax[i].imshow(self.gamma_correction(img, gamma=gamma))
                PH.noaxes(ax[i])
                ax[i].set_title(fne.replace('_', '\_'), fontsize=8)
                imfig.set_cmap(self.cmap)
            if show:
                mpl.show()
        elif window == 'pyqtgraph':
            self.graphs = PGH.LayoutMaker(cols=cols, rows=rows, win=self.win, labelEdges=False, ticks='talbot', items='images')
            k = 0
            for i in range(rows):
                for j in range(cols):
                    if k >= len(self.imagedata):
                        break
                    thisimg = self.gamma_correction(self.imagedata[k], 2.2)
                    print(thisimg.shape)
                    imgview = self.graphs.plots[i][j]
                                        # textlabel = pg.TextItem(f"t = {sd['elapsedtime']:.2f}", anchor=(0, 1.1))
                    imgview.ui.roiBtn.hide()
                    imgview.ui.menuBtn.hide()
                    # imgview.ui.histogram.hide()
                    imgview.setImage(thisimg)
                    k += 1

                
            
            
    def generate_colormap(self, mplname):
        from matplotlib import cm

        # Get the colormap
        colormap = cm.get_cmap(mplname)
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        self.pgMergedImage.setLookupTable(lut)

    def flatten_all_videos(self, gamma=1.0, sigma=2.0):
        self.allflat = []
        for i, vdat in enumerate(self.videodata):
            print('   Flattening video %03d' % i)
            self.allflat.append(self.flatten_video(vdat, gamma=gamma, sigma=sigma))

    def flatten_video(self, video, gamma=2.2, sigma=2.0):
        """
        Merge a video stack to a single image using max projection and filter
        """
        maxv = 0.
        stack,h,w = video.shape
        # focus = np.array([MH.sobel(self.gamma_correction(t, gamma=gamma), just_filter=True) for t in video])
        focus = self.gamma_correction(t, gamma-gamma)
        best = np.argmax(focus, 0)
        video = video.reshape((stack,-1))# image is now (stack, nr_pixels)
        video = video.transpose()  # image is now (nr_pixels, stack)
        video = video[np.arange(len(video)), best.ravel()] # Select the right pixel at each location
        video = video.reshape((h,w)) # reshape to get final result
        # video = MH.gaussian_filter(video, sigma=sigma)
        return video

    def show_merged_MPL(self, video, ax, show=True):
        imfig = ax.imshow(video)
        PH.noaxes(ax)
        imfig.set_cmap('gray_r')
        if show:
            mpl.show()

    def show_merged_pyqtgraph(self, video, show=True):
        # downsample video image to 1024x1024 at most
        np.nan_to_num(video, copy=False)
        print('video - image shape: ', video.shape)
        print(np.max(np.max(video)))
        videon = scipy.misc.imresize(video, (1024, 1024), 'lanczos')
        print(videon.shape)
        print(np.max(np.max(videon)))
        
        self.pgMergedImage = pg.image(videon)  #

    def show_unmerged(self):
        r, c = PH.getLayoutDimensions(len(self.videos), pref='height')
        f, ax = mpl.subplots(r, c)
        ax = ax.ravel()
        PH.noaxes(ax)

    def rebin(self, a, newshape):
        '''
        Rebin an array to a new shape.
        '''
        # print(a.shape)
        # print(newshape)
        assert len(a.shape) == len(newshape)
        if a.shape == newshape:
            return(a)  # skip..
        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        rb = a[tuple(indices)]
        nbins = np.prod(newshape)/np.prod(a.shape)
        return rb/nbins

    def merge_flattened_videos(self, imagedata, metadata, gamma=1.0, register=False, mosaic_data=None):
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
        # determine order of merge. Brightest images are done first so they
        # are underneath all others.
        imax = np.zeros(len(imagedata))
        for i, vdata in enumerate(imagedata):
            imax[i] = np.max(np.max(vdata))
        jmax = np.argsort(imax)
        # print('jmax: ', jmax)
        # find extent for all images
        polymap = MultiPolygon([])
        vs = []
        for i, v in enumerate(metadata):
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
            mapext = ((extentx[0], extenty[0]), (extentx[0], extenty[1]), (extentx[1], extenty[1]),
                        (extentx[1], extenty[0]), (extentx[0], extenty[0]))
            data_poly = Polygon(mapext)
            polymap = polymap.union(data_poly)
            if polymap.type == 'Polygon': # that sometimes converts to a basic Polygon
                polymap = MultiPolygon([polymap])

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
            vs.append(vt['scale'][0])
            vscaley = np.min((np.fabs(vt['scale'][1]), vscaley))
            if vt['scale'][1] < 0.:
                yflip = -1
        grid_x = int(1.5*(maxx-minx)/vscalex)
        grid_y = int(1.5*(maxy-miny)/vscaley)
        if self.verbose:
            print('minx: %15g  maxx: %15g miny: %g  maxy: %g' % (minx*1000, maxx*1000, miny, maxy))
            print('spanx: %f  spany: %f' % (1000*(maxx-minx), 1000.*(maxy-miny)))
            print('scalex, scaley: ', vscalex, vscaley*yflip)
            print('grid_x, y: ', grid_x, grid_y)
        # print('vscalex, y: ', vscalex, vscaley)
        # print('Bounds: ', polymap.bounds)
        # b = [x for x in polymap.bounds]
        # b[0] *= 0.9
        # b[2] *= 1.1
        # b[1] *= 0.9
        # b[3] *= 1.1
        # ib = b.copy()
        # ib[0] = int(b[0]/vscalex)
        # ib[2] = int(b[2]/vscalex)
        # ib[1] = int(b[1]/vscaley)
        # ib[3] = int(b[3]/vscaley)
        # print('b: ', b)
        # print('ib: ', ib)
        # print('ibx: ', ib[2]-ib[0])
        # print('ibx: ', ib[3]-ib[1])
        # print(grid_x, grid_y)
        # print(vs)
        # exit()
        self.merged_image = np.zeros((grid_x, grid_y))  # create space to hold image
        self.n_images = np.zeros((grid_x, grid_y))
        bkgd = 0.
        
        bigpoly = MultiPolygon([])
        bigpoly2 = MultiPolygon([])
        for i, vdata in enumerate(imagedata):  # place the images onto the big map
            vdata = vdata/exposures[i]
            v = metadata[i]
            vt = v['transform']
            vp = vt['pos']
            vr = v['region']
            vb = v['binning']

            ix = int((vp[0]-minx)/vscalex)
            iy = int((vp[1]-miny)/vscaley)
            if self.verbose:
                print('vb: ', vb)
                print('vr: ', vr)
                print('Mapping Video image i: %d' % i, vdata.shape)
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
                    mode='constant', constant_values=bkgd)
                self.n_images = np.pad(self.n_images,
                    pad_width=((0, delx),(0, dely+1)),
                    mode='constant', constant_values=bkgd)

            yseq = range(iyw, iy, -1)

            # all shapes are rectangles.
            # So we make a polygon that represents the overlap between two subsequent rectangles.
            
            data_poly = Polygon([(ix, iy), (ix, iyw), (ixw, iyw), (ixw, iy), (ix, iy)])

            # if not bigpoly2.is_empty:
            #     c = ['g', 'b', 'r', 'y', 'c', 'm']
            #     fig = mpl.figure(1, figsize=(8,6), dpi=90)
            #     # 1: valid polygon
            #     ax1 = fig.add_subplot(121)
            #     for ie, p in enumerate(bigpoly2):
            #         ax1.plot(p.exterior.xy[0], p.exterior.xy[1], c[ie])
            #     patch1 = PolygonPatch(bigpoly2, facecolor='k', edgecolor='r', alpha=0.3, zorder=2)
            #     ax1.add_patch(patch1)
            #     patch2 = PolygonPatch(data_poly, facecolor='b', edgecolor='g', alpha=0.2, zorder=2)
            #     ax1.add_patch(patch2)
            #     patch3 = PolygonPatch(bigpoly2.intersection(data_poly), facecolor='y', edgecolor='k', linewidth=2, alpha=0.2, zorder=2)
            #     ax1.add_patch(patch3)
            #     bigpoly2 = bigpoly2.union(data_poly)
            #     if bigpoly2.type == 'Polygon': # that sometimes converts to a basic Polygon
            #         bigpoly2 = MultiPolygon([bigpoly2])
            #     mpl.show()
            #     continue
            if bigpoly.intersects(data_poly):
                overlap = bigpoly.intersection(data_poly)  # get intersection region
                b = [int(a) for a in overlap.bounds]
                target_shape = self.n_images[b[0]:b[2], b[1]:b[3]].shape
                thr = 0.1 * np.prod(target_shape)  # minimal overlap region 
                off = {'angle': 0., 'tvec': [0, 0]}
                if mosaic_data is None and np.sum(self.n_images[b[0]:b[2], b[1]:b[3]] > 0) > thr:   # exceed threshold of overlap, try alignment against existing
                    # print('threshold: ', np.sum(self.n_images[b[0]:b[2], b[1]:b[3]] > 0), thr, target_shape)
                    i_m = self.merged_image[b[0]:b[2], b[1]:b[3]]  # get intersection in image
                    x0 = b[0] - ix
                    x1 = b[2] - ix
                    y0 = b[1] - iy
                    y1 = b[3] - iy
                    # print('overlap translated: ', x0, x1, y0, y1)
                    v_m = np.fliplr(vdata[x0:x1, y0:y1]) # get intersecting  data
                    v_m = vdata[x0:x1, y0:y1] # get intersecting  data
                    # fig = mpl.figure(1, figsize=(8,6), dpi=90)
                    # # 1: valid polygon
                    # ax1 = fig.add_subplot(121)
                    # ax1.imshow(i_m.T, origin='upper')
                    # ax1.set_title('Main Image (intersect)')
                    # ax2 = fig.add_subplot(122)
                    # ax2.imshow(v_m.T, origin='upper')
                    # ax2.set_title('new data')
                    # mpl.show()
                    if register:
                        try:
                            off = imreg.translation(i_m, v_m)
                        except:
                            print('failed translation')
                            print(np.any(np.isnan(i_m)))
                            print(np.any(np.isnan(v_m)))
                            print(np.amax(i_m), np.amax(v_m))
                            print(np.amin(i_m), np.amin(v_m))
                            raise ValueError()
                if mosaic_data is not None:
                    # print(mosaic_data[i])
                    pos = mosaic_data[i]['userTransform']['pos']
                    pos[0] = int(pos[0]/vscalex)  # pixels shift
                    pos[1] = int(pos[1]/vscaley)
                    off['tvec'] = pos

                print('Registration Offsets: ', off)

                if off['angle'] == 0 and abs(off['tvec'][0]) < 200 and abs(off['tvec'][1]) < 200:  # rotation and big translation is not allowed
                    offv = off['tvec']
                    ix = int(ix-offv[1])
                    ixw = int(ixw-offv[1])
                    y0 = int(iyw-offv[0])
                    y1 = int(iy-offv[0])
                    # timg = imreg.transform_img(vdata, angle=off['angle'], tvec=off['tvec'])
                    if ix < 0:
                        vdata = vdata[-ix+1:, :]  # clip vdata on the left side
                        ix = 0  
                    self.merged_image[ix:ixw, y0:y1:-1] += vdata
                    # timg = imreg.transform_img(vdata, angle=off['angle'], tvec=off['tvec'])
                    # imreg.imshow(self.merged_image[ix:ixw, y0:y1:-1], vdata, timg, cmap=None, fig=None)
                    # mpl.show()
                    print('aligned')
                else:
                    self.merged_image[ix:ixw, yseq] += vdata
                # else:
                #     self.merged_image[ix:ixw, yseq] += vdata
                    
            else:
                # fig = mpl.figure(1, figsize=(8,6), dpi=90)
                # # 1: valid polygon
                # ax1 = fig.add_subplot(121)
                # ax1.imshow(self.merged_image[ix:ixw, yseq], origin='upper')
                # ax1.set_title('Main Image2')
                # ax2 = fig.add_subplot(122)
                # ax2.imshow(vdata, origin='upper')
                # ax2.set_title('new data2')
                # mpl.show()
                self.merged_image[ix:ixw, yseq] += vdata
                print('added')

            self.n_images[ix:ixw, yseq] += 1.0  # keep track of how many planes were added here
            bigpoly = bigpoly.union(data_poly)
            if bigpoly.type == 'Polygon': # that sometimes converts to a basic Polygon
                bigpoly = MultiPolygon([bigpoly])
            # print('bigpoly: ', bigpoly)
        self.bigpoly = bigpoly
        self.image_boundary = [self.bigpoly.bounds[0]*vscalex, self.bigpoly.bounds[1]*vscaley,
                self.bigpoly.bounds[2]*vscalex, self.bigpoly.bounds[3]*vscaley]
        
        self.n_images[self.n_images == 0.] = np.nan  # trick for avoiding divide by 0
        self.merged_image = self.merged_image/self.n_images
        minintensity = np.nanmin(self.merged_image)
        self.merged_image = np.nan_to_num(self.merged_image, minintensity)
        # self.merged_image -= minintensity + 1e-6
        self.merged_image = self.gamma_correction(self.merged_image, gamma=1.5)
        print('image minm: ', np.min(self.merged_image), '  max: ', np.max(self.merged_image))
        
        # self.merged_image = np.fliplr(self.gamma_correction(self.merged_image, gamma=gamma))
        #self.merged_image = self.threshold_image(self.merged_image, 2000.)
       # self.merged_image = self.rolling_ball(self.merged_image)

    def gamma_correction(self, image, gamma=2.2, imagescale=np.power(2, 16)):
        if gamma == 0.0:
            return image
        imagescale= float(imagescale)
        # print('iscale, gamma: ', imagescale, gamma)
        try:
            imr = image/imagescale
            corrected = (np.power(imr, (1. / gamma)))*imagescale
        except:
            print('image minm: ', np.min(image), '  max: ', np.max(image))
            print('any neg or zero? ', np.any(image <= 0.))
            raise ValueError
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

def doit():
    # sel = FS.FileSelector(dialogtype='dir')
    # print('main sel filename: ', sel.fileName)
    # if sel.fileName is not None:
    M = Montager(verbose=False) # , celldir=Path(sel.fileName))
    M.run()
    # M.app = pg.mkQApp()
    # M.win = pg.QtGui.QWidget()
    M.list_images_and_videos()
    # M.process_images_and_videos(window='pyqtgraph', show=True, gamma=2.2, merge_gamma=2., sigma=3.0)
    M.process_videos(window='mpl', show=True, gamma=1.5, merge_gamma=-1., sigma=2.5)
    # M.process_images(window='mpl', show=True, gamma=1.5)
    # print('M: ', M)
    # if sys.flags.interactive == 0:
#         pg.QtGui.QApplication.exec_()

    
if __name__ == '__main__':
    M = doit()
