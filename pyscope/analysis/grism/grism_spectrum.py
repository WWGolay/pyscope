
##imports and definitions
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io.fits import getdata
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter1d
from scipy.signal import medfilt, find_peaks, detrend
from datetime import datetime
import os,sys
import csv
import click
import validators

balmer = np.array([397.0, 410.17, 434.05, 486.14, 656.45])
deg = np.pi/180.



##create grism_utils class
class StopExecution(Exception):
    def _render_traceback_(self):
        pass

class grism_utils:
    def __init__(self, grism_image, cal_file, rot_angle, box, f_wave,f_gain):
        ''' Utilities for calibrating and plotting spectra from grism images'''
        
        self.grism_image = grism_image
        self.cal_file = cal_file
        self.balmer = np.array([397.0, 410.17, 434.05, 486.14, 656.45])
        self.wavelength_range = [380,750]
        
        ''' Open image, extract header information '''
        im, hdr = getdata(grism_image, 0, header=True)
        self.im = im
        self.hdr = hdr
        self.object_name  = hdr['OBJECT']
        self.obs_date     = hdr['DATE-OBS'][:-12].replace('T',' ')
        self.telescope    = hdr['TELESCOP']
        self.camera       = hdr['CAMSN'][0:6]
        self.imsize_x = hdr['NAXIS1'] ; self.imsize_y = hdr['NAXIS2']
        # Create default plot title
        self.title = '%s (%s)\nGrism spectrum: %s %s' % \
        (self.object_name, self.obs_date, self.telescope, self.camera)
                                                     
        self.rot_angle = rot_angle
        self.box = box
        self.f_wave = f_wave
        self.f_gain = f_gain
        
    def summary_info(self):
        return self.object_name, self.obs_date,self.telescope,self.camera,self.title,self.im,self.rot_angle, self.box, self.f_wave, self.f_gain
    
    def create_subimage(self):       
        ''' Create a strip subimage using parameters from calibration file, or user-supplied'''
        
        # Rotate full image                 
        im_rot = np.fliplr(self.im)
        
        # Flip image so L-> R corresponds to short -> long wavelength 
        im_flip = rotate(im_rot, self.rot_angle,reshape=False)
        
        # Trim image to subimage using box
        xmin,xmax,ymin,ymax = self.box
        self.subim = im_flip[ymin:ymax,xmin:xmax]
        return self.subim
    
    def plot_image(self,image = None, vmin=None, vmax=None, title='',figsize =(8,4),cmap='gray'):
        '''Plot image: defaults to full image '''
        fig, ax = plt.subplots(figsize=figsize)
        zmean = np.median(image); s = np.std(image)
        if vmin == None:
            vmin = zmean - 2*s; 
        if vmax == None:
            vmax = zmean + 12*s
        myplot = ax.imshow(image,cmap=cmap, vmin= vmin, vmax = vmax)
        plt.title(title)
        return fig 
    
    def calibrate_spectrum(self,subim,norm):
        '''Calculates raw spectrum by summing pixels in all vertical slices, then apply wavelength, gain calibration'''

        pixels = np.arange(subim.shape[1])
        # find sum of pixel values along each column (spectral channel)
        uncal_amp = []
        for pixel in pixels:
            ymax,signal,signal_max = self.__calc_channel_signal(subim,pixel)
            uncal_amp.append(signal)
        uncal_amp = np.array(uncal_amp)
        if norm==True:
            uncal_amp /= np.max(uncal_amp)
        else:
            pass
     
        # Wavelength calibration
        wave = self.f_wave(pixels)
    
        # Gain calibration
        gain_curve = self.f_gain(wave)
        cal_amp = uncal_amp / gain_curve
    
        spectrum = np.vstack([pixels,wave,uncal_amp,cal_amp])
        return spectrum

    def __calc_channel_signal(self,subim,xpixel):  
        ''' Calculates total counts in specified spectral channel xpixel 
        by subtracting background and summing. The spectral signal is assumed 
        to be in middle half of the spectrum. '''
        
        yvals = subim[:,xpixel]
        yindex = np.arange(len(yvals))
        
        # Choose first, last quartiles for base, fit linear slope
        n1 = int(len(yindex)/4); n2 = 3*n1
        x1 = yindex[0:n1] ; x2 = yindex[n2:]
        y1 = yvals[0:n1]  ; y2 = yvals[n2:]
        X = np.concatenate((x1,x2),axis=0)
        Y = np.concatenate((y1,y2),axis=0)
        c = np.polyfit(X,Y,1) # linear fit  
        p = np.poly1d(c)
        base = p(yindex)
        # Calculate signal vs pixel by subtracting baseline, sum and get index of maximum pixel
        signal = yvals - base
        signal_max = np.max(signal)
        ymax = np.argmax(signal)       
        return(ymax, np.sum(signal),signal_max) 
    
    def plot_gain_curve(self, spectrum,wmin=400,wmax=750,color='r',title=''):
        ''' Plot the gain vs wavelength using the f_gain function '''
        p,w,u,c = self.clip_spectrum(spectrum,wmin,wmax)
        gain_curve =self.f_gain(w)
        gain_curve /= np.max(gain_curve)
        fig, ax = plt.subplots(1, 1,figsize=(8,5))
        ax.plot(w,gain_curve,color =color)
        plt.suptitle(title)
        plt.grid()
        return fig
    
    # Normal 2x2 grism plot
    def plot_2x2(self,object_spectrum, reference_spectrum,title='', medavg=1):
        ''' Creates 2x2 plot: 
        UL: uncalibrated object spectrum
        UR: Gain curve
        LL: Calibrated object spectrum
        LR: Reference spectrum 
        Note: object spectrum has 4 rows (pixels, wavelength, uncal. amp., calib. amplitude), 
        reference spectrum has 2 rows (wavelength, calib. amplitude)'''
    
        _,wave, amp_uncal, amp_cal = object_spectrum
        wave_ref,amp_ref = reference_spectrum
    
        # Median average if requested
        amp_uncal = medfilt(amp_uncal,kernel_size = medavg) 
        amp_cal = medfilt(amp_cal,kernel_size = medavg) 
    
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2,figsize=(10,8))
        plt.suptitle(title)
        # Uncalibrated spectrum
        ax1.plot(wave, amp_uncal)
        ax1.set_xlabel('Wavelength')
        ax1.set_ylabel('Uncalibrated amp.')
        ax1.set_title('Raw Spectrum')
        ax1.grid()

        # Gain curve
        gain =  self.f_gain(wave_ref)
        gain /= np.max(gain)
        ax2.plot(wave_ref,gain,'k.',ms=2)
        ax2.set_xlabel('Wavelength')
        ax2.set_ylabel('Gain')
        ax2.set_xlim(380,750)
        ax2.set_title('Gain vs. Wavelength')
        ax2.grid()

        # Calibrated amplitude
        ax3.plot(wave, amp_cal)
        ax3.set_xlabel('Wavelength')
        ax3.set_ylabel('Calibrated amplitude')
        ax3.set_xlim(380,750)
        ax3.set_title('Calibrated Spectrum')
        ax3.grid()

        # Reference spectrum
        ax4.plot(wave_ref, amp_ref)
        ax4.set_xlabel('Wavelength')
        ax4.set_ylabel('Calibrated amplitude')
        ax4.set_title('Jacoby Spectrum')
        ax4.grid()
        plt.tight_layout()
        return fig

    def fit_gaussian(self,spectrum,wave_min,wave_max):
        '''Fit a Gaussian function to spectral line in the wavelength range wave_min to wave_max.
        Returns fitted parameters with uncertainties and a plot of spectral line and fitted Gaussian'''
    
        # Constrain spectrum to desired wavelength range, unpack needed arrays
        _,wave,_,amp = self.clip_spectrum(spectrum,wave_min,wave_max)
    
        # Detrend to remove slope
        amp = detrend(amp)
    
        # Find index of max value
        imax = np.argmax(np.abs(amp))
    
        # Fit a Gaussian model
        p0 = (wave[imax],amp[imax],2,0,0)
        params, pcov = curve_fit(self.__gftn, wave, amp,p0)
        wave_ctr, a, fwhm,m,b = params
        amp_mod = self.__gftn(wave, *params)
    
        # Extract uncertainties from covariance
        wave_ctr_err, a_err, fwhm_err, _ ,_ = np.abs(np.sqrt(np.diag(pcov)))
        f= 2*np.sqrt(2*np.log(2)) *0.5 # sigma -> FWHM factor 
        fwhm *= f; fwhm_err *= f
        fwhm = np.abs(fwhm)
        params = (wave_ctr,wave_ctr_err,fwhm,fwhm_err,a,a_err)
        return params, wave, amp, amp_mod

    def plot_spectral_line(self,wave,amp,amp_mod,wave_ctr,wave_ctr_err,fwhm,fwhm_err,color='r',title=''):
        fig, ax = plt.subplots(1, 1,figsize=(8,5))
        ax.plot(wave,amp_mod,color =color)
        ax.plot(wave,amp,'k.')
        T = '%s\n$\lambda_c$ = %.1f +/- %.1f nm, FWHM = %.1f +/- %.1f nm' % \
            (title,wave_ctr,wave_ctr_err,fwhm,fwhm_err)
        plt.suptitle(T)
        plt.grid()
        return fig
    
    def __gftn(self,x, x0, a, sigma, m, b):
        ''' Define a Gaussian function with a sloping pedestal '''
        t = (x-x0) /sigma
        f = a*np.exp(-t**2) + m*x + b
        return f

    def clip_spectrum(self,spectrum, wave_min, wave_max):
        '''Restrict wavelength range to avoid crazy gain curve at edges '''
        pixel, wave, uncal_amp, cal_amp = spectrum
        mask = (wave > wave_min) & (wave < wave_max)
        pixel_m = pixel[mask]
        wave_m  = wave[mask]
        uncal_amp_m   = uncal_amp[mask]
        cal_amp_m = cal_amp[mask]
        spectrum = np.vstack( [pixel_m, wave_m,uncal_amp_m, cal_amp_m] )
        return spectrum
    
    def plot_spectrum(self, spectrum, xaxis = 'pixel',yaxis = 'uncal', subrange = slice(0,9999), title='', \
        jacoby = False, plot_balmer = False, medavg = 1,xlims =[0,0],ylims =[0,0]):
        '''Plots science or Jacoby  spectrum, with selectable axes [x: pixels or wavelength; y: uncalibrated or calibrated. '''
        fig, ax = plt.subplots(1,1,figsize=(8, 4))           
        fig.suptitle(title) 
    
        if jacoby:
            w,c = spectrum
            s = subrange
            wave = w[s]; amp_cal = c[s]
            x = wave    ; ax.set_xlabel('Wavelength [nm]') 
            y = amp_cal ; ax.set_ylabel('Calibrated amplitude')
        else:
            # Slice spectrum by selecting user-selected pixel range (Unpythonic!)
            p,w,u,c = spectrum
            s = subrange
            p = p[s] ; w = w[s]; u =u[s]; c = c[s]
            spectrum = np.vstack([p,w,u,c])
        
            pixels, wave, amp_uncal, amp_cal = spectrum
            if xaxis == 'pixel':
                x = pixels ; ax.set_xlabel('Pixels')
            else:
                x = wave ; ax.set_xlabel('Wavelength [nm]')
            if yaxis == 'uncal':
                y = amp_uncal ; ax.set_ylabel('Uncalibrated amplitude')
            else:
                y = amp_cal   ; ax.set_ylabel('Calibrated amplitude')
  
        if plot_balmer:
            for b in balmer: ax.axvline(x=b,linestyle='dotted')
     
        # Median average if requested 
        y = medfilt(y,kernel_size = medavg)  
        ax.plot(x,y,'k-')    
        if xlims != [0,0]: 
            xmin,xmax = xlims
            ax.set_xlim(xmin,xmax)   
        if ylims != [0,0]: 
            ymin,ymax = ylims
            ax.set_ylim(ymin,ymax)
        else:
            ax.set_ylim(0,np.max(y)*1.1)  
        ax.grid()
        return fig


##define functions for reading in images
def read_calfile(fname):
    ''' Reads grism calibration file (.csv), extracts header line, subimage box, rotation angle, 
    and coefficients to generate functions that convert pixels to wavelength and gain vs wavelength'''
    with open(fname,'r') as f:
        lines = f.read().split('\n')
        hdr_line = lines[0]
        box = [int(x) for x in lines[1].split(',')]
        angle = float(lines[2])
        wavelength_coefficients = np.array([float(x) for x in lines[3].split(',')])
        gain_coefficients = np.array([float(x) for x in lines[4].split(',')] )    
    return(hdr_line, box, angle, wavelength_coefficients, gain_coefficients)

def read_jacoby_file(fname):
        ''' Reads a standard Jacoby .csv file, packs wavelength and amplitude arrays into spectrum object '''
        wave_ref, spec_ref = np.loadtxt(fname, unpack=True, \
        comments='#',usecols=(0,1),dtype = float)
        spec_ref /= np.max(spec_ref)
        jacoby_spectrum = np.vstack([wave_ref,spec_ref])  # contains wavelength and amp. arrays
        return jacoby_spectrum

# read in .last files
with open('imget.last', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        grism_image = row['grism_image']
        cal_file = row['cal_file']
if os.path.isfile('spectrum.last'):
    with open('box.last', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rotangle = row['rotangle']
            xwidth = row['xwidth']
            ywidth = row['ywidth']
            xoffset = row['xoffset']
            yoffset = row['yoffset']
            mybox= row['mybox']
            f_wave=row['f_wave']
            f_gain=row['f_gain']
            savedir=row['savedir']
else:
    with open('box.last', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rotangle = row['rotangle']
            xwidth = row['xwidth']
            ywidth = row['ywidth']
            xoffset = row['xoffset']
            yoffset = row['yoffset']
            mybox= row['mybox']
            f_wave=row['f_wave']
            f_gain=row['f_gain']
            savedir=row['savedir']

params = [rotangle, xwidth, ywidth, xoffset, yoffset, mybox, f_wave, f_gain, savedir]


# initiating click command and main function
@click.command(
        epilog="""Check out the documentation at
                https://pyscope.readthedocs.io/ for more
                information."""
)

@click.option(
    "-sd",
    "--savedir",
    help='the directory to save output images in',
)

def ga_spectra(
    savedir,
):
    '''
    Grism analysis

    This script takes the subimage created by 'grism_box.py' and draws some helpful spectra. It will create figures for raw and calibrated spectra, 
    gain vs. wavelength, and several gaussian fits to the Balmer lines. It will also save these figures to a directory of your choice. 

    This is an adaptation of original code by RLM.

    V0.1 (24 January 2024) CHR
    '''

    # read in .last files
    with open('imget.last', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            grism_image = row['grism_image']
            cal_file = row['cal_file']
    with open('box.last', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rotangle = int(row['rotangle'])
            xwidth = int(row['xwidth'])
            ywidth = int(row['ywidth'])
            xoffset = int(row['xoffset'])
            yoffset = int(row['yoffset'])
            mybox= row['mybox']
            f_wave=row['f_wave']
            f_gain=row['f_gain']
            if savedir is None:
                savedir=row['savedir']
            else:
                pass

    # read in image and calibration file
    im, hdr = getdata(grism_image, 0, header=True)
    cal_hdr, box, rot_angle, wavelength_coefficients, gain_coefficients = read_calfile(cal_file)
    f_wave = np.poly1d(wavelength_coefficients)
    f_gain = np.poly1d(gain_coefficients)

    #Defining subimage box dimensions as specified by user
    click.echo('Creating subimage...')
    xs,ys = im.shape
    xmin = xs//2 -1000 + xoffset
    xmax = xmin + xwidth
    ymin = ys//2 + yoffset - ywidth//2
    ymax = ymin + ywidth
    mybox  = [xmin,xmax,ymin,ymax]
    xi, yi = im.shape

    # Instantiate with rotation angle and subimage box
    B = grism_utils(grism_image,cal_file,rotangle,mybox,f_wave,f_gain)

    # Create subimage using optional box parameters
    subim = B.create_subimage()
    xs,ys = subim.shape
    zmax = np.max(subim)

    # Plot subimage
    object_name, obs_date,telescope,camera,title,im,rot_angle, box, _,_ = B.summary_info()
    fig = B.plot_image(image=subim,figsize =(10,2),cmap='jet',title=title)
    click.echo(object_name)
    
    # Plot uncalibrated spectrum
    click.echo('Plotting uncalibrated spectrum...')
    B = grism_utils(grism_image,cal_file,rotangle,mybox,f_wave,f_gain)
    object_name, obs_date,telescope,camera,title,im,rot_angle, box, _,_ = B.summary_info()
    
    spectrum = B.calibrate_spectrum(subim, norm=True)
    fig = B.plot_spectrum(spectrum, xaxis='pixel', yaxis='uncal', subrange = slice(300,2000),\
                        title='%s Uncalibrated Spectrum' % object_name, medavg = 5,xlims =[0,0],ylims =[0,0])
    fig.savefig(f'{object_name}(uncalibrated)')
    
    # # Plot gain curve
    # click.echo('Plotting gain curve...')
    # fig = B.plot_gain_curve(spectrum,color='r',title='Gain curve')
    # fig.savefig('gain curve')
    
    # # Plot calibrated spectrum
    # click.echo('Plotting calibrated spectrum...')
    # fig = B.plot_spectrum(spectrum, xaxis='wave', yaxis='cal',medavg=7, title='%s Calibrated Spectrum' % object_name,
    # xlims=[400,750],ylims =[0,0],plot_balmer=True)
    # fig.savefig(f'{object_name}.png',dpi=200, bbox_inches = 'tight', transparent=False, facecolor='whitesmoke')    

    # # Fit a line with Gaussian and  plot
    # click.echo('Fitting Balmer Gaussian...')
    # wave_min= []
    # wave_max= []
    # for i in range(5):
    #     wave_min.append(balmer[i]-20)
    #     wave_max.append(balmer[i]+20)
    # for i in range(5):
    #     params, wave, amp, amp_mod = B.fit_gaussian(spectrum,wave_min[i],wave_max[i])
    #     wave_ctr,wave_ctr_err,fwhm,fwhm_err,a,a_err = params
    #     click.echo(f'Wave_ctr = {round(wave_ctr, 1)} +/- {round(wave_ctr_err, 1)} nm, FWHM = {round(fwhm, 1)} +/- {round(fwhm_err, 1)} nm' )
    #     fig = B.plot_spectral_line(wave,amp,amp_mod,wave_ctr,wave_ctr_err,fwhm,fwhm_err,color='red',title=object_name)
    #     fig.savefig(f'{object_name} {round(wave_ctr, 1)}nm.png',dpi=200, bbox_inches = 'tight', transparent=False, facecolor='whitesmoke')

    # #save images to directory
    # if savedir == 'None':
    #     pass
    # else:
    #     click.echo(f'Saving images to {savedir}...')
    #     os.rename(f'{object_name}(uncalibrated).png', f'{savedir}/{object_name}(uncalibrated).png')
    #     os.rename('gain curve.png', f'{savedir}/gain curve.png')
    #     os.rename(f'{object_name}.png', f'{savedir}/{object_name}.png')
    #     os.rename(f'{object_name} 397.0nm.png', f'{savedir}/{object_name} 397.0nm.png')
    #     os.rename(f'{object_name} 410.17nm.png', f'{savedir}/{object_name} 410.17nm.png')
    #     os.rename(f'{object_name} 434.05nm.png', f'{savedir}/{object_name} 434.05nm.png')
    #     os.rename(f'{object_name} 486.14nm.png', f'{savedir}/{object_name} 486.14nm.png')
    #     os.rename(f'{object_name} 656.45nm.png', f'{savedir}/{object_name} 656.45nm.png')

    # # write .last file
    # with open('spectrum.last', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     field = ["rotangle","xwidth","ywidth","xoffset","yoffset","mybox","f_wave","f_gain","savedir"]
    #     writer.writerow(field)
    #     writer.writerow([f"{rotangle}",f"{xwidth}",f"{ywidth}",f"{xoffset}",f"{yoffset}",f"{mybox}",f"{f_wave}",f"{f_gain}",f"{savedir}"])

    print(spectrum)
    print(subim)


if __name__ == '__main__':
    ga_spectra()
