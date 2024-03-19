
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
    
    def calibrate_spectrum(self,subim):
        '''Calculates raw spectrum by summing pixels in all vertical slices, then apply wavelength, gain calibration'''

        pixels = np.arange(subim.shape[1])
        # find sum of pixel values along each column (spectral channel)
        uncal_amp = []
        for pixel in pixels:
            ymax,signal,signal_max = self.__calc_channel_signal(subim,pixel)
            uncal_amp.append(signal)
        uncal_amp = np.array(uncal_amp)
        uncal_amp /= np.max(uncal_amp) # Normalize
     
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

    def plot_spectral_line(self,wave,amp,amp_mod,color='r',title=''):
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


#initiating click command and main function
@click.command(
        epilog="""Check out the documentation at
                https://pyscope.readthedocs.io/ for more
                information."""
)
@click.option(
    "-u",
    "--url",
    default = 'https://macroconsortium.org/images/',
    show_default=True,
    help='url of database containing images. (ONLY COMPATIBLE WTH CHRONOS)'
)
@click.option(
    "-p",
    "--parent-dir",
    default = None,
    show_default=True,
    help='Parent directory containing images.'
)
@click.option(
    "-y",
    "--year",
    default = '2023',
    show_default=True,
    help='The year of the image you want to analyze.'
)
@click.option(
"-o",
"--obs-code",
prompt='Observer code',
help='The name of the image you want to analyze.'
)
@click.option(
    "-n",
    "--obs-night",
    prompt='Observing night',
    help='The observation night of the image you want to analyze.'
)
@click.option(
"-i",
"--image-number",
prompt='Image number',
help='The number of the image you want to analyze. Must be 2 digits (XX)'
)
@click.option(
"-c",
"--cal-dir",
help='The full directory of the calibration file.',
default='grism_cal.csv'
)

def ga_cli(
    url,
    parent_dir,
    year,
    obs_night,
    obs_code,
    image_number,
    cal_dir,
):
    '''
    Grism analysis

    This script looks for a FITS grism image and calibration file specified by the user. It can take images from a specified directory or URL.
    They are saved in the directory for analysis by 'grism_box.py' and 'grism_spectrum.py'.

    This is an adaptation of the original code by RLM.

    V0.1 (24 January 2024) CHR
    '''
    #update this to use a dictionary and less lines
    institutions=['macalester','augustana','coe','iowa','knox']
    if obs_code[0] == 'm':
        institution = institutions[0]
    elif obs_code[0] == 'a':
        institution = institutions[1]
    elif obs_code[0] == 'c':
        institution = institutions[2]
    elif obs_code[0] == 'i':
        institution = institutions[3]
    elif obs_code[0] == 'k':
        institution = institutions[4]

    grism_image = obs_code + obs_night + image_number + '.fts'
    url = url + '/' + institution + '/' + obs_code + '/' + year + '/' + obs_night + '/' + grism_image

    if parent_dir is not None:
        image_path = parent_dir + '/' + grism_image
    else:
        pass

    if validators.url(url):
        fits_head = fits.getheader(url)
        fits_data = fits.getdata(url)
        hdu = fits.PrimaryHDU(data=fits_data,header=fits_head)
        hdu.writeto(grism_image,overwrite=True)
        click.echo(
            f"Found grism image: {grism_image} at {url}"
        )
        im, hdr = getdata(grism_image, 0, header=True)
        object_name  = hdr['OBJECT']
        obs_date = hdr['DATE-OBS']
    elif os.path.exists(image_path):
        im, hdr = getdata(grism_image, 0, header=True)
        object_name  = hdr['OBJECT']
        obs_date = hdr['DATE-OBS']
    else:
        click.echo(
            f"Cannot find {grism_image}, stopping"
        )
        raise StopExecution  
    if os.path.exists(cal_dir):
        #adjust to find the actual name of the cal file
        cal_file = 'grism_cal.csv'
        click.echo(
            f"Found calibration file: {cal_file}"
        )
    else:
        click.echo(
            f"Cannot find calibration file:{cal_file}, stopping"
        )
        raise StopExecution 
    with open('imget.last', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["grism_image", "cal_file"]
        
        writer.writerow(field)
        writer.writerow([f"{grism_image}", f"{cal_file}"])
        

if __name__ == '__main__':
    ga_cli()

