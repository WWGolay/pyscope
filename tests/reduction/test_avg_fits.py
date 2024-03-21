import pytest
import numpy as np
from astropy.io import fits
from pyscope.reduction import avg_fits
import glob


def test_avg_fits(tmp_path):
    # create a test FITS file of 
    for i in range(1,11):
        data = np.ones((10,10), dtype=np.uint16)
        fits.writeto(f"{tmp_path}/avg_fits_test_{i}.fits", data * i, overwrite=True)
        print(data * i)

    files = glob.glob(f"{tmp_path}/avg_fits_test_*.fits")
    
    # test median mode
    avg_fits(mode="0", outfile=f"{tmp_path}/averaged/avg_fits_median.fits", fnames=files, verbose=True)
    med_data = fits.getdata(f"{tmp_path}/averaged/avg_fits_median.fits")
    med = np.ones((10,10), dtype=np.uint16)
    med = med * np.mean(range(1,11)).astype(np.uint16)
    print(med_data, med)
    assert np.array_equal(med_data, med)
    print("correctly calculated median")
    
    # test mean mode
    avg_fits(mode="1", outfile=f"{tmp_path}/averaged/avg_fits_mean.fits", fnames=files, verbose=True)
    mean_data = fits.getdata(f"{tmp_path}/averaged/avg_fits_mean.fits")
    mean = np.ones((10,10), dtype=np.uint16)
    mean = mean * np.mean(range(1,11)).astype(np.uint16)
    print(mean_data, mean)
    assert np.array_equal(mean_data, mean)
    print("correctly calculated mean")

    # test different data type
    avg_fits(mode="1", outfile=f"{tmp_path}/averaged/avg_fits_mean.fits", fnames=files, datatype='float64', verbose=True)
    mean_dataf = fits.getdata(f"{tmp_path}/averaged/avg_fits_mean.fits")
    meanf = np.ones((10,10), dtype=np.float64)
    meanf = meanf * np.mean(range(1,11)).astype(np.float64)
    assert np.array_equal(mean_dataf, meanf)
    print(mean_dataf, meanf)
    print("correctly calculated mean with different data type")

    # test different bad datatype
    with pytest.raises(ValueError, match="Invalid datatype: float100") as e:
        avg_fits(mode="1", outfile=f"{tmp_path}/averaged/avg_fits_mean.fits", fnames=files, datatype='float100', verbose=True)
    print("passed invalid datatype test")

    print("passed all tests!")
    return


if __name__ == "__main__":
    test_avg_fits("./tmp_dir")