from . import abstract

import os
import astropy.io.fits as pyfits
from astroquery.astrometry_net import AstrometryNet

class WCS(abstract.WCS):
    def __init__(self, *args, **kwargs):
        self._solver = AstrometryNet()
    
    def Solve(self, *args, **kwargs):
        
        filepath = args[0]
        if filepath is None or not os.path.isfile(filepath): raise Exception('No filepath or invalid filepath provided.')

        try_again = True
        submission_id = None
        while try_again:
            try:
                if not submission_id:
                    wcs_header = self.solver.solve_from_image(filepath, submission_id=submission_id, **kwargs)
                else:
                    wcs_header = self.solver.monitor_submission(submission_id, solve_timeout=kwargs.get('solve_timeout', 120))
            except TimeoutError as e:
                submission_id = e.args[1]
            else:
                try_again = False

        if wcs_header:
            with pyfits.open(filepath, mode='update') as hdul:
                hdul[0].header.update(wcs_header)
        else:
            return False

    @property
    def solver(self):
        return self._solver