title ="this is a TEST" # only one title is allowed
observer: {wgolay@cfa.harvard.edu}
obs (wgolay@uiowa.edu) # can have multiple observers
code               wgolay # whitespace ignored, only one code is allowed
datestart 2017-08-17 # only one datestart is allowed, valid range
dateend 2024-12-31 # only one dateend is allowed, valid range

# this is a comment
! this is a comment too
% another type of comment


block start
    source test
    ra = 12:34:56.7
    dec = -12:34:56.7
    readout 0
    repositioning true
    exposure 1
    filter i,g,r
                    nexp 3
utstart 2024-01-01T12:34:56.7   cad 00:10:00 sc 00:02:00.0
block end


block s

source Algol
filename "Algol"
binning 2x2
frame_p 10,10
frame_s 100,100
filt l
exposure 10
comment "this is a comment"

                    block e


block sta
source "C/2003 A2" non_sidereal 1                    # this is a comment
priority 10
repeat 5
do-not-interrupt 1
filt l
exp 10
block e






source      Algol filt = r exp: 10






block star
sou "Jupiter" non-sidereal true
priority 11 filt l   exp 1
block e
