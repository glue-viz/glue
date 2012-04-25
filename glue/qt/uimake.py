from PyQt4.uic import compileUi
from glob import glob

def uimake():
    files = glob('*.ui')
    for infile in files:
        outfile = 'ui_' + infile[:-3] + '.py'
        outfile = open(outfile, 'w')
        compileUi(infile, outfile)

if __name__ == "__main__":
    uimake()