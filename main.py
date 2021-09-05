from detection_models import *

from Core import *

def main():
    model = MaskRCNNModule(path_to_wages='D:\\materialy_pwr\\7sem\\app\\mask_rcnn_tomato_0040.h5')
    core = Core(model)
    #core.runImage("RealSense_T20190528_230215_R02_P008860_H1630_A+000_RGB.tiff")
    core.runVideo("D:\\materialy_pwr\\7sem\\tomato_own_dataset\\video\\fifth.bag")
    #core.runLive()



if __name__ == "__main__":
    main()