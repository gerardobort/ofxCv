#include "ofxCv/Stereo.h"
#include "ofGraphics.h"

namespace ofxCv {
	
	using namespace cv;


    // --------------
	Camera::Camera(){
        isReady = false;
        distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrixRefined = cv::Mat::eye(3, 3, CV_64F);

        objectPoints = std::vector<std::vector<cv::Point3f> >(1);
        imagePoints = std::vector<std::vector<cv::Point2f> >(1);
	}
	
	Camera::~Camera(){
	}

	vector<Point2f> Camera::calibrate(Mat image){
        boardSize = cv::Size(9,7);
        imageSize = image.size();

        // CALIB_CB_FAST_CHECK saves a lot of time on images
        // that do not contain any chessboard corners
        bool success = findChessboardCorners(image, boardSize, imagePoints[0], CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

        if (success) {
            isReady = true;
            // call-again for better corner detection
            cornerSubPix(image, imagePoints[0], cv::Size(11, 11), cv::Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            // drawChessboardCorners(image, boardSize, Mat(imagePoints[0]), success);
            // http://programmingexamples.net/wiki/OpenCV/CheckerboardCalibration

            float squareSize = 10.f; // This is "1 arbitrary unit"

            // Find the chessboard corners
            objectPoints[0] = Create3DChessboardCorners(boardSize, squareSize);
            
            int flags = CV_CALIB_FIX_K4|CV_CALIB_FIX_K5;
            double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distortionCoefficients, rotationVectors, translationVectors, flags);
            cameraMatrixRefined = getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, imageSize, 0, imageSize, 0, true);

            std::cout << "RMS: " << rms << std::endl;
            std::cout << "Distortion coefficients: " << distortionCoefficients << std::endl;
            std::cout << "Camera matrix: " << cameraMatrix << std::endl;
            std::cout << "Camera matrix refined: " << cameraMatrixRefined << std::endl;
        }

        return imagePoints[0];
    }

    void Camera::rectify(ofImage srcImage, ofImage& dstImage) {
        // method 1
        cv::Mat srcMat = toCv(srcImage);
        cv::Mat dstMat;
        cv::undistort(srcMat, dstMat, cameraMatrixRefined, distortionCoefficients);

        // this conversion method doesn't work very well when applying later StereoSGBM
        //ofPixels pixels;
        //toOf(dstMat, pixels);
        //dstImage.setFromPixels(pixels);

        // copy resultant matrix into dstImage
        toOf(dstMat, dstImage);

        // for some reason this seems to be necessary, need to research if there's performance to improve here, apparently yes.
        // setting CV_*** Mat types, or pre-allocating the dstImage is not enough.
        dstImage.setImageType(OF_IMAGE_COLOR);
    }

    std::vector<cv::Point3f> Camera::Create3DChessboardCorners(cv::Size boardSize, float squareSize) {
        // This function creates the 3D points of your chessboard in its own coordinate system
        std::vector<cv::Point3f> corners;
     
        for( int i = 0; i < boardSize.height; i++ )
        {
            for( int j = 0; j < boardSize.width; j++ )
            {
                corners.push_back(cv::Point3f(float(j*squareSize),
                                                                    float(i*squareSize), 0));
            }
        }
     
        return corners;
    }







	
    // --------------
	Stereo::Stereo(int ndisparities, int SADWindowSize)
    :ndisparities(ndisparities)
    ,SADWindowSize(SADWindowSize) {
        isReady = false;
        reload();
	}
	
	Stereo::~Stereo(){
	}

    void Stereo::setNDisparities(int n) {
        if (n % 16 != 0) {
            ofLogWarning() << "ofxCv::Stereo: Invalid nDisparities value " << n << ".  The value must be multiplier of 16.";
            return;
        }
        ndisparities = n;
    }

    void Stereo::setSADWindowSize(int s) {
        if (s < 5 || s > 255 || s % 2 == 0 || s > imgDisparity8U.cols || s > imgDisparity8U.cols) {
            ofLogWarning() << "ofxCv::Stereo: Invalid SADWindowSize value " << s << ".  The value must be odd, be within 5..255 and be not larger than image width or height in function.";
            return;
        }
        SADWindowSize = s;
    }

    void Stereo::reload() {
        /*
            StereoSGBM
                int  minDisparity,
                int  numDisparities,
                int  SADWindowSize
        */
        // sbm = new StereoBM(StereoBM::PREFILTER_XSOBEL, ndisparities, SADWindowSize);
        /*
            StereoSGBM
                int  minDisparity,
                int  numDisparities,
                int  SADWindowSize,
                int  P1 = 0,
                int  P2 = 0,
                int  disp12MaxDiff = 0,
                int  preFilterCap = 0,
                int  uniquenessRatio = 0,
                int  speckleWindowSize = 0,
                int  speckleRange = 0,
                bool fullDP = false 
        */
        sbm = new StereoSGBM(0, ndisparities, SADWindowSize);
    }
	
	//call with two images
	void Stereo::compute(Mat leftImage, Mat rightImage){
        imgDisparity16S = Mat(leftImage.rows, leftImage.cols, CV_16S);
        imgDisparity8U = Mat(leftImage.rows, leftImage.cols, CV_8UC3);

        //-- 3. Calculate the disparity image
        sbm->operator()(leftImage, rightImage, imgDisparity16S);
	}
	
	void Stereo::draw(){
        double minVal, maxVal;

        //-- Check its extreme values
        minMaxLoc(imgDisparity16S, &minVal, &maxVal);

        //-- 4. Display it as a CV_8UC1 image
        imgDisparity16S.convertTo(imgDisparity8U, CV_8UC3, (255)/(maxVal - minVal));

        ofPixels pix8u;
        toOf(imgDisparity8U, pix8u);
        ofImage img;
        img.setFromPixels(pix8u);
        img.draw(0, 0);
	}


    void Stereo::calibrate(Camera& leftCamera, Camera& rightCamera) {
        double res = stereoCalibrate(
            leftCamera.objectPoints,
            leftCamera.imagePoints, rightCamera.imagePoints,
            leftCamera.cameraMatrixRefined, leftCamera.distortionCoefficients,
            rightCamera.cameraMatrixRefined, rightCamera.distortionCoefficients,
            leftCamera.imageSize,
            // output matrices
            R, T, E, F);
        isReady = true;
    }

    void Stereo::rectifyLeft(ofImage& leftImage) {
        cv::Mat img = toCv(leftImage);
        img *= R;
        img *= T;
        //cv::warpAffine(img,R*T, (img.cols,img.rows))
        ofPixels pix;
        toOf(img, pix);
        leftImage.setFromPixels(pix);
    }
        
}
