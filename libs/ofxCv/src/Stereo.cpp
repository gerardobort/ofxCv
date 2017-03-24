#include "ofxCv/Stereo.h"
#include "ofGraphics.h"

namespace ofxCv {
	
	using namespace cv;
	
	Stereo::Stereo(int ndisparities, int SADWindowSize)
    :ndisparities(ndisparities)
    ,SADWindowSize(SADWindowSize) {
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
        //-- 2. Call the constructor for StereoBM
        sbm = new StereoBM(StereoBM::BASIC_PRESET, ndisparities, SADWindowSize);
        //sbm = new StereoSGBM(0, ndisparities, SADWindowSize);
    }
	
	//call with two images
	void Stereo::compute(Mat leftImage, Mat rightImage){
        imgDisparity16S = Mat(leftImage.rows, leftImage.cols, CV_16S);
        imgDisparity8U = Mat(leftImage.rows, leftImage.cols, CV_8UC1);
        //-- 3. Calculate the disparity image
        sbm->operator()(leftImage, rightImage, imgDisparity16S);

        //-- Check its extreme values
        minMaxLoc(imgDisparity16S, &minVal, &maxVal);
	}
	
	void Stereo::draw(){
        //-- 4. Display it as a CV_8UC1 image
        imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));

        ofPixels pix8u;
        toOf(imgDisparity8U, pix8u);
        ofImage img;
        img.setFromPixels(pix8u);
        img.draw(0, 0);
	}


        
    // --------------
	Camera::Camera(){
        isReady = false;
        distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrixRefined = cv::Mat::eye(3, 3, CV_64F);
	}
	
	Camera::~Camera(){
	}

	vector<Point2f> Camera::calibrate(Mat image){
        cv::Size boardSize(9,7);
        cv::Size imageSize = image.size();
        std::vector<std::vector<cv::Point2f> > imagePoints(1);

        // CALIB_CB_FAST_CHECK saves a lot of time on images
        // that do not contain any chessboard corners
        bool success = findChessboardCorners(image, boardSize, imagePoints[0], CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

        if (success) {
            isReady = true;
            cornerSubPix(image, imagePoints[0], cv::Size(11, 11), cv::Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            // drawChessboardCorners(image, boardSize, Mat(imagePoints[0]), success);
            // http://programmingexamples.net/wiki/OpenCV/CheckerboardCalibration

            float squareSize = 1.f; // This is "1 arbitrary unit"

            // Find the chessboard corners
            std::vector<std::vector<cv::Point3f> > objectPoints(1);
            objectPoints[0] = Create3DChessboardCorners(boardSize, squareSize);
            
            int flags = 0|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5;
            double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distortionCoefficients, rotationVectors, translationVectors, flags);
            cameraMatrixRefined = getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, imageSize, 0.2);

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

        // copy resultant matrix into dstImage
        toOf(dstMat, dstImage);
        dstImage.setImageType(OF_IMAGE_GRAYSCALE);
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

}
