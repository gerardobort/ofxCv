#pragma once

#include "ofxCv.h"

namespace ofxCv {

    /**
     * Reference:
     *  @see https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/stereoBM/SBM_Sample.cpp
     *  @see http://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
     *  @author Gerardo Bort <gerardobort@gmail.com>
     */
	
	class Stereo {
	public:
		Stereo(int ndisparities, int SADWindowSize);
		virtual ~Stereo();
		
        template <class T>
		void compute(T& leftImage, T& rightImage) {
            // StereoBM requires CV_8UC1 for both input images, StereoSGBM accepts color inputs
            // leftImage.setImageType(OF_IMAGE_GRAYSCALE);
            // rightImage.setImageType(OF_IMAGE_GRAYSCALE);

            compute(toCv(leftImage), toCv(rightImage));
        }
		void compute(cv::Mat leftImage, cv::Mat rightImage);
		
        void setNDisparities(int n);
        void setSADWindowSize(int s);
        void reload();
		void draw();


        // chessboard calibration
        template <class T>
		ofPolyline calibrate(T& image, bool& success) {
            return toOf(calibrate(toCv(image), success));
        }
	    vector<cv::Point2f> calibrate(cv::Mat image, bool& success);
        vector<cv::Point3f> Create3DChessboardCorners(cv::Size boardSize, float squareSize);

        cv::Mat dst;
        void getDst(ofImage& img) {
            return toOf(dst, img);
        }
        
    private:
        int ndisparities = 16*5; /**< Range of disparity */
        int SADWindowSize = 21; /**< Size of the block window. Must be odd */
        cv::Mat imgDisparity16S;
        cv::Mat imgDisparity8U;
        //cv::Ptr<cv::StereoBM> sbm;
        cv::Ptr<cv::StereoSGBM> sbm;
        
    protected:
	};


	class Camera {
	public:
		Camera();
		virtual ~Camera();

        // chessboard calibration
        template <class T>
		ofPolyline calibrate(T& image) {
            image.setImageType(OF_IMAGE_GRAYSCALE);
            return toOf(calibrate(toCv(image)));
        }
	    vector<cv::Point2f> calibrate(cv::Mat image);
        vector<cv::Point3f> Create3DChessboardCorners(cv::Size boardSize, float squareSize);

        void rectify(ofImage srcImage, ofImage& dstImage);
            
        std::vector<cv::Mat> rotationVectors;
        std::vector<cv::Mat> translationVectors;
        cv::Mat distortionCoefficients;
        cv::Mat cameraMatrix;
        cv::Mat cameraMatrixRefined;

        bool isReady;
        
    private:
    protected:
	};
	
}
