
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

using namespace cv::xfeatures2d;

class Match {


    std::string rect_map = "/home/fusy/Documents/risiko/imgs/rect_map.jpg";
    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110943.jpg";        // india has 2
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110513.jpg";        // india has 0
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110627.jpg";          // india has 1
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110540.jpg"; // difficult: map is not well localized
    std::string india_map = "/home/fusy/Documents/risiko/imgs/rect_map_india.jpg";
    Mat img_object;
    Mat img_scene;
    Mat img_scene_color;
    Mat india;

    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;

    std::vector<DMatch> good_matches;
    Mat img_matches;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    // Homography
    Mat H;

    std::vector<Point2f> obj_corners;
    std::vector<Point2f> scene_corners;

    void loadImages() {
        img_object = imread( rect_map, IMREAD_GRAYSCALE );
        img_scene = imread( test_map, IMREAD_GRAYSCALE );
        img_scene_color = imread( test_map, IMREAD_COLOR );
        india = imread( india_map, IMREAD_GRAYSCALE );

        if ( img_object.empty() || img_scene.empty() )
        {
            cout << "Could not open or find the image!\n" << endl;
            return ;
        }
    }

    void loadKeypointsAndDescriptors() {
        int minHessian = 400;
        Ptr<SURF> detector = SURF::create( minHessian );
        detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
        detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
    }

    void findMatches() {
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.68f;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        for( size_t i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }
    }

    void computeObjectCorners() {
        obj_corners.resize(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)img_object.cols, 0 );
        obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
        obj_corners[3] = Point2f( 0, (float)img_object.rows );
    }

    void drawLines() {
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
              scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
              scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
              scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
              scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    }

public:
    Match() {

        loadImages();

        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        loadKeypointsAndDescriptors();

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        findMatches();

        //-- Draw matches
        drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Localize the object
        H = findHomography( obj, scene, RANSAC );

        //-- Get the corners from the image_1 ( the object to be "detected" )
        computeObjectCorners();

        // compute corners in scene image frame
        scene_corners.resize(4);
        perspectiveTransform( obj_corners, scene_corners, H);

        drawLines();

        //-- Show detected matches
        cv::resize(img_matches, img_matches, cv::Size(img_matches.cols/2, img_matches.rows/2));
        imshow("Good Matches & Object detection", img_matches );
        waitKey();




        Mat th;
        Mat img_scene_hsv;
        cvtColor(img_scene_color, img_scene_hsv, COLOR_BGR2HSV);
        inRange(img_scene_hsv, Scalar(110, 142, 68), Scalar(141, 255, 255), th);
        cv::resize(th, th, cv::Size(th.cols/2, th.rows/2));
        imshow("thresholded", th );
        waitKey();



        cv::Mat projected = img_scene.clone();

        size_t pix_cont = 0;

        for (int r = 0; r<india.rows; r++) {
            for (int c=0; c<india.cols; c++) {
                if (india.at<uchar>(r, c) == 0) {
                    std::vector<Point2f> camera_corners;
                    Point2f p(c, r);
                    camera_corners.push_back(p);
                    std::vector<Point2f> world_corners;
                    perspectiveTransform(camera_corners, world_corners, H);
                    projected.at<uchar>((int) world_corners[0].y, (int) world_corners[0].x) = 255;
                    if (th.at<uchar>((int) world_corners[0].y, (int) world_corners[0].x)==255) {
                        pix_cont++;
                    }
                }

            }
        }

        cout << "PIX_CONT: " << pix_cont << endl;

        cv::resize(projected, projected, cv::Size(projected.cols/2, projected.rows/2));
        imshow("second", projected );
        waitKey();
    }

};

int main(int argc, char** argv) {

    Match m;

    return 0;
}