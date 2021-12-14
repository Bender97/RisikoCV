
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

std::string rect_map = "/home/fusy/Documents/risiko/imgs/rect_map.jpg";
std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110513.jpg";
//std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110540.jpg"; // difficult: map is not well localized
std::string india_map = "/home/fusy/Documents/risiko/imgs/rect_map_india.jpg";

class Match {

public:
    Match() {

        Mat img_object = imread( rect_map, IMREAD_GRAYSCALE );
        Mat img_scene = imread( test_map, IMREAD_GRAYSCALE );
        Mat india = imread( india_map, IMREAD_GRAYSCALE );

        if ( img_object.empty() || img_scene.empty() )
        {
            cout << "Could not open or find the image!\n" << endl;
            return ;
        }
        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        int minHessian = 400;
        Ptr<SURF> detector = SURF::create( minHessian );
        std::vector<KeyPoint> keypoints_object, keypoints_scene;
        Mat descriptors_object, descriptors_scene;
        detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
        detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.68f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        //-- Draw matches
        Mat img_matches;
        drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;
        for( size_t i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }
        Mat H = findHomography( obj, scene, RANSAC );
        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)img_object.cols, 0 );
        obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
        obj_corners[3] = Point2f( 0, (float)img_object.rows );
        std::vector<Point2f> scene_corners(4);
        perspectiveTransform( obj_corners, scene_corners, H);

//        Mat map = img_scene.clone();
//        Mat mask = Mat::zeros( img_scene.rows, img_scene.cols, img_scene.type() );
//        Mat warp_mat = getAffineTransform( obj_corners.data(), scene_corners.data() );
//        warpAffine( india, mask, warp_mat, india.size() );

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
              scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
              scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
              scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
              scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        //-- Show detected matches

        cv::resize(img_matches, img_matches, cv::Size(img_matches.cols/2, img_matches.rows/2));
        imshow("Good Matches & Object detection", img_matches );
        waitKey();

        cv::Mat projected = img_scene.clone();

        for (int r = 0; r<india.rows; r++) {
            for (int c=0; c<india.cols; c++) {
                if (india.at<uchar>(r, c) == 0) {
                    std::vector<Point2f> camera_corners;
                    Point2f p(c, r);
                    camera_corners.push_back(p);
                    std::vector<Point2f> world_corners;
                    perspectiveTransform(camera_corners, world_corners, H);
                    projected.at<uchar>((int) world_corners[0].y, (int) world_corners[0].x) = 255;
                }

            }
        }

        cv::resize(projected, projected, cv::Size(projected.cols/2, projected.rows/2));
        imshow("second", projected );
        waitKey();
    //        cv::resize(mask, mask, cv::Size(mask.cols/2, mask.rows/2));
    //        imshow("second", mask );
    //        waitKey();
        return;
    }

};

int main(int argc, char** argv) {

    Match m;

    return 0;
}