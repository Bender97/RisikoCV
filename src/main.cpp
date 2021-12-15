
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
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110513.jpg";        // india has 0
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110720.jpg";          // india has 1
    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110738.jpg";        // india has 2
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110720.jpg";        // africadelnord has 1
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110800.jpg";        // africadelnord has 3
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110943.jpg";        // africadelnord has 4
//    std::string test_map = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110540.jpg"; // difficult: map is not well localized
    std::string india_map = "/home/fusy/Documents/risiko/imgs/rect_map_africadelnord.jpg";
//    std::string india_map = "/home/fusy/Documents/risiko/imgs/rect_map_india.jpg";
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

    cv::Mat armies_projected_mask;

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
    Match(std::string scenepath, std::string objpath) {

        img_object = imread( rect_map, IMREAD_GRAYSCALE );
        img_scene = imread( scenepath, IMREAD_GRAYSCALE );
        img_scene_color = imread( scenepath, IMREAD_COLOR );
        india = imread( objpath, IMREAD_GRAYSCALE );

        if ( img_object.empty() || img_scene.empty() )
        {
            cout << "Could not open or find the image!\n" << endl;
            return ;
        }

        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        loadKeypointsAndDescriptors();

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        findMatches();

        //-- Draw matches
//        drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
//                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Localize the object
        H = findHomography( obj, scene, RANSAC );

        //-- Get the corners from the image_1 ( the object to be "detected" )
        computeObjectCorners();

        // compute corners in scene image frame
        scene_corners.resize(4);
        perspectiveTransform( obj_corners, scene_corners, H);

//        drawLines();
//
//        //-- Show detected matches
//        cv::resize(img_matches, img_matches, cv::Size(img_matches.cols/2, img_matches.rows/2));
//        imshow("Good Matches & Object detection", img_matches );
//        waitKey();




        Mat thresholded_img;
        Mat th_copy;
        Mat img_scene_hsv;
        cvtColor(img_scene_color, img_scene_hsv, COLOR_BGR2HSV);
//        inRange(img_scene_hsv, Scalar(108, 125, 38), Scalar(137, 233, 223), th);
        inRange(img_scene_hsv, Scalar(100, 135, 97), Scalar(145, 255, 215), thresholded_img);

        int dilate_size = 4;

        Mat element = getStructuringElement( MORPH_RECT,
                                             Size( 2*dilate_size + 1, 2*dilate_size+1 ),
                                             Point( dilate_size, dilate_size ) );
        dilate( thresholded_img, thresholded_img, element );

        thresholded_img = 255- thresholded_img;

//        cv::resize(th, th_copy, cv::Size(th.cols/2, th.rows/2));
//        imshow("thresholded", th_copy );


        cv::Mat projected = img_scene.clone();
        cv::Mat state_projected_mask = Mat::zeros(projected.rows, projected.cols, CV_8UC1);

        size_t pix_cont = 0;

        size_t original_india_area = india.rows*india.cols;
        size_t projected_state_area = 0;

        armies_projected_mask = Mat::zeros(projected.rows, projected.cols, CV_8UC1);

        for (int r = 0; r<india.rows; r++) {
            for (int c=0; c<india.cols; c++) {
                if (india.at<uchar>(r, c) == 255) {

                    std::vector<Point2f> camera_corners;
                    Point2f p(c, r);
                    camera_corners.push_back(p);
                    std::vector<Point2f> world_corners;
                    perspectiveTransform(camera_corners, world_corners, H);
                    int x_proj = (int) world_corners[0].x;
                    int y_proj = (int) world_corners[0].y;

                    projected.at<uchar>(y_proj, x_proj) = 0;                // annerisci lo stato sulla mappa colorata (grayscale)
                    state_projected_mask.at<uchar>(y_proj, x_proj) = 255;   // segna lo stato in prospettiva sulla rispettiva maschera vuota

                    if (thresholded_img.at<uchar>(y_proj, x_proj) ==0 ) {           // if an army was found by thresholding
                        armies_projected_mask.at<uchar>(y_proj, x_proj) = 255;      // segna l'army in prospettiva sulla rispettiva maschera vuota
                        projected.at<uchar>(y_proj, x_proj) = 120;                  // segna in grigio lo stato sulla mappa colorata (per bellezza)
                    }
                }

            }
        }

        for (int r = 0; r<state_projected_mask.rows; r++) {
            for (int c = 0; c < state_projected_mask.cols; c++) {
                if (state_projected_mask.at<uchar>(r, c)  == 255 ) projected_state_area++;
                if (armies_projected_mask.at<uchar>(r, c) == 255 ) pix_cont++;
            }
        }

        cout << "PIX_CONT: " << pix_cont << endl;
        cout << "original_state_area : " << original_india_area << endl;
        cout << "projected_state_area: " << projected_state_area << endl;

        cout << "ratio: " << (float) (pix_cont)/(projected_state_area) << std::endl;

//        float k = 403;
//
//        cout << "num_armies: " << (float)pix_cont/k << endl;


        cv::resize(armies_projected_mask, armies_projected_mask, cv::Size(armies_projected_mask.cols/2, armies_projected_mask.rows/2));
        imshow("armies_projected_mask", armies_projected_mask );

        cv::resize(projected, projected, cv::Size(projected.cols/2, projected.rows/2));
        imshow("projected", projected );

        cv::resize(state_projected_mask, state_projected_mask, cv::Size(state_projected_mask.cols/2, state_projected_mask.rows/2));
        imshow("state_projected_mask", state_projected_mask );
       while(true){
           char key = (char) waitKey(30);
           if (key == 'q' || key == 27)
           {
               break;
           }
       }
    }

    void pipeline2() {
        Mat thresholded_img;
        Mat th_copy;
        Mat img_scene_hsv;
        int erode_size = 1;
//        cvtColor(img_scene_color, img_scene_hsv, COLOR_BGR2HSV);
////        inRange(img_scene_hsv, Scalar(108, 125, 38), Scalar(137, 233, 223), th);
//        inRange(img_scene_hsv, Scalar(107, 142, 68), Scalar(141, 255, 255), thresholded_img);
//        Mat element = getStructuringElement( MORPH_RECT,
//                                             Size( 2*erode_size + 1, 2*erode_size+1 ),
//                                             Point( erode_size, erode_size ) );
//        erode( thresholded_img, thresholded_img, element );
        thresholded_img = armies_projected_mask.clone();

        Mat thresh;
        threshold(thresholded_img, thresh, 0, 255, THRESH_OTSU + THRESH_BINARY);
        Mat kernel = getStructuringElement( MORPH_ELLIPSE, cv::Size(3,3));
        Mat opening;
        morphologyEx(thresh, opening,  MORPH_OPEN, kernel, Point( -1, -1 ), 5);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours( opening, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

        cout << "BLOBS: " << contours.size() << endl;

        cvtColor(opening, opening, COLOR_GRAY2BGR);
        for( size_t i = 0; i< contours.size(); i++ )
        {
            drawContours( opening, contours, (int)i, Scalar(0, 0, 255), 2, LINE_8, hierarchy, 0 );
        }

        resize(opening, opening, cv::Size(opening.cols/2, opening.rows/2));

        imshow("thresholded_img", opening);
        waitKey(0);
    }

};


int main(int argc, char** argv) {

    std::string object_path = "/home/fusy/Documents/risiko/imgs/rect_map_india.jpg";

//    Match m1("/home/fusy/Documents/risiko/imgs/IMG_20211214_110720.jpg", object_path);        // africadelnord has 1
    Match m3("/home/fusy/Documents/risiko/imgs/IMG_20211214_110800.jpg", object_path);        // africadelnord has 3
//    Match m4("/home/fusy/Documents/risiko/imgs/IMG_20211214_110943.jpg", object_path);        // africadelnord has 4);

    m3.pipeline2();

    return 0;
}