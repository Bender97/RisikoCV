
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

class System {

    Mat rectified_map, scene_grayscale, scene_color;
    Mat rectified_state_gray, state_mask;

    // data for keypoints and descriptors extraction
    Ptr<SURF> detector;
    int minHessian = 400;
    std::vector<KeyPoint> keypoints_rectified_map, keypoints_scene;
    Mat descriptors_rectified_map, descriptors_scene;

    // data for features matching
    std::vector<DMatch> good_matches;
    Mat img_matches;
    std::vector<Point2f> obj_kp;
    std::vector<Point2f> scene_kp;

    // Homography matrix
    Mat H;

    // data for blobbing the armies in the scene
    Mat armies_in_scene_mask;
    Mat armies_in_state_mask;

    void loadImage(std::string &path, cv::Mat &img, ImreadModes mode ) {
        img = imread( path, mode );
        if ( img.empty()) {
            cout << "Could not open or find img at path " << path << endl;
            assert(false);
        }
    }

    void loadKeypointsAndDescriptors(cv::Mat &img, std::vector<KeyPoint> &kp, cv::Mat &desc) {
        detector->detectAndCompute( img, noArray(), kp, desc );
    }

    void findMatches() {
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors_rectified_map, descriptors_scene, knn_matches, 2 );
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
            obj_kp.push_back( keypoints_rectified_map[ good_matches[i].queryIdx ].pt );
            scene_kp.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }
    }

    void extractArmiesMaskFromScene() {
        Mat temp;
        cvtColor(scene_color, temp, COLOR_BGR2HSV);
//        inRange(img_scene_hsv, Scalar(108, 125, 38), Scalar(137, 233, 223), th);
        inRange(temp, Scalar(110, 135, 95), Scalar(145, 255, 220), temp);

        int dilate_size = 3;

        Mat element = getStructuringElement( MORPH_RECT,
                                             Size( 2*dilate_size + 1, 2*dilate_size+1 ),
                                             Point( dilate_size, dilate_size ) );
        dilate( temp, temp, element );

        armies_in_scene_mask = 255 - temp;
    }

    void show(cv::Mat &img) {
        Mat copy;
        resize(img, copy, cv::Size(img.cols/2, img.rows/2));
        imshow("prova", copy);
        waitKey(0);
    }

public:
    System(std::string rectified_map_path, std::string scene_path) {
        loadImage(rectified_map_path, rectified_map, cv::IMREAD_GRAYSCALE);
        loadImage(scene_path, scene_grayscale, IMREAD_GRAYSCALE);
        loadImage(scene_path, scene_color, IMREAD_COLOR);

        detector = SURF::create( minHessian );

        loadKeypointsAndDescriptors(rectified_map, keypoints_rectified_map, descriptors_rectified_map);
        loadKeypointsAndDescriptors(scene_grayscale, keypoints_scene, descriptors_scene);

        findMatches();

        H = findHomography( obj_kp, scene_kp, cv::RANSAC );

        extractArmiesMaskFromScene();
    }

    void findMapinScene() {

        std::vector<Point2f> obj_corners;
        std::vector<Point2f> scene_corners;
        obj_corners.resize(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)rectified_map.cols, 0 );
        obj_corners[2] = Point2f( (float)rectified_map.cols, (float)rectified_map.rows );
        obj_corners[3] = Point2f( 0, (float)rectified_map.rows );
        scene_corners.resize(4);
        perspectiveTransform( obj_corners, scene_corners, H);
        Mat map = scene_color.clone();

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( map, scene_corners[0] ,
              scene_corners[1] , Scalar(0, 255, 0), 4 );
        line( map, scene_corners[1] ,
              scene_corners[2] , Scalar( 0, 255, 0), 4 );
        line( map, scene_corners[2] ,
              scene_corners[3] , Scalar( 0, 255, 0), 4 );
        line( map, scene_corners[3] ,
              scene_corners[0] , Scalar( 0, 255, 0), 4 );

        show(map);
    }

    void projectState(std::string &rectified_state_path) {

        loadImage(rectified_state_path, rectified_state_gray, cv::IMREAD_GRAYSCALE);

//        Mat copy = scene_grayscale.clone();

        for (int r = 0; r<rectified_state_gray.rows; r++) {
            for (int c=0; c<rectified_state_gray.cols; c++) {
                if (rectified_state_gray.at<uchar>(r, c) == 255) {
                    std::vector<Point2f> camera_corners;
                    Point2f p(c, r);
                    camera_corners.push_back(p);
                    std::vector<Point2f> world_corners;
                    perspectiveTransform(camera_corners, world_corners, H);
                    int x_proj = (int) world_corners[0].x;
                    int y_proj = (int) world_corners[0].y;

                    scene_grayscale.at<uchar>(y_proj, x_proj) = 0;                // annerisci lo stato sulla mappa colorata (grayscale)
                }

            }
        }



        show(scene_grayscale);

    }

    void getArmiesInState(std::string &rectified_state_path) {
        loadImage(rectified_state_path, rectified_state_gray, cv::IMREAD_GRAYSCALE);

        armies_in_state_mask = Mat::zeros(scene_grayscale.rows, scene_grayscale.cols, CV_8UC1);

        for (int r = 0; r<rectified_state_gray.rows; r++) {
            for (int c=0; c<rectified_state_gray.cols; c++) {
                if (rectified_state_gray.at<uchar>(r, c) == 255) {
                    std::vector<Point2f> camera_corners;
                    Point2f p(c, r);
                    camera_corners.push_back(p);
                    std::vector<Point2f> world_corners;
                    perspectiveTransform(camera_corners, world_corners, H);
                    int x_proj = (int) world_corners[0].x;
                    int y_proj = (int) world_corners[0].y;

                    if (armies_in_scene_mask.at<uchar>(y_proj, x_proj) ==0 )
                        armies_in_state_mask.at<uchar>(y_proj, x_proj) = 255;                // annerisci lo stato sulla mappa colorata (grayscale)
                }

            }
        }
    }

    void countArmiesInState(std::string &rectified_state_path) {
        getArmiesInState(rectified_state_path);

        Mat thresh;
        threshold(armies_in_state_mask, thresh, 0, 255, THRESH_OTSU + THRESH_BINARY);
        Mat kernel = getStructuringElement( MORPH_ELLIPSE, cv::Size(3,3));
        Mat opening;
        morphologyEx(thresh, opening,  MORPH_OPEN, kernel, Point( -1, -1 ), 5);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours( opening, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

        cout << contours.size() << endl;

        cvtColor(opening, opening, COLOR_GRAY2BGR);
        for( size_t i = 0; i< contours.size(); i++ )
        {
            drawContours( opening, contours, (int)i, Scalar(0, 0, 255), 2, LINE_8, hierarchy, 0 );
        }

//        show(opening);
    }

    void showScene() {
        show(scene_color);
    }

};

int main(int argc, char **argv) {
    std::string rectified_map_path = "/home/fusy/Documents/risiko/imgs/rect_map.jpg";
    std::string scene_path = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110912.jpg";
    std::string rectified_state_mask = "/home/fusy/Documents/risiko/imgs/rect_map_africadelnord.jpg";

    std::vector<std::string> states;
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_afghanistan.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_africadelnord.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_africadelsud.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_africaorientale.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_alaska.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_alberta.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_americacentrale.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_argentina.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_australiaoccidentale.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_australiaorientale.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_brasile.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_cina.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_cita.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_congo.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_egitto.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_europacentrale.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_europaoccidentale.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_europasettentrionale.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_giappone.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_granbretagna.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_groenlandia.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_india.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_indonesia.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_islanda.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_jacutia.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_kamchatcka.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_madagascar.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_mediooriente.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_mongolia.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_nuovaguinea.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_ontario.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_peru.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_quebec.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_scandinavia.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_siam.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_siberia.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_statiunitioccidentali.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_statiunitiorientali.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_territoridelnordovest.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_ucraina.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_urali.jpg");
    states.push_back("/home/fusy/Documents/risiko/imgs/state_masks/rect_map_venezuela.jpg");

    System s(rectified_map_path, scene_path);

    for (auto state: states) {
        cout << state.substr(54) << " \t   ";
//        s.findMapinScene();
//            s.projectState(state);
        s.countArmiesInState(state);
    }

    s.showScene();

    return 0;
}