
#include <iostream>
#include <map>
#include <dirent.h>
#include <sys/stat.h>
#include <string>
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



    std::vector<std::string> states_path;



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
        inRange(temp, Scalar(107, 135, 45), Scalar(145, 255, 255), temp);

        int dilate_size = 1;

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


    void loadStatesMasksPaths() {
        DIR *dir;
        class dirent *ent;
        class stat st{};
        std::string directory = "../imgs/state_masks";

        dir = opendir(directory.c_str());
        while ((ent = readdir(dir)) != nullptr) {
            const string file_name = ent->d_name;
            const string full_file_name = directory + "/" + file_name;

            if (file_name[0] == '.')
                continue;

            if (stat(full_file_name.c_str(), &st) == -1)
                continue;

            const bool is_directory = (st.st_mode & S_IFDIR) != 0;

            if (is_directory)
                continue;

            states_path.push_back(full_file_name);
        }
        closedir(dir);
    }


public:

    // contouring
    std::map<int, vector<vector<Point>>> big_army_areas;
    vector<double> single_areas;
    double avg_army_area;
    std::vector<int> armies_per_state;

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
//        show(armies_in_scene_mask);
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

    int countArmiesInState(std::string &rectified_state_path, int idx) {
        getArmiesInState(rectified_state_path);

        Mat thresh;
        threshold(armies_in_state_mask, thresh, 0, 255, THRESH_OTSU + THRESH_BINARY);
        Mat kernel = getStructuringElement( MORPH_ELLIPSE, cv::Size(3,3));
        Mat opening;
        morphologyEx(thresh, opening,  MORPH_OPEN, kernel, Point( -1, -1 ), 5);
        vector<vector<Point> > contours;
        vector<vector<Point> > valid_contours;
        vector<Vec4i> hierarchy;
        findContours( opening, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

        vector<vector<Point>> lb;

        for( size_t i = 0; i< contours.size(); i++ )
        {
            double area = contourArea(contours[i]);
            if (area > 300) {
                if (area < 2700) {
                    valid_contours.push_back(contours[i]);
                    single_areas.push_back(area);
                }
                else {
                    lb.push_back(contours[i]);
                }
            }
        }

        if (lb.size()>0)
            big_army_areas.insert(std::pair<int, vector<vector<Point>>>(idx, lb));

        int armies = valid_contours.size();
//        if (armies>0) {
//            cout << rectified_state_path.substr(29) << " \t   " << armies << endl;
//            cout << "AREA: " << contourArea(valid_contours[0]) << endl;
//        }

//        cvtColor(opening, opening, COLOR_GRAY2BGR);
//        for( size_t i = 0; i< valid_contours.size(); i++ )
//        {
//            drawContours( opening, valid_contours, (int)i, Scalar(0, 0, 255), 2, LINE_8, hierarchy, 0 );
//        }

//        show(opening);
        return armies;
    }

    void showScene() {
        show(scene_color);
    }

    void computeAvgArmyArea() {
        double sum = 0;
        for (auto area: single_areas) sum += area;
        avg_army_area = sum / (double) single_areas.size();
    }

    void updateArmiesCount() {
        for (const auto &big_army_area: big_army_areas) {
            for (const auto big_contour: big_army_area.second) {
                double big_area = contourArea(big_contour);
                if (big_area>0) {
                    int armies = (int) round(big_area / avg_army_area);
                    armies_per_state[big_army_area.first] += armies;
                }
            }

        }
    }

    void count() {

        loadStatesMasksPaths();

        // EXTRACT THE NUMBER OF SINGLE ARMY PER STATE
        // STACK THE COMPLEX AREA TO BE SPLITTED (according to avg army size)
        for (int i=0; i<states_path.size(); i++) {
            armies_per_state.push_back(countArmiesInState(states_path[i], i));

            cout << "                                    \r";
            cout << "loading ... " << (i+1) << " / " << states_path.size() << "\r" << std::flush;

        }
        cout << endl;

        // COMPUTE AVERAGE SINGLE ARMY AREA SURFACE
        computeAvgArmyArea();
        cout << "average area: " << avg_army_area << endl;

        // UPDATE ARMIES WITH THE BIG ARMIES AREA FOUND (GROUPS OF ARMIES)
        updateArmiesCount();
    }

    void print() {
        cout << endl;
        for (int i=0; i<states_path.size(); i++) {
            int armies = armies_per_state[i];
            if (armies>0)
                cout << states_path[i].substr(29) << " " << armies << endl;
        }
    }
};




int main(int argc, char **argv) {
    std::string rectified_map_path = "/home/fusy/Documents/risiko/imgs/rect_map.jpg";
    std::string scene_path = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110927.jpg";
//    std::string scene_path = "/home/fusy/Documents/risiko/imgs/IMG_20211214_110912.jpg";



    System s(rectified_map_path, scene_path);

    s.count();

    s.print();

    s.showScene();



    return 0;
}