
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

#include <set>

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

    multimap<int, int> mmap{};
    
    // DMatch structure contains 2 keypoints indices : queryIdx and trainIdx which corresponds to the matched kpts found in the previous and the current frame.
    for(auto match : matches) {

        cv::KeyPoint previousKpt = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currentKpt = currFrame.keypoints[match.trainIdx];


        // If -1 occurs in a pair of bbox ids it means that a kpt has been found in only 1 bbox
        int previousBboxID = -1;
        int currentBboxID = -1;

        //Find which bbox in the previous frame contains our matched keypoint
        for(auto previousBbox : prevFrame.boundingBoxes) {
            if(previousBbox.roi.contains(previousKpt.pt)) {
                previousBboxID = previousBbox.boxID;
            }
        }

        //Find which bbox in the current frame contains our matched keypoint
        for(auto currentBbox : currFrame.boundingBoxes) {
            if(currentBbox.roi.contains(currentKpt.pt)) {
                currentBboxID = currentBbox.boxID;
            }
        }
        //TODO: Check if kp matches can be in multiple bboxes
        mmap.insert({currentBboxID, previousBboxID});
    }


    //For each bboxID in the current frame, count how many occurence of a matched bbox ID in the previous frame we have.
    for(auto it = currFrame.boundingBoxes.begin(); it != currFrame.boundingBoxes.end(); ++it) {

        //Get the list of all elements for a specific current bbox ID. equal_range return a pair of iterators {beginning of range, end of range}
        auto rangeCurrBoxID = mmap.equal_range(it->boxID);

        //By copying the multimap in a set we remove duplicate data and therefore get a list of all the unique pair possibilities (potential matches) for a specific boxID
        set<pair<int,int>> potentialMatches{rangeCurrBoxID.first , rangeCurrBoxID.second};
        //Remove pairs with value -1 because they correspond to kpt matches which don't belong to a bbox in the previous frame
        for(auto it = potentialMatches.begin(); it != potentialMatches.end(); ++it) {
            if(it->second == -1) potentialMatches.erase(it);
        }

        if(potentialMatches.empty()) {
            continue;
        }

        //First corresponds to a potential best match and second to its number of occurences in the multimap
        set<pair<pair<int,int>, int>> countingResults = {}; //multimap<pair<int,int>,int>???
        //Count occurences of each potential best match in the multimap
        for(auto potentialMatch : potentialMatches) {

            size_t counter = count_if(mmap.begin(), mmap.end(), [&potentialMatch](const pair<int,int>& p1) {return p1 == potentialMatch;});
            auto newCounterResult = make_pair(potentialMatch, counter);
            countingResults.insert(newCounterResult); // => ZEMKNVOUIREBHERIÀ)JERBJER if same value, keep highest

            
        }

        //Find the pair with the most occurences
        auto bestMatch = max_element(countingResults.begin(), countingResults.end(), [] (const pair<pair<int,int>, int>& p1, const pair<pair<int,int>, int>& p2) {return p1.second < p2.second;});
        cout << "1. Best match " << "{" << bestMatch->first.first << "," << bestMatch->first.second << "} " << bestMatch->second << endl;
        
        //Add it to the best matches
        bbBestMatches.insert(make_pair(bestMatch->first.second, it->boxID));
        
        
    }



    
    //for(auto res : countingResults) cout << "{" << res.first.first << "," << res.first.second << "} " << res.second << endl;
    for(auto res : bbBestMatches) cout << "Best match {" << res.first << "," << res.second << "} " << endl;

    

}
