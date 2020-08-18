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
    vector<double> distances{};

    for(auto match : kptMatches) {
        if(boundingBox.roi.contains(kptsCurr.at(match.trainIdx).pt)){
            boundingBox.kptMatches.push_back(match);
            distances.push_back(match.distance);
        } 
    }

    //Compute a mean of the euclidean distance of all the associated matches
    double meanDistance = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

    //Remove matches with euclidean distance above 150% or below 50% of the mean
    boundingBox.kptMatches.erase(remove_if(boundingBox.kptMatches.begin(), boundingBox.kptMatches.end(), 
        [meanDistance](const cv::DMatch& match){return match.distance < 0.5*meanDistance || match.distance > 1.5*meanDistance;}),
        boundingBox.kptMatches.end());
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}

void filterOutliers(vector<LidarPoint>& lidarPoints, float clusterTolerance, int minSize) {

    //Convert Lidar points to pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(auto p : lidarPoints) cloud->push_back(pcl::PointXYZ(p.x,p.y,p.z));

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    //We assume that there is only one cluster possible if we filter noise
    lidarPoints.clear();
    for(auto cluster : cluster_indices) {
        for(auto index : cluster.indices) {
            lidarPoints.push_back(LidarPoint{cloud->points.at(index).x,cloud->points.at(index).y,cloud->points.at(index).z});
        }
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    //Filter outliers using nearest neighboor method (clustering) with KD-tree data structure (Inspired from the sensor fusion Lidar course)
    //Another way to remove outlier could be by  only using the median point regarding the x value after sorting the pointcloud
    filterOutliers(lidarPointsCurr,0.05,5);
    filterOutliers(lidarPointsPrev,0.05,5);
    
    double minXprev = 1e9;
    double minXcurr = 1e9;

    for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it) minXprev = minXprev > it->x ? it->x : minXprev;
    for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it) minXcurr = minXcurr > it->x ? it->x : minXcurr;

    TTC = minXcurr * (1.0 / frameRate) / minXcurr - minXprev;
    cout << "TTC : " << TTC << " sec" << endl;
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

        mmap.insert({currentBboxID, previousBboxID});
    }

    //Hold memory of current best match of previous bbox ID. 
    //Key = Current previous bbox ID used in a pair in bbBestMatches, value = Count used for this pair stored in bbBestMatches
    map<int,int> processedPreviousBboxCount{}; 

    //For each bboxID in the current frame, count how many occurences of a matched bbox ID in the previous frame we have.
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
        set<pair<pair<int,int>, int>> countingResults = {}; 

        
        //Count occurences of each potential best match in the multimap
        for(auto potentialMatch : potentialMatches) {

            size_t counter = count_if(mmap.begin(), mmap.end(), [&potentialMatch](const pair<int,int>& p1) {return p1 == potentialMatch;});
            auto newCounterResult = make_pair(potentialMatch, counter);
            countingResults.insert(newCounterResult); 

        }

        //Find the pair with the most occurences
        auto potentialBestMatch = max_element(countingResults.begin(), countingResults.end(), [] (const pair<pair<int,int>, int>& p1, const pair<pair<int,int>, int>& p2) {return p1.second < p2.second;});


        //Check if we already processed a match using this previous bbox ID
        auto processedPrevBboxIDit = processedPreviousBboxCount.find(potentialBestMatch->first.second);
        if(processedPrevBboxIDit != processedPreviousBboxCount.end()) {
            //The previousBboxID has aleady a match, check if the new one is better
            if(potentialBestMatch->second > processedPrevBboxIDit->second) {
                //A better match has been found, update processedPreviousBboxCount and bbBestMatches 
                auto bbBestMatchesIt = bbBestMatches.find(potentialBestMatch->first.second);
                if(bbBestMatchesIt != bbBestMatches.end()) bbBestMatchesIt->second = it->boxID;
                processedPrevBboxIDit->second = potentialBestMatch->second;
            }

        }
        else {
            //This previousBboxID hasn't been processed yet, add it to bbBestMatches
            processedPreviousBboxCount.insert(make_pair(potentialBestMatch->first.second, potentialBestMatch->second));
            bbBestMatches.insert(make_pair(potentialBestMatch->first.second, it->boxID));
        }      
    }
}
