
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include<unordered_map>
#include<unordered_set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "camFusion.hpp"
#include "dataStructures.h"

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
        //cout<<xwmin<<endl;
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
   int count = 0;
   double dist = 0;
   for(auto  m_it = kptMatches.begin(); m_it!= kptMatches.end(); m_it++)
    {
        cv::KeyPoint prev_pt = kptsPrev[m_it->queryIdx];
        cv::KeyPoint curr_pt = kptsCurr[m_it->trainIdx];
        if(boundingBox.roi.contains(curr_pt.pt))
        {
            dist += cv::norm(curr_pt.pt - prev_pt.pt);
            count+=1;
        }
    }
    double dist_avg = dist / count;
    for(auto  m_it = kptMatches.begin(); m_it!= kptMatches.end(); m_it++)
    {
        cv::KeyPoint prev_pt = kptsPrev[m_it->queryIdx];
        cv::KeyPoint curr_pt = kptsCurr[m_it->trainIdx];
        if(boundingBox.roi.contains(curr_pt.pt))
        {
            double new_dist = cv::norm(curr_pt.pt - prev_pt.pt);
            if(new_dist<dist_avg*1.4)
            {
                boundingBox.kptMatches.push_back(*m_it);
            }
            
        }
    }


}
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    //(replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = - dT / (1 - medDistRatio);
    
  
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double minXprev = 1e9;
    double minXcurr = 1e9;
    std::vector<double> prev_x_pt;
    std::vector<double> curr_x_pt;
    
    for(auto lip = lidarPointsPrev.begin(); lip!= lidarPointsPrev.end(); ++lip)
    {
        if(minXprev > lip->x)
        {
            //minXprev = lip->x;
            prev_x_pt.push_back(lip->x);
        }
    }

    // METHOD -- 1
    // int size_prev = prev_x_pt.size();
    // if(size_prev%2 == 0)
    // {
    //     minXprev = (prev_x_pt[size_prev/2] + prev_x_pt[size_prev/2 + 1])/2;
    // }
    // else
    // {
    //     minXprev = prev_x_pt[(size_prev+1)/2];
    // }

    //METHOD 2---
    // double sum_prev = std::accumulate(prev_x_pt.begin(), prev_x_pt.end(),0);
    // minXprev = sum_prev/size_prev;
    
    //METHOD 3---
    std::vector<double> sorted_prev;
    std::sort(prev_x_pt.begin(), prev_x_pt.end());
    sorted_prev = prev_x_pt;
    minXprev = sorted_prev[sorted_prev.size()/5];


    for(auto cip = lidarPointsCurr.begin(); cip!= lidarPointsCurr.end(); ++cip)
    {
        if(minXcurr > cip->x)
        {
            //minXcurr = cip->x;
            curr_x_pt.push_back(cip->x);
        }
    }

    // 1 -----
    // int size_curr = curr_x_pt.size();
    // if(size_curr%2 == 0)
    // {
    //     minXcurr = (curr_x_pt[size_curr/2] + curr_x_pt[size_curr/2 + 1])/2;
    // }
    // else
    // {
    //     minXcurr = curr_x_pt[(size_curr+1)/2];
    // }

    //2------
    // double sum_curr = std::accumulate(curr_x_pt.begin(), curr_x_pt.end(),0);
    // minXcurr = sum_curr/size_curr;

    //3------
    std::vector<double> sorted_curr;
    std::sort(curr_x_pt.begin(),curr_x_pt.end());
    sorted_curr = curr_x_pt;
    minXcurr = sorted_curr[sorted_curr.size()/5];



    TTC = (minXcurr * (1/frameRate)) / (minXprev - minXcurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    unordered_map <int, std::vector<int>> BestMatches;
    unordered_set <int> prev_pt;
    std::vector<int> count;
    
    int num;
    for(auto mit = matches.begin(); mit!=matches.end(); mit++)
    {
        cv::KeyPoint p1 = prevFrame.keypoints[mit->queryIdx];
        cv::KeyPoint p2 = currFrame.keypoints[mit->trainIdx];

        for(auto bitp = prevFrame.boundingBoxes.begin(); bitp!=prevFrame.boundingBoxes.end(); bitp++)
        {
            if(bitp->roi.contains(p1.pt))   
            {  
                for(auto bitc= currFrame.boundingBoxes.begin(); bitc!=currFrame.boundingBoxes.end(); bitc++)
                {
                    if(bitc->roi.contains(p2.pt))
                    {
                        // BestMatches.insert({bitp->boxID,bitc->boxID}); //used an multimap to store the matched keypoint in both the frames 
                        BestMatches[bitp->boxID].push_back(bitc->boxID); //used an multimap to store the matched keypoint in both the frames 
                        prev_pt.insert(bitp->boxID);//using a separate set to keep track of the matched keypoint in the previous frame. 
                        //using a set because it doesn't store duplicates.
                    }
                }
            }
        }
    }   

    for (auto p_it = prev_pt.begin(); p_it!=prev_pt.end(); p_it++)
    {
        std::vector<int> list_currBoxID = BestMatches[*p_it];
        int max_frq = -1;
        int max_Curr_BoxID;
        for(auto lit = list_currBoxID.begin(); lit !=list_currBoxID.end(); lit++)
        {
            int freq = std::count(list_currBoxID.begin(), list_currBoxID.end(), *lit);
            if(freq > max_frq)
            {
                max_frq = freq;
                max_Curr_BoxID = *lit;
            }

        }
        bbBestMatches[*p_it] = max_Curr_BoxID;
        
    }  

}



