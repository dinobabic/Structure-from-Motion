#pragma once
#include <eigen3/Eigen/Core>
#include <vector>
#include <string>

#include "sfm/utils.hpp"

class SfM 
{
public: 
    SfM(const std::string &dataset_path) : _dataset_path(dataset_path), K_cv(3, 3, CV_32F)
    {
        // read calibration matrix - assumes known calibration matrix and constant for all images
        K = read_calib_matrix(dataset_path + "/K.txt");

        // fill in K_cv with calibration matrix
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                K_cv.at<float>(r, c) = K(r, c);
    }

    void motion_2D2D(
        const cv::Mat &img1,
        const cv::Mat &img2,
        ImageDescription &img_description1,
        ImageDescription &img_description2,
        std::vector<Eigen::Vector3f> &_points3D,
        std::vector<Observation> &_observations,
        Eigen::Matrix3f &_R,
        Eigen::Vector3f &_t)
    { 
        detect_and_match(img1, img2, img_description1, img_description2);
        
        std::vector<cv::Point2f> kps1, kps2;
        std::vector<int> kps_idx1, kps_idx2;
        get_matches_from_descriptions(img_description1, img_description2, kps1, kps2, kps_idx1, kps_idx2);

        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(
            kps1, kps2, K_cv,
            cv::RANSAC, 0.999, 2.0, inlier_mask);

        filter_matches(img_description1, img_description2, kps_idx1, kps_idx2, inlier_mask);

        kps1.clear(); kps2.clear(); kps_idx1.clear(); kps_idx2.clear();
        get_matches_from_descriptions(img_description1, img_description2, kps1, kps2, kps_idx1, kps_idx2);
        cv::Mat R, t;
        inlier_mask = cv::Mat();
        cv::recoverPose(E, kps1, kps2, K_cv,  R, t, inlier_mask);
        R.convertTo(R, CV_32F);
        t.convertTo(t, CV_32F);
        
        filter_matches(img_description1, img_description2, kps_idx1, kps_idx2, inlier_mask);
        
        kps1.clear(); kps2.clear(); kps_idx1.clear(); kps_idx2.clear();
        get_matches_from_descriptions(img_description1, img_description2, kps1, kps2, kps_idx1, kps_idx2);
        std::vector<cv::Point2f> pts1_norm, pts2_norm;
        cv::undistortPoints(kps1, pts1_norm, K_cv, cv::Mat());
        cv::undistortPoints(kps2, pts2_norm, K_cv, cv::Mat());

        cv::Mat P1 = cv::Mat::eye(3,4,CV_32F);
        cv::Mat P2;
        cv::hconcat(R, t, P2);

        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, pts1_norm, pts2_norm, points4D);

        for (size_t i = 0; i < points4D.cols; i++)
        {
            cv::Mat x = points4D.col(i);
            x /= x.at<float>(3);
            
            // find stored match in both image descriptions for the given keypoint pair
            auto old_match_img1 = *img_description1.matches.find({kps_idx1[i], kps_idx2[i], img_description2.img_id});
            img_description1.matches.erase(old_match_img1);
            auto old_match_img2 = *img_description2.matches.find({kps_idx2[i], kps_idx1[i], img_description1.img_id});
            img_description2.matches.erase(old_match_img2);

            if (x.at<float>(2) <= 0) continue;

            _points3D.emplace_back(
                x.at<float>(0),
                x.at<float>(1),
                x.at<float>(2)
            );

            old_match_img1.point3D_idx = _points3D.size()-1;
            img_description1.matches.insert(old_match_img1);
            old_match_img2.point3D_idx = _points3D.size()-1;
            img_description2.matches.insert(old_match_img2);

            auto pt1 = img_description1.pts[old_match_img1.kp_idx_this];
            auto pt2 = img_description2.pts[old_match_img2.kp_idx_this];

            _observations.push_back({img_description1.img_id, _points3D.size()-1, kps_idx1[i], Eigen::Vector2f(pt1.x, pt1.y)});
            _observations.push_back({img_description2.img_id, _points3D.size()-1, kps_idx2[i], Eigen::Vector2f(pt2.x, pt2.y)});
        }

        // convert cv pose to Eigen pose
        cv_pose_to_eigen_pose(R, t, _R, _t);
    }

    void motion_3D2D(
        const cv::Mat &img,
        ImageDescription &img_description,
        ImageDescription &img_description_prev,
        std::vector<Eigen::Vector3f> &_points3D,
        std::vector<Observation> &_observations,
        Eigen::Matrix3f &_R,
        Eigen::Vector3f &_t)
    {
        detect_features(img, img_description);
        match_features_pnp(img_description_prev, img_description);

        std::vector<cv::Point3f> pts3D;
        std::vector<cv::Point2f> pts2D;
        std::vector<int> pts3D_ids, pts2D_ids, pts2D_other_ids;

        for (const auto &m : img_description.matches)
        {
            auto pt3D = _points3D[m.point3D_idx];
            pts3D.push_back({pt3D[0], pt3D[1], pt3D[2]});
            pts2D.push_back(img_description.pts[m.kp_idx_this]); 
            pts3D_ids.push_back(m.point3D_idx);
            pts2D_ids.push_back(m.kp_idx_this);
            pts2D_other_ids.push_back(m.kp_idx_other);
        }

        std::cout << "The number of matches in PnP is: " << pts2D.size() << std::endl;

        cv::Mat R, R_vec, t, mask;
        bool success = cv::solvePnPRansac(
            pts3D,
            pts2D,
            K_cv,
            cv::Mat(),
            R_vec,
            t,
            false,
            100,
            8.0,
            0.99, 
            mask,
            cv::SOLVEPNP_ITERATIVE
        );
        R_vec.convertTo(R_vec, CV_32F);
        t.convertTo(t, CV_32F);

        if (!success)
        {
            std::cerr << "[PnP] Failed or too few inliers\n";
            _R.setIdentity();
            _t.setZero();
            return;
        }

        for (int i = 0; i < pts3D.size(); i++)
        {
            if (mask.at<uchar>(i))
                _observations.push_back(Observation(img_description.img_id, pts3D_ids[i], pts2D_ids[i], Eigen::Vector2f(pts2D[i].x, pts2D[i].y)));
            else
            {
                // remove this match
                auto old_match_img = *img_description.matches.find({pts2D_ids[i], pts2D_other_ids[i], img_description_prev.img_id});
                img_description.matches.erase(old_match_img);
            }
        }

        cv::Rodrigues(R_vec, R);
        R.convertTo(R, CV_32F);
    
        // convert cv pose to Eigen pose
        cv_pose_to_eigen_pose(R, t, _R, _t);
    }   

    void triangulate_new_matches(ImageDescription &img_description1,
                                 ImageDescription &img_description2,
                                 const Sophus::SE3f &_P1, const Sophus::SE3f &_P2,
                                 std::vector<Eigen::Vector3f> &points3D,
                                 std::vector<Observation> &observations)
    {
        match_features_triangulation(img_description1, img_description2);
        
        cv::Mat points4D;
        cv::Mat P1, P2;
        sophus_pose_to_cv(_P1, P1);
        sophus_pose_to_cv(_P2, P2);

        std::vector<cv::Point2f> kps1, kps2;
        std::vector<int> kps_idx1, kps_idx2;
        for (const auto &m : img_description1.matches)
        {
            if (m.point3D_idx == -1)
            {
                kps1.push_back(img_description1.pts[m.kp_idx_this]);
                kps2.push_back(img_description2.pts[m.kp_idx_other]);
                kps_idx1.push_back(m.kp_idx_this);
                kps_idx2.push_back(m.kp_idx_other);
            }
        }

        std::vector<cv::Point2f> pts1_norm, pts2_norm;
        cv::undistortPoints(kps1, pts1_norm, K_cv, cv::Mat());
        cv::undistortPoints(kps2, pts2_norm, K_cv, cv::Mat());

        std::cout << "The number of new points that will be triangulated is: " << kps1.size() << std::endl;

        cv::triangulatePoints(P1, P2, pts1_norm, pts2_norm, points4D);

        int success_count = 0;
        int failed_depth_count = 0;
        int failed_reprojection_count = 0;
        for (size_t i = 0; i < points4D.cols; i++)
        {
            cv::Mat x = points4D.col(i);
            x /= x.at<float>(3);

            auto old_match_img1 = *img_description1.matches.find({kps_idx1[i], kps_idx2[i], img_description2.img_id});
            img_description1.matches.erase(old_match_img1);
            auto old_match_img2 = *img_description2.matches.find({kps_idx2[i], kps_idx1[i], img_description1.img_id});
            img_description2.matches.erase(old_match_img2);

            Eigen::Vector3f point(x.at<float>(0), x.at<float>(1), x.at<float>(2));

            // Cheirality: must be in front of both cameras
            Eigen::Vector3f p1_cam = _P1 * point;
            Eigen::Vector3f p2_cam = _P2 * point;
            if (p1_cam(2) <= 0 || p2_cam(2) <= 0) 
            {
                failed_depth_count++;
                continue;
            }

            Eigen::Vector2f kp1(pts1_norm[i].x, pts1_norm[i].y);
            Eigen::Vector2f kp2(pts2_norm[i].x, pts2_norm[i].y);

            Eigen::Vector3f proj1_pixel = (p1_cam / p1_cam(2));
            Eigen::Vector2f reproj1(proj1_pixel(0), proj1_pixel(1));
            
            Eigen::Vector3f proj2_pixel = (p2_cam / p2_cam(2));
            Eigen::Vector2f reproj2(proj2_pixel(0), proj2_pixel(1));

            float reproj_thresh = 0.02f; 
            if (i % 20 == 0)
                    std::cout << "The error is " << (reproj1 - kp1).norm() << std::endl;
            if ((reproj1 - kp1).norm() > reproj_thresh || (reproj2 - kp2).norm() > reproj_thresh)
            {
                failed_reprojection_count++;
                continue;
            }
                
            points3D.emplace_back(point);
            success_count++;

            old_match_img1.point3D_idx = points3D.size()-1;
            img_description1.matches.insert(old_match_img1);
            old_match_img2.point3D_idx = points3D.size()-1;
            img_description2.matches.insert(old_match_img2);

            auto pt1 = img_description1.pts[old_match_img1.kp_idx_this];
            auto pt2 = img_description2.pts[old_match_img2.kp_idx_this];

            observations.push_back({img_description1.img_id, points3D.size()-1, kps_idx1[i], Eigen::Vector2f(pt1.x, pt1.y)});
            observations.push_back({img_description2.img_id, points3D.size()-1, kps_idx2[i], Eigen::Vector2f(pt2.x, pt2.y)});
        }

        std::cout << "Out of all the triangulated points, only this many has been added: " << success_count << std::endl; 
        std::cout << "Failed depth count: " << failed_depth_count << ", Failed reprojection count: " << failed_reprojection_count << std::endl;
    }


private:
    std::string _dataset_path;
    Eigen::Matrix3f K;
    cv::Mat K_cv;

    void recover_pose( const std::vector<cv::Point2f>& pts1, 
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& P1,
                              const cv::Mat& P2,
                              std::vector<cv::Point3f>& points3D,
                              cv::Mat& mask)
    {
        
        cv::Mat points4D; // triangulated 3D points - homogeneous representation
        cv::triangulatePoints(P1, P2, pts1, pts2, points4D);
        mask = cv::Mat::zeros(points4D.cols, 1, CV_8U);
        for (size_t i = 0; i < points4D.cols; i++)
        {
            cv::Mat point4D = points4D.col(i);
            point4D /= point4D.at<float>(3);
            cv::Point3f point3D(point4D.at<float>(0), point4D.at<float>(1), point4D.at<float>(2)); // point 3D in the first camera
            cv::Mat point3D_hom = (cv::Mat_<float>(4,1) << point3D.x, point3D.y, point3D.z, 1.0f);
            cv::Mat point3D_second_camera = P2 * point3D_hom; // point 3D in the second camera
            if (point3D.z > 0 && point3D_second_camera.at<float>(2) > 0) // valid points are those that lie in front of both cameras - depth is positive
            {
                points3D.push_back(point3D);
                mask.at<uchar>(i) = 1;
            }
        }

    }
};