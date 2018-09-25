#include <iostream>
#include <dirent.h>

#include <Eigen/Core>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/pipeline/visual-pipeline-brisk.h>
#include <aslam/pipeline/visual-pipeline-freak.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/distortion-radtan.h>

#include "vi-map/vertex.h"
#include "vi-map/vi-mission.h"
#include "vi-map/vi-map.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <brisk/brisk.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(images_folder, "", "images ");

int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);

        if(name != "." && name != "..")
            files.push_back(name);
    }
    closedir(dp);


    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
    for(unsigned int i=0;i<files.size();i++)
    {
        if(files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    return files.size();
}

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    
    if (FLAGS_images_folder.empty()) {
        return -1;
    }
    
    std::vector<std::string> files;
    getdir(FLAGS_images_folder, files);
    aslam::PinholeCamera::Ptr pinhole_A = aslam::createCamera<aslam::PinholeCamera, aslam::RadTanDistortion>(
          Eigen::Vector4d(411,411,319,190),  // intrinsics
          640, 360,  //resolution
          Eigen::Vector4d(-0.3368, 0.1009, -0.0018, 3.8804e-04)); // distortion coeffs
    aslam::VisualPipeline::Ptr pipeline_;
    pipeline_.reset(new aslam::FreakVisualPipeline(pinhole_A, false, 4, 100, 3, true, true, 5));
    //pipeline_.reset(new aslam::BriskVisualPipeline(pinhole_A, false, 8, 5, 30, 1000, false, true));
    vi_map::VIMap* map= new vi_map::VIMap();
    
    aslam::Transformation trans;
    aslam::TransformationVector T_C_B;
    T_C_B.push_back(trans);
    aslam::NCameraId n_camera_id_;
    common::generateId(&n_camera_id_);
    std::vector<std::shared_ptr<aslam::Camera>> cameras;
    cameras.push_back(pinhole_A);
    aslam::NCamera::Ptr cameras_(new aslam::NCamera(n_camera_id_, T_C_B, cameras, "chamo"));
    
    vi_map::MissionId mission_id_;
    common::generateId(&mission_id_);
    map->addNewMissionWithBaseframe(
        mission_id_, aslam::Transformation(),
        Eigen::Matrix<double, 6, 6>::Identity(), cameras_,
        vi_map::Mission::BackBone::kViwls);
    
    float time_stamp=0;
    for (int i=0;i<files.size();i++){
        std::cout<<files[i]<<std::endl;
        cv::Mat img = cv::imread(files[i]);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        std::shared_ptr<aslam::VisualFrame> frame=pipeline_->processImage(img, time_stamp);
        std::cout<<frame->getKeypointMeasurement(0)<<std::endl;
        pose_graph::VertexId vertex_id_;
        common::generateId(&vertex_id_);
        aslam::NFramesId n_frame_id_;
        common::generateId(&n_frame_id_);
        std::cout<<vertex_id_<<std::endl;
        aslam::VisualNFrame::Ptr visual_n_frame(new aslam::VisualNFrame(n_frame_id_, cameras_));
        visual_n_frame->setFrame(0, frame);
        vi_map::Vertex* map_vertex = new vi_map::Vertex(vertex_id_, visual_n_frame, mission_id_);
        map->addVertex(vi_map::Vertex::UniquePtr(map_vertex));
        pose_graph::VertexIdList id_list;
        map->getAllVertexIds(&id_list);
        std::cout<<id_list.size()<<std::endl;
        time_stamp=time_stamp+0.1;
    }
    return 0;
}