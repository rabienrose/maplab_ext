#include <iostream>
#include <dirent.h>

#include <Eigen/Core>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/feature-track.h>
#include <aslam/pipeline/visual-pipeline-brisk.h>
#include <aslam/pipeline/visual-pipeline-freak.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/matcher/match.h>
#include "aslam/matcher/match-visualization.h"
#include <aslam/matcher/matching-engine-exclusive.h>
#include <aslam/matcher/matching-problem-frame-to-frame.h>
#include <aslam/tracker/track-manager.h>
#include "feature-tracking/feature-track-extractor.h"
#include <aslam/triangulation/test/triangulation-fixture.h>
#include "vi-map/landmark.h"
#include <visualization/viwls-graph-plotter.h>
#include <visualization/rviz-visualization-sink.h>

#include "vi-map/vertex.h"
#include "vi-map/vi-mission.h"
#include "vi-map/vi-map.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp> 

#include <brisk/brisk.h>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(images_folder, "", "images ");
DEFINE_string(traj_addr, "", "traj ");

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

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    
    visualization::RVizVisualizationSink::init();
    
    if (FLAGS_images_folder.empty()) {
        return -1;
    }
    if (FLAGS_traj_addr.empty()) {
        return -1;
    }
    int scale_rate=2;
    std::ifstream infile(FLAGS_traj_addr);
    aslam::PinholeCamera::Ptr pinhole_A = aslam::createCamera<aslam::PinholeCamera, aslam::RadTanDistortion>(
          Eigen::Vector4d(1418/scale_rate,1418/scale_rate,975/scale_rate,611/scale_rate),  // intrinsics
          1920/scale_rate, 1200/scale_rate,  //resolution
          Eigen::Vector4d(-0.138302, 0.095746, 0.000033, 0.003568)); // distortion coeffs
    aslam::VisualPipeline::Ptr pipeline_;
    pipeline_.reset(new aslam::FreakVisualPipeline(pinhole_A, false, 8, 5000, 3, true, true, 5));
    //pipeline_.reset(new aslam::BriskVisualPipeline(pinhole_A, false, 8, 5, 30, 1000, false, true));
    vi_map::VIMap* map= new vi_map::VIMap();
    
    aslam::Transformation trans(Eigen::Vector3d(0,0,0), kindr::minimal::RotationQuaternion(Eigen::Vector3d(0,0,0)));
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
        mission_id_, trans,
        Eigen::Matrix<double, 6, 6>::Identity(), cameras_,
        vi_map::Mission::BackBone::kOdometry);
    
    std::vector<Eigen::Matrix4d> pose_list;
    std::string line;
    std::getline(infile, line);
    while (true)
    {
        std::getline(infile, line);
        if (line==""){
            break;
        }
        std::vector<std::string> splited = split(line, " ");
        Eigen::Vector3d posi;
        posi(0)=atof(splited[2].c_str());
        posi(1)=atof(splited[3].c_str());
        posi(2)=atof(splited[4].c_str());
        Eigen::Vector3d rpy;
        rpy(0)=atof(splited[5].c_str());
        rpy(1)=atof(splited[6].c_str());
        rpy(2)=atof(splited[7].c_str());
        
        Eigen::AngleAxisd rollAngle(rpy(0), Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(rpy(1), Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(rpy(2), Eigen::Vector3d::UnitZ());

        Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;

        Eigen::Matrix3d rotationMatrix = q.matrix();
        Eigen::Matrix4d pose=Eigen::Matrix4d::Identity();
        pose.block<3,3>(0,0)=rotationMatrix;
        pose.block<3,1>(0,3)=posi;
        pose_list.push_back(pose);
    }
    pose_graph::VertexId last_ver_id;
    for (int c = 0; c < pose_list.size(); ++c) {
        pose_graph::VertexId vertex_id_;
        common::generateId(&vertex_id_);
        aslam::VisualNFrame::Ptr nframe= aslam::VisualNFrame::createEmptyTestVisualNFrame(cameras_, c);
        vi_map::Vertex* map_vertex = new vi_map::Vertex(vertex_id_, nframe, mission_id_);
        Eigen::Matrix4d pose= pose_list[c];
        Eigen::Matrix3d rot=pose.block<3,3>(0,0);
        aslam::Transformation T_G_B(pose.block<3,1>(0,3), kindr::minimal::RotationQuaternion(rot));
        map_vertex->set_T_M_I(T_G_B);
        map->addVertex(vi_map::Vertex::UniquePtr(map_vertex));
        if (c==0){
            map->getMission(mission_id_).setRootVertexId(vertex_id_);
        }else{
            pose_graph::EdgeId edge_id_;
            common::generateId(&edge_id_);
            vi_map::TransformationEdge* map_edge = new vi_map::TransformationEdge(vi_map::Edge::EdgeType::kOdometry, edge_id_, last_ver_id, vertex_id_,T_G_B ,Eigen::Matrix<double, 6, 6>::Identity());
            map->getVertex(last_ver_id).addOutgoingEdge(edge_id_);
            map_vertex->addIncomingEdge(edge_id_);
            map->addEdge(vi_map::Edge::UniquePtr(map_edge));
        }
        last_ver_id=vertex_id_;
    }  

    std::string map_folder = "/home/chamo/Documents/temp_map";
    backend::SaveConfig overwrite_files;
    overwrite_files.overwrite_existing_files=true;
    map->saveToFolder(map_folder, overwrite_files);
    
    //visualization::ViwlsGraphRvizPlotter plotter_;
    //plotter_.visualizeMap(*map);
    //ros::spin();
    return 0;
}